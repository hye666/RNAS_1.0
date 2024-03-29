""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
import torchattacks
import torch.nn.functional as F

config = SearchConfig()
device = torch.device("cuda")


logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("cell searching start")

    torch.cuda.set_device(config.gpus[0])
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)

    # thetas optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training
    best_top1 = 0.
    for epoch in range(config.epochs):
        lr = lr_scheduler.get_last_lr()
        model.print_alphas(logger)

        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)
        lr_scheduler.step()

        # validation
        top1 = validate(valid_loader, model, epoch)

        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.train()

    # attack
    atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7)
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        model.net.F = []

        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        # adversarial training examples
        adv_trn_X = atk(trn_X, trn_y)

        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        # adversarial validation examples
        adv_val_X = atk(val_X, val_y)

        N = trn_X.size(0)

        if epoch < 15:
            w_optim.zero_grad()
            logits = model(adv_trn_X)
            loss = model.criterion(logits, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
            w_optim.step()

        if epoch >= 15:
            alpha_optim.zero_grad()
            architect.unrolled_backward(adv_trn_X, trn_y, adv_val_X, val_y, lr, w_optim)

            if step < (0.5 * len(train_loader)):
                alpha_optim.step()
                w_optim.zero_grad()
                logits = model(adv_trn_X)
                loss = model.criterion(logits, trn_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
                w_optim.step()

            else:
                model.net.F = []
                adv_logits = model(adv_val_X)
                feature_adv = model.net.F
                # print(len(feature_adv))

                model.net.F = []
                clean_logits = model(val_X)
                feature_clean = model.net.F
                # print(len(feature_clean))
                # print(feature_clean)

                KL = []
                for i in range(len(feature_adv)-1):
                    kl = F.kl_div(feature_adv[i].softmax(dim=-1).log(), feature_clean[i].softmax(dim=-1), reduction='mean')
                    KL.append(kl.item())
                v = np.mean(KL)

                architect.loss = architect.loss + 0.0005 * v
                alpha_optim.step()
                w_optim.zero_grad()
                logits = model(adv_trn_X)
                loss = model.criterion(logits, trn_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
                w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        logger.info(
            "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
            "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                top1=top1, top5=top5))

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)


            logger.info(
                "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
