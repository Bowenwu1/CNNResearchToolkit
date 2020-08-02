import os
import sys
import torch
from tqdm import tqdm
from box import Box
# from apex import amp, optimizers
from data import custom_create_dataloaders
from models import custom_create_model_wrapper
from utils import (
    create_lr_scheduler,
    create_optimizer,
    adjust_learning_rate,
    read_process_config,
    seed_torch,
)
from reporter import Reporter


def train_epoch(
    model_wrapper, dataloader_train, optimizer, apex, batch_accumulate=1
):
    optimizer.zero_grad()
    model_wrapper.model.train()

    loss_total = 0
    correct_total = 0
    total = 0

    for iter_in_epoch, sample in enumerate(tqdm(dataloader_train, leave=False)):
        loss, num, correct = model_wrapper.get_loss(sample)

        loss_total += loss.item()
        correct_total += correct
        total += num

        loss = loss / batch_accumulate
        if apex != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (iter_in_epoch + 1) % batch_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
    return loss_total / total, correct_total / total


def train(config, reporter):
    seed_torch(config.seed)
    dataloader_train, dataloader_test = custom_create_dataloaders(config.data)
    model_wrapper = custom_create_model_wrapper(config.model)
    model_wrapper.to_gpus(*config.gpu_ids)
    optimizer = create_optimizer(
        model_wrapper._core_model.parameters(), config.optimizer
    )
    lr_scheduler = create_lr_scheduler(
        optimizer, config.lr_scheduler, config.epoch
    )

    # apex
    if config.apex == "O1":
        model_wrapper.model, optimizer = amp.initialize(
            model_wrapper._core_model, optimizer, opt_level=config.apex
        )

    # warmup
    if config.warmup.epoch != 0:
        lr_step = (
            config.warmup.final_lr - config.warmup.init_lr
        ) / config.warmup.epoch
        adjust_learning_rate(optimizer, config.warmup.init_lr)

    for epoch in range(1, config.epoch + config.warmup.epoch + 1):
        reporter.log_metric("lr", optimizer.param_groups[0]["lr"], epoch)

        train_loss, train_acc = train_epoch(
            model_wrapper,
            dataloader_train,
            optimizer,
            config.apex,
            batch_accumulate=config.data.batch_accumulate,
        )

        reporter.log_metric("train_acc", train_acc, epoch)
        reporter.log_metric("train_loss", train_loss, epoch)
        reporter.log_param(model_wrapper.model, epoch)

        # lr schedule
        if epoch <= config.warmup.epoch:
            # warmup stage
            lr = config.warmup.init_lr + epoch * lr_step
            print("==> Warm Up epoch={} lr={}".format(epoch, lr))
            adjust_learning_rate(optimizer, lr)
        else:
            lr_scheduler.step()

        # eval
        if epoch % config.eval_interval == 0:
            scores = model_wrapper.get_eval_scores(dataloader_test)
            print("==> Evaluation: Epoch={} Acc={}".format(epoch, str(scores)))
            reporter.log_metric("eval_acc", scores["accuracy"], epoch)
            reporter.log_metric("eval_loss", scores["loss"], epoch)

        save_checkpoints(
            scores["accuracy"],
            model_wrapper._core_model,
            reporter,
            config.exp_name,
            epoch,
        )
        print("==> Training epoch %d" % epoch)


def save_checkpoints(acc, model, reporter, exp_name, epoch):
    if not hasattr(save_checkpoints, "best_acc"):
        save_checkpoints.best_acc = 0

    state_dict = model.state_dict()
    reporter.save_checkpoint(
        state_dict, "{}_latest.pth".format(exp_name), epoch
    )
    if acc > save_checkpoints.best_acc:
        reporter.save_checkpoint(
            state_dict, "{}_best.pth".format(exp_name), epoch
        )
        save_checkpoints.best_acc = acc
    if epoch % config.save_interval == 0:
        reporter.save_checkpoint(
            state_dict, "{}_{}.pth".format(exp_name, epoch), epoch
        )


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = read_process_config(filename=config_path)
    reporter = Reporter(config)
    reporter.log_config(config_path)
    reporter.log_config(sys.argv[0])
    train(config, reporter)
