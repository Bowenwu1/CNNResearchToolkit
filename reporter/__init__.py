import os
import os.path as osp
from datetime import datetime
import shutil
from tensorboardX import SummaryWriter
import torch
from utils import write_param


def _check_mk_path(path):
    if not osp.exists(path):
        os.makedirs(path)


class Reporter:
    def __init__(self, opt):
        now = datetime.now().strftime("-%Y-%m-%d-%H:%M:%S")

        self.log_dir = osp.join(opt.log_dir, opt.exp_name + now)
        _check_mk_path(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)

        self.ckpt_log_dir = osp.join(self.log_dir, "checkpoints")
        _check_mk_path(self.ckpt_log_dir)

        self.config_log_dir = osp.join(self.log_dir, "config")
        _check_mk_path(self.config_log_dir)

    def log_config(self, path):
        target = osp.join(self.config_log_dir, path.split("/")[-1])
        shutil.copyfile(path, target)

    def get_writer(self):
        return self.writer

    def log_metric(self, key, value, step):
        self.writer.add_scalar("data/" + key, value, step)

    def log_text(self, msg):
        print(msg)
    
    def log_param(self, model, step, lp=1):
        write_param(model, self.writer, step, lp=lp)

    def save_checkpoint(self, state_dict, ckpt_name, epoch=0):
        checkpoint = {"state_dict": state_dict, "epoch": epoch}
        torch.save(checkpoint, osp.join(self.ckpt_log_dir, ckpt_name))
