import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import shutil
from utils import LabelSmoothingCrossEntropy, LabelSmoothingLoss


class ModelWrapper:
    def __init__(self, opt, model):
        super(ModelWrapper, self).__init__()
        self.__model = model
        if opt.label_smoothing > 0:
            self._criterion = LabelSmoothingCrossEntropy(opt.label_smoothing)
        elif opt.eps > 0:
            print('#'*20, 'use LabelSmoothingLoss')
            self._criterion = LabelSmoothingLoss(opt.eps)
        else:
            self._criterion = nn.CrossEntropyLoss()
        self.opt = opt

        # load_checkpoint
        if hasattr(opt, 'checkpoint_path'):
            print('load checkpoint from {}'.format(opt.checkpoint_path))
            state_dict = torch.load(opt.checkpoint_path, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            self.__model.load_state_dict(state_dict)
    
    def forward(self, x):  # test forward
        if type(x[0]) == dict:
            x = x[0]["data"]
        else:
            x, _ = x

        self.model.eval()
        x = x.to(self.device)
        out = self.model(x)

        return out

    def to_gpus(self, *gpu_ids):
        """
        Apply torch.nn.DataParallel() to `self.model`, do not override this method.
        Code used for creating runner could be as follows::
            gpu_ids = (0,)
            runner = ResnetRunner('./data/cifar10')
            runner.load_checkpoint(checkpoint_file)
            runner.to_gpus(gpu_ids)
        Then your model could run on GPU.

        :param tuple gpu_ids: For example, `(0,1,2,3)`. If you want to pass single GPU, then it should be like `(0,)`
        """
        if not gpu_ids:
            gpu_ids = (0,)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)[1:-1]
        if torch.cuda.is_available():
            # torch.backends.cudnn.benchmark = True
            self.model = self.model.to("cuda")
        if len(gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model)

    def get_loss(self, inputs):
        device = self.device

        self.model.train()
        images, targets = inputs
        images, targets = images.to(device), targets.to(device)
        out = self.model(images)
        loss = self._criterion(out, targets)

        _, predicted = out.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        return loss, total, correct

    def get_eval_scores(self, dataloader_test):
        from tqdm import tqdm
        device = self.device

        total = 0
        correct = 0
        total_loss = 0
        self.model.eval()
        # self.model.reset()
        # self.model.train()

        with torch.no_grad():
            for _, sample in enumerate(
                tqdm(dataloader_test, leave=False, desc="evaluating accuracy")
            ):
                outputs = self.forward(sample)
                _, predicted = outputs.max(1)
                targets = sample[1].to(device)

                loss = self._criterion(outputs, targets)
                total_loss += loss.item()

                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        scores = {"accuracy": acc, "loss": total_loss / total}

        return scores

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def _core_model(self):
        """
        Function to get the original `model` you want to compress. The returned `model` is forced to remove `torch.nn.DataParallel`.

        :returns: torch.nn.Module
        """
        return getattr(self.__model, "module", self.__model)
    
    @property
    def device(self):
        return next(self._core_model.parameters()).device

    def load_checkpoint(self, checkpoint_file):
        checkpoint = checkpoint = torch.load(
            checkpoint_file, map_location="cpu"
        )
        # support mutiple format checkpoint file
        if "net" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self._core_model.load_state_dict(checkpoint)
        print("Loaded resume checkpoint from", checkpoint_file)

    def save(self, checkpoint_file, epoch=0):
        """
        Function to save checkpoint.

        :param str checkpoint_file: path to where you want to locate checkpoint file
        """
        checkpoint = {
            "state_dict": self._core_model.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, checkpoint_file)
