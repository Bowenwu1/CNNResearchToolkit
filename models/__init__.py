import os
import torch
import importlib
from .wrapper import ModelWrapper


def custom_create_model_wrapper(opt):
    model_name = "models." + opt.arch
    print(model_name)
    model_lib = importlib.import_module(model_name)
    for name, method in model_lib.__dict__.items():
        if "wrapper" in name.lower() and method != ModelWrapper:
            print("#" * 5, " {} is imported ".format(name), "#" * 5)
            return method(opt)
