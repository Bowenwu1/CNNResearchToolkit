import os
import torch
import importlib


def custom_create_dataloaders(opt):
    dataset_filename = "data." + opt.dataset
    datasetlib = importlib.import_module(dataset_filename)
    # find method named `get_dataloaders`
    for name, method in datasetlib.__dict__.items():
        if name.lower() == "get_dataloaders":
            get_data_func = method
    return get_data_func(opt.batch_size, opt.n_workers, path=opt.root_path)
