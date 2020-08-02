from box import Box
import sys

def read_process_config(filename, default_config="config/default.yaml"):
    config = Box.from_yaml(filename=filename)
    # handle label smoothing
    config.model.label_smoothing = config.label_smoothing
    config.model.eps = config.eps

    return config
