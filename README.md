# CNNResearchToolkit
Research Toolkit for Cutting-Edge DL/ML algorithms.

## Setup

```shell
pip install -r requirements.txt
```

## Usage

```shell
python3 train.py [path to config]
```

**Examples**

Training MobileNetV1 on ImageNet

```shell
python3 train.py config/mbv1_imagenet.yaml
```

## Citation

If you use this toolkit in your research, please consider citing:

```
@misc{li2020eagleeye,
    title={EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning},
    author={Bailin Li and Bowen Wu and Jiang Su and Guangrun Wang and Liang Lin},
    year={2020},
    eprint={2007.02491},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
