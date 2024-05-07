import os
import json
import torchvision.transforms as T
import numpy as np
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor, MobileViTImageProcessor
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from data.dataset import (
    MaskRCNNDataset,
    SegFormerDataset,
    MobileViTDataset,
    DeepLabV3Dataset)


DATA_JSON = 'data/satellite_bushfire_json.json'
IMAGE_DIR = 'data/images'


def get_intersection(pred, ground_truth):
    return (pred * ground_truth).sum()


def get_precision(pred, ground_truth):
    tp = get_intersection(pred, ground_truth)
    fp = np.where((pred == 1) & (ground_truth == 0), 1, 0).sum()
    return tp / (tp + fp + 1e-8)


def get_recall(pred, ground_truth):
    tp = get_intersection(pred, ground_truth)
    fn = np.where((pred == 0) & (ground_truth == 1), 1, 0).sum()
    return tp / (tp + fn + 1e-8)


def get_f1_score(pred, ground_truth):
    intersection = get_intersection(pred, ground_truth)
    f1_acc = (2 * intersection) / (ground_truth.sum() + pred.sum() + 1e-8)
    return float(f1_acc)


def get_iou(pred, ground_truth):
    intersection = get_intersection(pred, ground_truth)
    union = ground_truth.sum() + pred.sum() - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)

    return float(iou)


def get_mcc(pred, ground_truth):
    tp = int((pred * ground_truth).sum())
    fp = int((pred * (1 - ground_truth)).sum())
    fn = int(((1 - pred) * ground_truth).sum())
    tn = int(((1 - pred) * (1 - ground_truth)).sum())

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + + 1e-8) ** 0.5

    return numerator / (denominator + 1e-8)


def get_optimiser(args, params):
    optimiser = args.optimiser
    if optimiser == "sgd":
        optimiser = SGD(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    elif optimiser == "adam":
        optimiser = Adam(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif optimiser == "adamw":
        optimiser = AdamW(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    return optimiser


def get_data(batch_size, image_size, device, model):
    with open(DATA_JSON, 'r') as f:
        raw_json_data = json.load(f)
    filenames = [value['filename'] for value in raw_json_data.values()]
    filepaths = [os.path.join(IMAGE_DIR, filename) for filename in filenames]
    for i, filepath in enumerate(filepaths):
        idx = filepath.find('_mask')
        filepaths[i] = filepath[:idx] + '.jpg'
    annotations = []
    image_size = image_size
    for value in raw_json_data.values():
        regions = value['regions']
        annotations.append(regions)
    train_ratio = 0.7

    train_filepaths, val_test_filepaths, train_annotations, val_test_annotations = train_test_split(
        filepaths, annotations, train_size=train_ratio, random_state=42,
        shuffle=True
    )
    val_filepaths, test_filepaths, val_annotations, test_annotations = train_test_split(
        val_test_filepaths, val_test_annotations, test_size=0.5,
        random_state=42, shuffle=True

    )
    transform = T.Compose([
        T.ToTensor()
    ])
    if model == 'segformer':
        processor = SegformerImageProcessor(do_reduce_labels=False)
        train_dataset = SegFormerDataset(train_filepaths, train_annotations,
                                         transform, image_size, device,
                                         processor, random_crop=True)
        val_dataset = SegFormerDataset(val_filepaths, val_annotations,
                                       transform, image_size, device,
                                       processor)
        test_dataset = SegFormerDataset(test_filepaths, test_annotations,
                                        transform, image_size, device,
                                        processor)
    if model == 'mobilevit':
        processor = MobileViTImageProcessor(do_reduce_labels=False)
        train_dataset = MobileViTDataset(train_filepaths, train_annotations,
                                         transform, image_size, device,
                                         processor, random_crop=True)
        val_dataset = MobileViTDataset(val_filepaths, val_annotations,
                                       transform, image_size, device,
                                       processor)
        test_dataset = MobileViTDataset(test_filepaths, test_annotations,
                                        transform, image_size, device,
                                        processor)

    elif model == 'mask_rcnn':
        train_dataset = MaskRCNNDataset(train_filepaths, train_annotations,
                                        transform, image_size, device,
                                        random_crop=True)
        val_dataset = MaskRCNNDataset(val_filepaths, val_annotations,
                                      transform, image_size, device)
        test_dataset = MaskRCNNDataset(test_filepaths, test_annotations,
                                       transform, image_size, device)

    elif model == 'deeplabv3':
        train_dataset = DeepLabV3Dataset(train_filepaths, train_annotations,
                                         transform, image_size, device,
                                         random_crop=True)
        val_dataset = DeepLabV3Dataset(val_filepaths, val_annotations,
                                       transform, image_size, device)
        test_dataset = DeepLabV3Dataset(test_filepaths, test_annotations,
                                        transform, image_size, device)

    def mask_rcnn_collate_fn(data):
        return data

    if model == 'mask_rcnn':
        fn = mask_rcnn_collate_fn
    else:
        fn = None

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, collate_fn=fn)

    return train_dataloader, val_dataloader, test_dataloader


def parse_args():
    argparse = ArgumentParser(description='Hyperparameters for training the MaskRCNN model')
    argparse.add_argument("-e", "--num_epochs", type=int, default=500)
    argparse.add_argument("-p", "--patience", type=int, default=100)
    argparse.add_argument("-l", "--learning_rate", type=float, default=0.001)
    argparse.add_argument("-w", "--weight_decay", type=float, default=5e-4)
    argparse.add_argument("-s", "--save_path", type=str,
                          default="saved_models")
    argparse.add_argument("-m", "--momentum", type=float, default=0.9)
    argparse.add_argument("-o", "--optimiser", type=str,
                          choices=["adam", "adamw", "sgd"], default="sgd")
    argparse.add_argument("-b", "--batch_size", type=int, default=32)
    argparse.add_argument("-i", "--image_size", type=int, default=1830)
    argparse.add_argument("-v", "--validation_step", type=int, default=10)
    args = argparse.parse_args()
    return args
