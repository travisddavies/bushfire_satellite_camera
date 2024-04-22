import torch
import torchvision.transforms as T
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import random_split, DataLoader
from transformers import SamProcessor, SegformerImageProcessor
from argparse import ArgumentParser
from data.dataset import (
    MaskRCNNDataset,
    SegmentAnythingDataset,
    SegFormerDataset)


def get_intersection(pred, ground_truth):
    return (pred * ground_truth).sum().to(torch.float32)


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

    train_ratio = 0.7

    torch.manual_seed(42)
    transform = T.Compose([
        T.ToTensor()
    ])
    if model == 'segformer':
        processor = SegformerImageProcessor(do_reduce_labels=False)
        full_dataset = SegFormerDataset(transform, image_size, device,
                                        processor)
    elif model == 'mask_rcnn':
        full_dataset = MaskRCNNDataset(transform, image_size, device)
    elif model == 'sam':
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        full_dataset = SegmentAnythingDataset(transform, image_size, device,
                                              processor)
    train_len = int(len(full_dataset) * train_ratio)
    val_test_len = len(full_dataset) - train_len
    val_len = val_test_len // 2
    test_len = val_test_len - val_len

    train_dataset, val_test_dataset = random_split(
        full_dataset, [train_len, val_test_len])
    val_dataset, test_dataset = random_split(
        val_test_dataset, [val_len, test_len])

    def collate_fn(data):
        return data

    fn = collate_fn if model == 'mask_rcnn' else None

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
