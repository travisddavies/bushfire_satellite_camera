import torch
from torch.optim import Adam, AdamW, SGD
from argparse import ArgumentParser


def get_intersection(pred, ground_truth):
    return (pred * ground_truth).sum().to(torch.float32)


def get_f1_score(pred, ground_truth):
    intersection = get_intersection(pred, ground_truth)
    f1_acc = (2 * intersection) / (ground_truth.sum() + pred.sum() + 1e-8)

    return int(f1_acc)


def get_iou(pred, ground_truth):
    intersection = get_intersection(pred, ground_truth)
    union = ground_truth.sum() + pred.sum() - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)

    return int(iou)


def get_mcc(pred, ground_truth):
    tp = (pred * ground_truth).sum()
    fp = (pred * (1 - ground_truth)).sum()
    fn = ((1 - pred) * ground_truth).sum()
    tn = ((1 - pred) * (1 - ground_truth)).sum()

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return int(numerator / (denominator + 1e-8))


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
    args = argparse.parse_args()
    return args
