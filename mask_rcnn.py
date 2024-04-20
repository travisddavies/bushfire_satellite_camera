import os
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from torch.optim import Adam, AdamW, SGD
from argparse import ArgumentParser
from data.dataset import ImageDataset
from torch.utils.data import random_split, DataLoader


def collate_fn(data):
    return data


def get_data(batch_size, image_size, device):
    train_ratio = 0.7

    torch.manual_seed(42)
    transform = T.Compose([
        T.ToTensor()
    ])

    full_dataset = ImageDataset(transform, image_size, device)
    train_len = int(len(full_dataset) * train_ratio)
    val_test_len = len(full_dataset) - train_len
    val_len = val_test_len // 2
    test_len = val_test_len - val_len

    train_dataset, val_test_dataset = random_split(
        full_dataset, [train_len, val_test_len])
    val_dataset, test_dataset = random_split(
        val_test_dataset, [val_len, test_len])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def get_model(device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=2)
    model.to(device)
    return model


def get_optimiser(optimiser, params):
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


def train(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device,
    patience,
    optimiser
):
    best_state_dict = None
    best_iou = 0
    init_patience = 0

    for epoch in tqdm(range(num_epochs)):
        perform_train(model, train_dataloader, optimiser, device)
        if epoch % 10 == 0:
            acc_dict = perform_validation(model, val_dataloader, device)
            f1_score = acc_dict['f1']
            iou = acc_dict['iou']
            mcc = acc_dict['mcc']
            print(f'Epoch: {epoch}. F1 score: {f1_score}. IOU: {iou}. MCC: {mcc}.')
            if iou < best_iou:
                init_patience = 0
                best_iou = iou
                best_state_dict = model.state_dict()
        if init_patience >= patience:
            break
        init_patience += 1

    return best_state_dict


def perform_train(model, train_dataloader, optimiser, device):
    model.train()
    train_iter = iter(train_dataloader)
    for i in range(len(train_dataloader)):
        batch = next(train_iter)
        images, targets = zip(*batch)
        images = list(images)
        images = torch.stack(images)
        images = images.to(device)
        targets = list(targets)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimiser.step()


def perform_validation(model, val_dataloader, device):
    f1_score = 0
    iou = 0
    mcc = 0
    n = 0

    model.eval()
    val_iter = iter(val_dataloader)
    with torch.no_grad():
        for _ in range(len(val_dataloader)):
            batch = next(val_iter)
            images, targets = zip(*batch)
            images, targets = list(images), list(targets)
            images = torch.stack(images)
            images = images.to(device)
            predictions = model(images)

            for target, prediction in zip(targets, predictions):
                pred_masks = (prediction["masks"] > 0.5).byte()
                gt_masks = target["masks"]
                combined_pred_masks = torch.clamp(pred_masks.sum(dim=0), 0, 1)
                combined_gt_masks = torch.clamp(gt_masks.sum(dim=0), 0, 1)
                combined_pred_masks = combined_pred_masks.squeeze(dim=0)
                f1_score += get_f1_score(combined_pred_masks, combined_gt_masks)
                iou += get_iou(combined_pred_masks, combined_gt_masks)
                mcc += get_mcc(combined_pred_masks, combined_gt_masks)
                n += 1

    av_f1 = f1_score / n
    av_iou = iou / n
    av_mcc = mcc / n

    return {'f1': av_f1, 'iou': av_iou, 'mcc': av_mcc}


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


if __name__ == "__main__":
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
    argparse.add_argument("-i", "--image_size", type=int, default=512)
    args = argparse.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    train_dataloader, val_dataloader, test_dataloader = get_data(
        args.batch_size, args.image_size, device)
    model = get_model(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = get_optimiser(args.optimiser, params)

    num_epochs = args.num_epochs
    save_path = args.save_path
    patience = args.patience
    best_state_dict = train(model, train_dataloader, val_dataloader,
                            num_epochs, device, patience, optimiser)
    if best_state_dict:
        torch.save(best_state_dict,
                   os.path.join(args.save_path, 'mask_rcnn.pth'))
