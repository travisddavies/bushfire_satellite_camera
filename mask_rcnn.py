import os
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from data.dataset import MaskRCNNDataset
from torch.utils.data import random_split, DataLoader

from utils import get_iou, get_mcc, get_f1_score, get_optimiser, parse_args


def collate_fn(data):
    return data


def get_data(batch_size, image_size, device):
    train_ratio = 0.7

    torch.manual_seed(42)
    transform = T.Compose([
        T.ToTensor()
    ])

    full_dataset = MaskRCNNDataset(transform, image_size, device)
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

    for epoch in range(num_epochs):
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
    for i in tqdm(range(len(train_dataloader))):
        batch = next(train_iter)
        images, targets = zip(*batch)
        images = list(images)
        targets = list(targets)
        images = torch.stack(images).to(device)

        optimiser.zero_grad()
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


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    train_dataloader, val_dataloader, test_dataloader = get_data(
        args.batch_size, args.image_size, device)
    model = get_model(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = get_optimiser(args, params)

    num_epochs = args.num_epochs
    save_path = args.save_path
    patience = args.patience
    best_state_dict = train(model, train_dataloader, val_dataloader,
                            num_epochs, device, patience, optimiser)
    if best_state_dict:
        torch.save(best_state_dict,
                   os.path.join(args.save_path, 'mask_rcnn.pth'))
