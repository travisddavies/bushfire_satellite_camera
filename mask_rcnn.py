import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from utils import (get_iou, get_mcc, get_f1_score, get_optimiser, parse_args,
                   get_data)


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
        print(f'Epoch {epoch}')
        perform_train(model, train_dataloader, optimiser, device)
        if epoch % 10 == 0:
            acc_dict = perform_validation(model, val_dataloader, device)
            f1_score = acc_dict['f1']
            iou = acc_dict['iou']
            mcc = acc_dict['mcc']
            print(f'F1 score: {f1_score}. IOU: {iou}. MCC: {mcc}.')
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
    print('Training...')
    for batch in tqdm(train_dataloader):
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
    print('Validating...')
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
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
        args.batch_size, args.image_size, device, model='mask_rcnn')
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
