import os
import torch

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

from tqdm import tqdm
from torch.nn import MSELoss

from utils import (get_iou, get_mcc, get_f1_score, get_optimiser, parse_args,
                   get_data, get_recall, get_precision)


def get_model(device):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                   progress=True)
    model.classifier = DeepLabHead(2048, 1)
    return model.to(device)


def train(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device,
    patience,
    val_step,
    optimiser
):
    best_state_dict = None
    best_iou = 0
    init_patience = 0

    criterion = MSELoss().to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        perform_train(model, train_dataloader, optimiser, criterion, device)
        if epoch % val_step == 0:
            acc_dict = perform_validation(model, val_dataloader, criterion,
                                          device)
            f1_score = acc_dict['f1']
            iou = acc_dict['iou']
            mcc = acc_dict['mcc']
            precision = acc_dict['precision']
            recall = acc_dict['recall']
            val_loss = acc_dict['loss']
            print(f'F1 score: {f1_score:.3f}. '
                  f'Precision: {precision:.3f}. '
                  f'Recall: {recall:.3f}. '
                  f'IOU: {iou:.3f}. '
                  f'MCC: {mcc:.3f}. '
                  f'Loss: {val_loss:.3f}.')
            if iou > best_iou:
                init_patience = 0
                best_iou = iou
                best_state_dict = model.state_dict()
                torch.save(best_state_dict,
                           os.path.join(args.save_path, 'deeplabv3.pth'))
                print(f'Saved model at epoch {epoch}')

        if init_patience >= patience:
            break
        init_patience += 1

    return best_state_dict


def perform_train(model, train_dataloader, optimiser, criterion, device):
    total_loss = 0
    n = 0
    model.eval()
    print('Training...')
    for batch in tqdm(train_dataloader):
        images, masks = batch
        images, masks = images.to(device), masks.float().to(device)
        optimiser.zero_grad()
        output = model(images)
        preds = output['out'].float()
        preds = preds[:, 0]
        loss = criterion(preds, masks)
        loss.backward()
        optimiser.step()
        n += 1
        total_loss += loss
    av_train_loss = total_loss / n
    print(f'Train loss: {av_train_loss}.')


def perform_validation(model, val_dataloader, criterion, device):
    f1_score = 0
    iou = 0
    mcc = 0
    n = 0
    recall = 0
    precision = 0
    loss = 0
    print('Validating...')
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            predictions = output['out']
            predictions = predictions[:, 0].float()
            masks = masks.float()
            loss += criterion(predictions, masks)
            for gt_mask, prediction in zip(masks, predictions):
                pred_mask = (prediction > 0.5).byte()
                gt_mask = gt_mask.cpu().numpy()
                pred_mask = pred_mask.cpu().numpy()
                f1_score += get_f1_score(pred_mask, gt_mask)
                iou += get_iou(pred_mask, gt_mask)
                mcc += get_mcc(pred_mask, gt_mask)
                recall += get_recall(pred_mask, gt_mask)
                precision += get_precision(pred_mask, gt_mask)

                n += 1

    av_f1 = f1_score / n
    av_iou = iou / n
    av_mcc = mcc / n
    av_recall = recall / n
    av_precision = precision / n
    av_loss = loss / n

    return {'f1': av_f1, 'iou': av_iou, 'mcc': av_mcc, 'recall': av_recall,
            'precision': av_precision, 'loss': av_loss}


if __name__ == "__main__":
    args = parse_args()
    num_epochs = args.num_epochs
    save_path = args.save_path
    patience = args.patience
    val_step = args.validation_step
    batch_size = args.batch_size
    image_size = args.image_size
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    train_dataloader, val_dataloader, test_dataloader = get_data(
        batch_size,
        image_size,
        device,
        model="deeplabv3")

    model = get_model(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = get_optimiser(args, params)

    best_state_dict = train(model, train_dataloader, val_dataloader,
                            num_epochs, device, patience, val_step, optimiser)
    if best_state_dict:
        torch.save(best_state_dict,
                   os.path.join(args.save_path, 'deeplabv3.pth'))
