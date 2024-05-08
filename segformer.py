import os
from time import time
import torch
from torch.nn.functional import interpolate
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

from utils import (parse_args, get_optimiser, get_f1_score, get_mcc, get_iou,
                   get_data, get_recall, get_precision)


def get_model(device):
    id2label = {
        0: "background",
        1: "object",
    }

    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                             num_labels=2,
                                                             id2label=id2label,
                                                             label2id=label2id)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() + param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() + buffer.element_size()
    size_all = (param_size + buffer_size) / 1024**2

    print(f'model size: {size_all:.3f}MB')

    path = os.path.join(os.getcwd(), args.save_path, 'segformer-b5.pth')
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        print('Loading pretrained model')

    return model.to(device)


def train(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device,
    patience,
    optimiser,
    val_step
):
    best_state_dict = None
    best_iou = 0
    init_patience = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        perform_train(model, train_dataloader, optimiser, device)
        if epoch % val_step == 0:
            acc_dict = perform_validation(model, val_dataloader, device)
            f1_score = acc_dict['f1']
            iou = acc_dict['iou']
            mcc = acc_dict['mcc']
            val_loss = acc_dict['loss']
            precision = acc_dict['precision']
            recall = acc_dict['recall']
            print(f'F1 score: {f1_score:.3f}. '
                  f'IOU: {iou:.3f}. '
                  f'MCC: {mcc:.3f}. '
                  f'Precision: {precision:.3f}. '
                  f'Recall: {recall:.3f}. '
                  f'Loss: {val_loss}')
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
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        optimiser.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimiser.step()


def perform_validation(model, val_dataloader, device):
    model.eval()

    running_f1 = 0.0
    running_iou = 0.0
    running_mcc = 0.0
    running_loss = 0.0
    running_recall = 0.0
    running_precision = 0.0

    n = 0

    print('Validating...')
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            if labels[0].cpu().sum() == 0:
                continue

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            running_loss += loss.item()

            upsampled_logits = interpolate(logits,
                                           size=labels.shape[-2:],
                                           mode='bilinear',
                                           align_corners=False)

            seg = upsampled_logits.argmax(dim=1).double()
            pred = (labels > 0.5).double()
            pred = pred.cpu().numpy()
            seg = seg.cpu().numpy()
            running_f1 += get_f1_score(pred, seg)
            running_iou += get_iou(pred, seg)
            running_mcc += get_mcc(pred, seg)
            running_recall += get_recall(pred, seg)

            n += 1

    accuracy = {}
    accuracy['f1'] = running_f1 / n
    accuracy['iou'] = running_iou / n
    accuracy['mcc'] = running_mcc / n
    accuracy['loss'] = running_loss / n
    accuracy['recall'] = running_recall / n
    accuracy['precision'] = running_precision / n

    return accuracy


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
        model="segformer")
    model = get_model(device)
    optimiser = get_optimiser(args, model.parameters())

    if args.train_mode:
        best_state_dict = train(model, train_dataloader, val_dataloader,
                                num_epochs, device, patience, optimiser,
                                val_step)
        if best_state_dict:
            torch.save(best_state_dict,
                       os.path.join(args.save_path, 'mask_rcnn.pth'))
    else:
        start = time()
        acc_dict = perform_validation(model, val_dataloader, device)
        end = time()
        print(f'FPS: {82 / (end - start)}')
        f1_score = acc_dict['f1']
        iou = acc_dict['iou']
        mcc = acc_dict['mcc']
        loss = acc_dict['loss']
        print(f'F1 score: {f1_score:.3f}. IOU: {iou:.3f}. MCC: {mcc:.3f}. '
              f'Loss: {loss:.3f}.')
