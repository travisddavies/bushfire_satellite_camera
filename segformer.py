import os
import torch
from torch.nn.functional import interpolate
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

from utils import (parse_args, get_optimiser, get_f1_score, get_mcc, get_iou,
                   get_data)


def get_model():
    id2label = {
        0: "background",
        1: "object",
    }

    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                             num_labels=2,
                                                             id2label=id2label,
                                                             label2id=label2id)

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
    for _ in tqdm(range(len(train_dataloader))):
        batch = next(train_iter)
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

    n = 0

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

            seg = upsampled_logits.argmax(dim=1).float()
            pred = (labels > 0.5).float()

            running_f1 += get_f1_score(pred, seg)
            running_iou += get_iou(pred, seg)
            running_mcc += get_mcc(pred, seg)

            n += 1

    avg_running_loss = running_loss / n
    accuracy = {}
    accuracy['avg_f1'] = running_f1 / n
    accuracy['avg_iou'] = running_iou / n
    accuracy['avg_mcc'] = running_mcc / n

    return accuracy


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    num_epochs = args.num_epochs
    save_path = args.save_path
    patience = args.patience

    train_dataloader, val_dataloader, test_dataloader = get_data(
        args.batch_size, args.image_size, device)
    model = get_model()
    optimiser = get_optimiser(args, model.parameters())

    best_state_dict = train(model, train_dataloader, val_dataloader,
                            num_epochs, device, patience, optimiser)
    if best_state_dict:
        torch.save(best_state_dict,
                   os.path.join(args.save_path, 'segformer.pth'))
