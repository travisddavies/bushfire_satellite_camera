import os
import torch
from transformers import SamModel
from tqdm import tqdm
from monai.losses import DiceCELoss

from utils import (parse_args, get_optimiser, get_f1_score, get_mcc, get_iou,
                   get_data)


def get_model():
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    return model


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
    init_patience = 0
    best_iou = 0
    best_state_dict = None
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    for epoch in range(num_epochs):
        perform_train(model, train_dataloader, optimiser, seg_loss, device)
        if epoch % val_step == 0:
            acc_dict = perform_validation(model, val_dataloader, device)
            f1_score = acc_dict['f1']
            iou = acc_dict['iou']
            mcc = acc_dict['mcc']
            val_loss = acc_dict['loss']
            print(f'F1 score: {f1_score:.3f}. IOU: {iou:.3f}. MCC: {mcc:.3f}. Loss: {val_loss}')
            if iou < best_iou:
                init_patience = 0
                best_iou = iou
                best_state_dict = model.state_dict()
        if init_patience >= patience:
            break
        init_patience += 1

    return best_state_dict


def perform_train(model, train_dataloader, optimiser, loss, device):
    model.train()
    for batch in tqdm(train_dataloader):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


def perform_validation(model, dataloader, device, loss):
    running_f1 = 0
    running_iou = 0
    running_mcc = 0
    running_loss = 0
    n = 0
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_points=batch["input_points"].to(device),
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            running_loss += loss.item()

            medsam_seg_prob = torch.sigmoid(predicted_masks)
            medsam_seg_prob = medsam_seg_prob.squeeze()
            medsam_seg = (medsam_seg_prob > 0.9).to(torch.float32)
            ground_truth_mask = batch["ground_truth_mask"].squeeze()

            running_f1 += get_f1_score(ground_truth_mask, medsam_seg)
            running_iou += get_iou(ground_truth_mask, medsam_seg)
            running_mcc += get_mcc(ground_truth_mask, medsam_seg)
            n += 1

    accuracy = {}
    accuracy['f1'] = running_f1 / n
    accuracy['iou'] = running_iou / n
    accuracy['mcc'] = running_mcc / n
    accuracy['loss'] = running_loss / n

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
        model="sam")
    model = get_model(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = get_optimiser(args, params)

    best_state_dict = train(model, train_dataloader, val_dataloader,
                            num_epochs, device, patience, val_step, optimiser)
    if best_state_dict:
        torch.save(best_state_dict,
                   os.path.join(args.save_path, 'sam_model.pth'))
