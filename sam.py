import os
import torch
from transformers import SamModel
from tqdm import tqdm
from monai.losses import DiceCELoss
from statistics import mean

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
    optimiser
):
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')


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
