import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from PIL import Image

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import SamModel, SamProcessor
import monai
from thop import profile
from torchinfo import summary

model = SamModel.from_pretrained("facebook/sam-vit-base")


param_size = 0
for param in model.parameters():
    param_size += param.nelement() + param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() + buffer.element_size()
size_all = (param_size + buffer_size) / 1024**2

print(f'model size: {size_all:.3f}MB')

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)


def get_bounding_box(ground_truth_map):

    image = ground_truth_map.copy().astype(np.uint8)

    kernel = np.ones((5,5),np.uint8)  # You can adjust the size based on the size of the noise
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(len(contours))

    points = []

    for contour in contours:
        # For each contour, get the bounding rectangle and save it to bounding_boxes
        x, y, w, h = cv2.boundingRect(contour)
        # bounding_boxes.append((x, y, w, h))

        x_min = float(x + w//2)
        y_min = float(y + h//2)

        # x_max = float(x + w)
        # y_max = float(y + h)

        points.append([x_min, y_min])

    points = [point for point in points if len(point) > 0]

    return points

class SAMDataset(Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, images_src, masks_src, processor):
        imgs = list(Path(images_src).glob("**/*.jpg"))
        masks = list(Path(masks_src).glob("**/*.jpg"))
        imgs = {img.stem: img for img in imgs}

        data = [[imgs[mask.stem], mask] for mask in masks if imgs.get(mask.stem)]
        self.resized_data = []

        for image_path, mask_path in tqdm(data):
            image_cv2 = cv2.imread(str(image_path))
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            image_cv2 = cv2.resize(image_cv2, (256, 256))
            # image_cv2 = np.expand_dims(image_cv2, axis=0)

            mask_cv2 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_cv2 = (mask_cv2 > 0).astype(np.uint8)
            mask_cv2 = cv2.resize(mask_cv2, (256, 256))

            if mask_cv2.max() != 0:
                if len(get_bounding_box(mask_cv2)) > 0:
                    self.resized_data.append((image_cv2, mask_cv2))

        print(f"Read in {len(data)} images, resized to 256, with {len(self.resized_data)} remaining with objects")

        # images_array = np.array([pair[0] for pair in resized_data])  # Shape: (100, 256, 256, 3)
        # masks_array = np.array([pair[1] for pair in resized_data])   # Shape: (100, 256, 256)


        # self.images = images_array
        # self.masks = masks_array
        self.processor = processor

    def __len__(self):
        return len(self.resized_data)

    def __getitem__(self, idx):

        image, ground_truth_mask = self.resized_data[idx]

        image = Image.fromarray(image)

        # ground_truth_mask = np.array(self.masks[idx])

        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_points=[prompt], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

    def output_example(self, idx):

        image, ground_truth_mask = self.resized_data[idx]

        image = Image.fromarray(image)

        # ground_truth_mask = np.array(self.masks[idx])

        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_points=[prompt], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return image, inputs

def train(
    model,
    num_epochs,
    patience,
    train_dataloader,
    valid_dataloader,
    save_path,
    optimizer,
    scheduler,
    device,
):

    model.to(device)
    model.train()

    for epoch in range(num_epochs):

        best_loss = 999.0
        num_stop = 0
        train_loss = 0

        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_points=batch["input_points"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()

            train_loss += loss.item()

        num_batch = len(train_dataloader)
        avg_train_loss = train_loss / num_batch

        val_loss, accuracy = validate_model(model, valid_dataloader, device)
        f1_score = accuracy['avg_f1']

        if val_loss < best_loss:
            best_loss = val_loss
            num_stop = 0
            torch.save(model.state_dict(), save_path + "/best_model.pth")
        else:
            num_stop += 1

        if num_stop == patience:
            print("Early stopping")
            break

        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        print(f'EPOCH: {epoch}, LR: {curr_lr}')
        print(f'Train loss: {avg_train_loss}, Val loss: {val_loss}, Val F1: {f1_score}')

    torch.save(model.state_dict(), save_path + "/best_model.pth")

def get_f1(label, predict):
    # dice acc
    predict = (predict > 0.5).astype(np.float32)

    intersection = (predict * label).sum()

    f1_acc = (2 * intersection) / (label.sum() + predict.sum() + 1e-8)

    return f1_acc

def get_iou(label, predict):
    predict = (predict > 0.5).astype(np.float32)

    intersection = (predict * label).sum()
    union = label.sum() + predict.sum() - intersection

    return (intersection + 1e-8) / (union + 1e-8)

def get_mcc(label, predict):
    predict = (predict > 0.5).astype(np.float32)

    tp = (predict * label).sum()
    fp = (predict * (1 - label)).sum()
    fn = ((1 - predict) * label).sum()
    tn = ((1 - predict) * (1 - label)).sum()

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return numerator / (denominator + 1e-8)

def validate_model(
    model,
    dataloader,
    device,
):

    running_f1 = 0
    running_iou = 0
    running_mcc = 0
    running_loss = 0
    n = 0
    model.eval()
    # forward pass
    with torch.no_grad():
        for batch in tqdm(dataloader):

            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_points=batch["input_points"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            running_loss += loss.item()

            # apply sigmoid
            medsam_seg_prob = torch.sigmoid(predicted_masks)
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.9).astype(np.uint8)
            ground_truth_mask = np.array(batch["ground_truth_mask"].squeeze())

            running_f1 += get_f1(ground_truth_mask, medsam_seg)
            running_iou += get_iou(ground_truth_mask, medsam_seg)
            running_mcc += get_mcc(ground_truth_mask, medsam_seg)

            n += 1
    avg_running_loss = running_loss / n
    accuracy = {}
    accuracy['avg_f1'] = running_f1 / n
    accuracy['avg_iou'] = running_iou / n
    accuracy['avg_mcc'] = running_mcc / n

    return avg_running_loss, accuracy

ROOT = "data"
MODEL_SAVE_PATH = "saved_models"

TRAIN_IMG_DIR = ROOT + "/images/train_images"
TRAIN_MASK_DIR = ROOT + "/masks/train_masks"
VAL_IMG_DIR = ROOT + "/images/val_images"
VAL_MASK_DIR = ROOT + "/masks/val_masks"
TEST_IMG_DIR = ROOT + '/images/test_images'
TEST_MASK_DIR = ROOT + '/masks/test_masks'
VISUAL_IMG_DIR = ROOT + '/images/visual_check_images'
VISUAL_MASK_DIR = ROOT + '/masks/visual_check_masks'

BATCH_SIZE = 1
IMG_SIZE = 512

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

train_dataset = SAMDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, processor)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)

valid_dataset = SAMDataset(VAL_IMG_DIR, VAL_MASK_DIR, processor)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)

test_dataset = SAMDataset(TEST_IMG_DIR, TEST_MASK_DIR, processor)
test_dataloader = DataLoader(test_dataset, batch_size=1)

visual_dataset = SAMDataset(VISUAL_IMG_DIR, VISUAL_MASK_DIR, processor)
visual_dataloader = DataLoader(visual_dataset, batch_size=1)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("using CUDA")
else:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using MPS")
    else:
        device = torch.device("cpu")
        print("using CPU")

NUM_EPOCHS = 3
PATIENCE = 100
LEARNING_RATE_0 = 0.0001
LEARNING_RATE_F = LEARNING_RATE_0 * 0.1
WEIGHT_DECAY = 5e-4

optimizer = Adam(model.mask_decoder.parameters(), lr=LEARNING_RATE_0, weight_decay=WEIGHT_DECAY)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = NUM_EPOCHS, eta_min=1e-5)

train(model=model,
      num_epochs=NUM_EPOCHS,
      patience=PATIENCE,
      train_dataloader=train_dataloader,
      valid_dataloader=valid_dataloader,
      save_path=MODEL_SAVE_PATH,
      optimizer=optimizer,
      scheduler=scheduler,
      device=device)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("using CUDA")
else:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using MPS")
    else:
        device = torch.device("cpu")
        print("using CPU")

NUM_EPOCHS = 3
PATIENCE = 100
LEARNING_RATE_0 = 0.0001
LEARNING_RATE_F = LEARNING_RATE_0 * 0.1
WEIGHT_DECAY = 5e-4

optimizer = Adam(model.mask_decoder.parameters(), lr=LEARNING_RATE_0, weight_decay=WEIGHT_DECAY)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = NUM_EPOCHS, eta_min=1e-5)

train(model=model,
      num_epochs=NUM_EPOCHS,
      patience=PATIENCE,
      train_dataloader=train_dataloader,
      valid_dataloader=valid_dataloader,
      save_path=MODEL_SAVE_PATH,
      optimizer=optimizer,
      scheduler=scheduler,
      device=device)
