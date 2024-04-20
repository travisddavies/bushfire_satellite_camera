import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os

DATA_JSON = 'data/satellite_bushfire_json.json'
IMAGE_DIR = 'data/prelim_dataset'


class ImageDataset(Dataset):
    def __init__(self, transform, image_size, device):
        with open(DATA_JSON, 'r') as f:
            raw_json_data = json.load(f)
        filenames = [value['filename'] for value in raw_json_data.values()]
        self.device = device
        self.transform = transform
        self.filepaths = [os.path.join(IMAGE_DIR, filename)
                          for filename in filenames]
        self.annotations = []
        self.image_size = image_size
        for value in raw_json_data.values():
            regions = value['regions']
            self.annotations.append(regions)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        img = cv2.imread(filepath)
        mask = self._get_mask(idx, False, img.shape[0])
        objs = np.unique(mask)[1:]
        num_objs = len(objs)
        masks = self._get_mask_set(mask)

        if num_objs == 0:
            bbox = torch.empty((0, 4), dtype=torch.int64)
            labels = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((1, mask.shape[0], mask.shape[1]),
                                dtype=torch.uint8)
        else:
            bbox = self._get_bbox(idx)
            bbox = torch.as_tensor(bbox, dtype=torch.int64)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        target = {'masks': masks.to(self.device),
                  'boxes': bbox.to(self.device),
                  'labels': labels.to(self.device)}

        if bbox.shape[0] != labels.shape[0]:
            print(len(self._get_regions(idx)))
            print(num_objs)
            print(filepath)
            cv2.imwrite('weird_mask.png', mask)
            self._get_mask(idx, True)
        assert bbox.shape[0] == labels.shape[0], f"labels and boxes don't match, len(boxes) = {bbox.shape[0]} and len(labels) = {labels.shape[0]}"

#        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, target

    def _get_regions(self, idx):
        regions_coords = []
        for region in self.annotations[idx]:
            x_coords = region['shape_attributes']['all_points_x']
            y_coords = region['shape_attributes']['all_points_y']
            polygon = [[x, y] for x, y in zip(x_coords, y_coords)]
            regions_coords.append(polygon)

        return regions_coords

    def _get_mask(self, idx, test, image_size):
        mask = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        regions = self._get_regions(idx)
        factor = len(regions)
        for i, region in enumerate(regions):
            colour_val = int(((i+1)/factor) * 255)

            region = np.array(region, dtype=np.int32)
            cv2.fillPoly(mask, [region], (colour_val, colour_val, colour_val))
            if test:
                print(f"colour_val: {colour_val}, region: {region}")
                cv2.imwrite(f'test_{i}.png', mask)

        return mask

    def _get_mask_set(self, mask):
        instance_vals = np.unique(mask)[1:]
        masks = np.array([np.where(mask == instance_val, 255, 0).astype(np.uint8)
                          for instance_val in instance_vals], dtype=np.int32)
        if masks.ndim == 4:
            masks = masks[:, :, :, 0]
        return masks

    def _get_bbox(self, idx):
        bboxes = []
        regions = self._get_regions(idx)
        for region in regions:
            region = np.array(region, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(region)
            bboxes.append([x, y, x+w, y+h])
        bboxes = np.array(bboxes, dtype=np.int64)
        return bboxes
