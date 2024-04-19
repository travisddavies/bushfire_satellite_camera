import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os

DATA_JSON = 'data/satellite_bushfire_json.json'
IMAGE_DIR = 'data/prelim_dataset'
H = 1830
W = 1830


class ImageDataset(Dataset):
    def __init__(self, transform):
        with open(DATA_JSON, 'r') as f:
            raw_json_data = json.load(f)
        filenames = [value['filename'] for value in raw_json_data.values()]
        self.transform = transform
        self.filepaths = [os.path.join(IMAGE_DIR, filename)
                          for filename in filenames]
        self.annotations = []
        for value in raw_json_data.values():
            regions = value['regions']
            self.annotations.append(regions)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        img = cv2.imread(filepath)
        img = cv2.resize(img, (H, W))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        mask = self._get_mask(idx)
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
        target = {'masks': masks, 'boxes': bbox, 'labels': labels}

        return img, target

    def _get_regions(self, idx):
        regions_coords = []
        for region in self.annotations[idx]:
            x_coords = region['shape_attributes']['all_points_x']
            y_coords = region['shape_attributes']['all_points_y']
            polygon = [[x, y] for x, y in zip(x_coords, y_coords)]
            regions_coords.append(polygon)

        return regions_coords

    def _get_mask(self, idx):
        mask = np.zeros((H, W, 3), dtype=np.uint8)
        regions = self._get_regions(idx)
        factor = len(regions)
        for i, region in enumerate(regions):
            colour_val = int(((i+1)/factor) * 255)

            region = np.array(region, dtype=np.int32)
            cv2.fillPoly(mask, [region], (colour_val, colour_val, colour_val))

        return mask

    def _get_mask_set(self, mask):
        instance_vals = np.unique(mask)[1:]
        masks = np.array([np.where(mask == instance_val, 255, 0).astype(np.uint8)
                          for instance_val in instance_vals], dtype=np.int32)

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
