import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random

H = 1830
W = 1830


class BushfireDataset(Dataset):
    def __init__(self, filepaths, annotations, transform, image_size, device):
        self.filepaths = filepaths
        self.annotations = annotations
        self.image_size = image_size
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        pass

    def _get_regions(self, idx):
        regions_coords = []
        for region in self.annotations[idx]:
            x_coords = region['shape_attributes']['all_points_x']
            y_coords = region['shape_attributes']['all_points_y']
            polygon = [[x, y] for x, y in zip(x_coords, y_coords)]
            regions_coords.append(polygon)

        return regions_coords

    def _get_mask(self, idx, instance_seg):
        mask = np.zeros((H, W, 3), dtype=np.uint8)
        regions = self._get_regions(idx)
        factor = len(regions)
        for i, region in enumerate(regions):
            colour_val = int(((i+1)/factor) * 255) if instance_seg else 1
            region = np.array(region, dtype=np.int32)
            cv2.fillPoly(mask, [region], (colour_val, colour_val, colour_val))
        mask = mask[:, :, 0]

        return mask

    def _get_random_crop_coords(self):
        furthest_left = W - self.image_size
        furthest_top = H - self.image_size
        x1 = random.randint(0, furthest_left)
        y1 = random.randint(0, furthest_top)
        x2 = x1 + self.image_size
        y2 = y1 + self.image_size

        return x1, y1, x2, y2


class BushfireInstanceSegmentationDataset(BushfireDataset):
    def __init__(self, filepaths, annotations, transform, image_size, device):
        super().__init__(filepaths, annotations, transform, image_size, device)

    def _get_mask_set(self, mask):
        instance_vals = np.unique(mask)[1:]
        masks = np.array([self._get_instance(mask, instance_val)
                          for instance_val in instance_vals], dtype=np.uint8)

        return masks

    def _get_instance(self, mask, instance_val):
        instance = np.where(mask == instance_val, 1, 0).astype(np.uint8)

        return instance

    def _get_bbox(self, masks):
        bboxes = []
        for mask in masks:
            region, _ = cv2.findContours(mask, mode=cv2.RETR_TREE,
                                                   method=cv2.CHAIN_APPROX_NONE)
            x, y, w, h = cv2.boundingRect(region[0])
            bboxes.append([x, y, x+w, y+h])
        bboxes = np.array(bboxes, dtype=np.int64)

        return bboxes


class MaskRCNNDataset(BushfireInstanceSegmentationDataset):
    def __init__(self, filepaths, annotations, transform, image_size, device,
                 random_crop=False):
        super().__init__(filepaths, annotations, transform, image_size, device)
        self.random_crop = random_crop

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self._get_mask(idx, instance_seg=True)
        if self.random_crop:
            x1, y1, x2, y2 = self._get_random_crop_coords()
            img, mask = img[x1:x2, y1:y2], mask[x1:x2, y1:y2]
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
        objs = np.unique(mask)[1:]
        num_objs = len(objs)
        masks = self._get_mask_set(mask)

        if num_objs == 0:
            bbox = torch.empty((0, 4), dtype=torch.int64)
            labels = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((1, mask.shape[0], mask.shape[1]),
                                dtype=torch.uint8)
        else:
            bbox = self._get_bbox(masks)
            bbox = torch.as_tensor(bbox, dtype=torch.int64)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        target = {'masks': masks.to(self.device),
                  'boxes': bbox.to(self.device),
                  'labels': labels.to(self.device)}


        assert mask.shape[:2] == img.shape[:2], f"mask shape {mask.shape[:2]} \
                does not equal img shape {img.shape[:2]}"

        img = self.transform(img)

        return img, target


class SegFormerDataset(BushfireDataset):
    def __init__(self, filepaths, annotations, transform, image_size, device,
                 processor, random_crop=False):
        super().__init__(filepaths, annotations, transform, image_size, device)
        self.processor = processor
        self.random_crop = True

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        img = cv2.imread(filepath)
        mask = self._get_mask(idx, instance_seg=False)
        if self.random_crop:
            x1, y1, x2, y2 = self._get_random_crop_coords()
            mask = mask[x1:x2, y1:y2]
            img = img[x1:x2, y1:y2]
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
        encoded_inputs = self.processor(img, mask, return_tensors="pt")
        encoded_inputs = {k: v.squeeze(0) for k, v in encoded_inputs.items()}

        return encoded_inputs


class SegmentAnythingDataset(BushfireInstanceSegmentationDataset):
    def __init__(self, transform, image_size, device, processor):
        super().__init__(transform, image_size, device, processor)
        self.processor = processor

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        image = cv2.imread(filepath)
        mask = self._get_mask(idx, instance_seg=True)
        masks = self._get_mask_set(mask)
        # get bounding box prompt
        boxes = self._get_bbox(masks)
        boxes = boxes.tolist()

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[boxes],
                                return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = mask

        return inputs
