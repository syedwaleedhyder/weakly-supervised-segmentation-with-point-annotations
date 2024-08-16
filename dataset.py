import os
import csv
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_dict, point_label=False, point_label_percentage=0.01):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.class_dict = self.load_class_dict(class_dict)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.ToTensor()
        self.point_label = point_label
        self.point_label_percentage = point_label_percentage

    def __len__(self):
        return len(self.images)

    def random_point_mask(self, shape):
        total_points = int(shape[0] * shape[1] * self.point_label_percentage)
        mask = torch.zeros(shape)
        points = torch.randperm(shape[0] * shape[1], generator=torch.Generator().manual_seed(42))[:total_points]
        mask.view(-1)[points] = 1
        return mask.unsqueeze(0)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.images[idx]}")
        mask_path = os.path.join(self.mask_dir, f"{self.images[idx]}")
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        mask_one_hot = np.zeros((mask.shape[0], mask.shape[1], len(self.class_dict)))

        for k, v in self.class_dict.items():
            r, g, b = k
            class_mask = (mask[:, :, 0] == r) & (mask[:, :, 1] == g) & (mask[:, :, 2] == b)
            mask_one_hot[:, :, v][class_mask] = 1

        image, mask_one_hot = self.img_transform(image), self.mask_transform(mask_one_hot)

        if self.point_label:
            mask_point = self.random_point_mask(mask_one_hot.shape[1:])
            return image, mask_one_hot, mask_point
        
        return image, mask_one_hot

    @staticmethod
    def load_class_dict(class_dict_path):
        class_dict = {}
        with open(class_dict_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                class_name, r, g, b = row
                class_dict[tuple(map(int, (r, g, b)))] = len(class_dict)
        return class_dict