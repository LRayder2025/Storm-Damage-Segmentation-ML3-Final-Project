#!/usr/bin/env python
# coding: utf-8

# # xView2 DeepLabV3+ Project

# ### Imports; Download Data

import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import cv2
import matplotlib.patches as mpatches
import PIL
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import manual_seed as torch_manual_seed
from torch.cuda import manual_seed_all
from torch.backends import cudnn
from torchvision.models.segmentation import (
    deeplabv3_resnet50, 
    DeepLabV3_ResNet50_Weights
)

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

def setup_seed(seed):
    torch_manual_seed(seed)
    manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False 

SEED = 42
setup_seed(SEED)

# Download Data
# !pip install gdown
# !gdown --id 1kMC2PCTyWoOiL0AItssA7Grh4CSoPO2K -O data.zip
# !unzip -q data.zip -d /content/data
# print("finished unzipping")

### Dataset loader

class XView2Dataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.files = sorted([
            f.replace("_post_disaster.jpg", "").replace("_post_disaster.png", "")
            for f in os.listdir(image_dir)
            if "_post_disaster" in f
        ])
        self.damage_map = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4,
            "un-classified": 0   
        }

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def _load_post_mask(self, json_path, shape):
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        with open(json_path) as f:
            data = json.load(f)
        for feature in data["features"]["xy"]:
            props = feature["properties"]
            if "subtype" not in props: continue
            damage = props["subtype"]
            class_id = self.damage_map[damage]
            wkt = feature["wkt"]
            coords = wkt.replace("POLYGON ((", "").replace("))", "")
            points = []
            for pair in coords.split(","):
                x, y = map(float, pair.strip().split())
                points.append([int(x), int(y)])
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], class_id)
        return mask

    def _load_pre_mask(self, json_path, shape):
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        with open(json_path) as f:
            data = json.load(f)
        for feature in data["features"]["xy"]:
            wkt = feature["wkt"]
            coords = wkt.replace("POLYGON ((", "").replace("))", "")
            points = []
            for pair in coords.split(","):
                x, y = map(float, pair.strip().split())
                points.append([int(x), int(y)])
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def __getitem__(self, idx):
        base = self.files[idx]
        for ext in [".jpg", ".png"]:
            pre_path = os.path.join(self.image_dir, base + "_pre_disaster" + ext)
            post_path = os.path.join(self.image_dir, base + "_post_disaster" + ext)
            if os.path.exists(pre_path): break

        pre_img = self._load_image(pre_path)
        post_img = self._load_image(post_path)
        image = np.concatenate([pre_img, post_img], axis=2)

        pre_label_path = os.path.join(self.label_dir, base + "_pre_disaster.json")
        post_label_path = os.path.join(self.label_dir, base + "_post_disaster.json")
        pre_mask = self._load_pre_mask(pre_label_path, pre_img.shape[:2])
        post_mask = self._load_post_mask(post_label_path, pre_img.shape[:2])

        image = torch.tensor(image).permute(2, 0, 1).float()
        return {
            "image": image,
            "pre_mask": torch.tensor(pre_mask).long(),
            "post_mask": torch.tensor(post_mask).long()
        }

### DeepLabv3 Methods & Utilities

label_map_xview2 = [
    (0, 0, 0),       # 0: Background
    (0, 255, 0),     # 1: No Damage
    (255, 255, 0),   # 2: Minor Damage
    (255, 165, 0),   # 3: Major Damage
    (255, 0, 0),     # 4: Destroyed
]

def draw_segmentation_map_xview2(outputs, num_classes=5):
    if outputs.dim() == 4:
        outputs = outputs.squeeze(0)
    labels = torch.argmax(outputs, dim=0).cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for class_id in range(num_classes):
        class_mask = labels == class_id
        R, G, B = label_map_xview2[class_id]
        red_map[class_mask], green_map[class_mask], blue_map[class_mask] = R, G, B

    return np.stack([red_map, green_map, blue_map], axis=2)

def load_model(model_name="resnet_50"):
    if model_name == "resnet_50":
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
    model.eval()
    _ = model(torch.randn(1, 3, 520, 520))
    return model, transforms

def modify_for_xview(model, num_classes=5):
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(6, old_conv.out_channels, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:, :, :] = old_conv.weight
    model.backbone.conv1 = new_conv
    in_channels_classifier = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels_classifier, num_classes, kernel_size=1)
    return model

def compute_metrics_deep(model_output, labels, num_classes=5):
    preds_tensor = model_output['out'] if isinstance(model_output, dict) else model_output
    preds = torch.argmax(preds_tensor, dim=1)
    f1_scores, iou_scores, recalls = [], [], []
    for cls in range(num_classes):
        pred_cls, label_cls = (preds == cls), (labels == cls)
        tp = (pred_cls & label_cls).sum().item()
        fp = (pred_cls & ~label_cls).sum().item()
        fn = (~pred_cls & label_cls).sum().item()
        
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if cls > 0:
            f1_scores.append(f1)
            iou_scores.append(iou)
        recalls.append(recall)
    return {"f1": np.mean(f1_scores), "iou": np.mean(iou_scores), "bal_acc": np.mean(recalls)}

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images, masks = batch["image"].to(device), batch["post_mask"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs['out'], masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

### Execution Logic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model, _ = load_model("resnet_50")
model = modify_for_xview(base_model).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Update these paths to your environment
base_folder = "/home/wwg2tp/Downloads"
train_dataset = XView2Dataset(base_folder + "/train/images", base_folder + "/train/labels")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

# --- TEST & VISUALIZATION ---
def test_with_dataloader(model, dataloader, criterion, device, num_batches=1):
    model.eval()
    results = []
    metrics_log = {"loss": [], "f1": [], "iou": [], "bal_acc": []}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
            images, masks = batch["image"].to(device), batch["post_mask"].to(device)
            outputs = model(images)
            loss = criterion(outputs['out'], masks)
            m = compute_metrics_deep(outputs, masks)

            metrics_log["loss"].append(loss.item())
            for key in ["f1", "iou", "bal_acc"]: metrics_log[key].append(m[key])

            if i < num_batches:
                pre_imgs = images[:, :3, :, :].cpu().numpy()
                post_imgs = images[:, 3:, :, :].cpu().numpy()
                preds = torch.argmax(outputs['out'], dim=1).cpu().numpy()
                for b in range(images.shape[0]):
                    results.append({
                        "pre": np.transpose(pre_imgs[b], (1, 2, 0)),
                        "post": np.transpose(post_imgs[b], (1, 2, 0)),
                        "mask": preds[b],
                        "f1": m['f1'],
                        "bal_acc": m['bal_acc']
                    })
    return results, {key: np.mean(val) for key, val in metrics_log.items()}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    base_model, _ = load_model("resnet_50")
    model = modify_for_xview(base_model).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    # Path Setup
    base_folder = "/home/wwg2tp/Downloads"
    train_loader = DataLoader(XView2Dataset(base_folder + "/train/images", base_folder + "/train/labels"), batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(XView2Dataset(base_folder + "/hold/images", base_folder + "/hold/labels"), batch_size=2, shuffle=False, drop_last=True)
    test_loader = DataLoader(XView2Dataset(base_folder + "/test/images", base_folder + "/test/labels"), batch_size=2, shuffle=False, drop_last=True)

    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}

    for epoch in range(EPOCHS):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_m = validate_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        print(f"TRAIN | Loss: {train_m['loss']:.4f} | F1: {train_m['f1']:.4f}")
        print(f"VAL   | Loss: {val_m['loss']:.4f} | F1: {val_m['f1']:.4f} | B_Acc: {val_m['bal_acc']:.4f}")

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            torch.save(model.state_dict(), "deeplab_best_siamese.pth")
            print("New best model saved")

    # Final Testing
    visuals, stats = test_with_dataloader(model, test_loader, criterion, DEVICE, num_batches=2)
    print(f"\nFinal F1: {stats['f1']:.4f} | Final IoU: {stats['iou']:.4f}")