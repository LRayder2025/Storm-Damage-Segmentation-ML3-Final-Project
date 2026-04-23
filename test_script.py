# %% [markdown]
# # xView2 CNN Project

# %% [markdown]
# ### Imports; Download Data

# %%
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import manual_seed as torch_manual_seed
from torch.cuda import max_memory_allocated, set_device, manual_seed_all
from torch.backends import cudnn
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("CUDA device name:", torch.cuda.get_device_name(0))

# %%
def setup_seed(seed):
    torch_manual_seed(seed)
    manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False # I added this one to try and make it deterministic

SEED = 42
setup_seed(SEED)

# %%
# This should download and unzip the data and store it in the content folder during your runtime
# full link: https://drive.google.com/file/d/1kMC2PCTyWoOiL0AItssA7Grh4CSoPO2K/view?usp=sharing
!pip install gdown
!gdown --id 1kMC2PCTyWoOiL0AItssA7Grh4CSoPO2K -O data.zip
!unzip -q data.zip -d /content/data
print("finished unzipping")

# %% [markdown]
# ### Dataset loader; View Images

# %%
class XView2Dataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir

        # anchor on post-disaster images
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
            "un-classified": 0   # sets building damage that was not classified to background which is not ideal but works for now
        }

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        img = cv2.imread(path)

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        return img

    def _load_post_mask(self, json_path, shape):
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)

        with open(json_path) as f:
            data = json.load(f)

        for feature in data["features"]["xy"]:
            props = feature["properties"]

            if "subtype" not in props:
                continue

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
            cv2.fillPoly(mask, [pts], 1)  # binary mask

        return mask

    def __getitem__(self, idx):
        base = self.files[idx]

        # handle jpg or png automatically
        for ext in [".jpg", ".png"]:
            pre_path = os.path.join(self.image_dir, base + "_pre_disaster" + ext)
            post_path = os.path.join(self.image_dir, base + "_post_disaster" + ext)
            if os.path.exists(pre_path):
                break

        pre_img = self._load_image(pre_path)
        post_img = self._load_image(post_path)

        # stack → 6 channels
        image = np.concatenate([pre_img, post_img], axis=2)

        pre_label_path = os.path.join(self.label_dir, base + "_pre_disaster.json")
        post_label_path = os.path.join(self.label_dir, base + "_post_disaster.json")

        pre_mask = self._load_pre_mask(pre_label_path, pre_img.shape[:2])
        post_mask = self._load_post_mask(post_label_path, pre_img.shape[:2])

        # convert to tensors
        image = torch.tensor(image).permute(2, 0, 1).float()

        return {
            "image": image,
            "pre_mask": torch.tensor(pre_mask).long(),
            "post_mask": torch.tensor(post_mask).long()
        }

# %%
#base_folder = "/content/data/xview2_jpeg"
base_folder = "/home/wwg2tp/Downloads"
dataset = XView2Dataset(base_folder+"/hold/images", base_folder+"/hold/labels")
sample = dataset[0]

print(sample["image"].shape)  # should be (6, H, W)
print(sample["post_mask"].shape)   # should be (H, W)
print(sample["post_mask"].unique())  # should show 0–4

# %%
def visualize_full(sample):
    img = sample["image"]
    pre_mask = sample["pre_mask"].numpy()
    post_mask = sample["post_mask"].numpy()

    pre = img[:3].permute(1,2,0).numpy()
    post = img[3:].permute(1,2,0).numpy()

    cmap = plt.cm.get_cmap("jet", 5)

    fig, axes = plt.subplots(2, 3, figsize=(14, 11))

    # --- PRE ---
    axes[0,0].imshow(pre)
    axes[0,0].set_title("Pre Image")
    axes[0,0].axis("off")

    axes[0,1].imshow(pre)
    axes[0,1].imshow(pre_mask, cmap="gray", alpha=0.5)
    axes[0,1].set_title("Pre + Building Mask")
    axes[0,1].axis("off")

    axes[0,2].imshow(pre_mask, cmap="gray")
    axes[0,2].set_title("Pre Mask Only")
    axes[0,2].axis("off")

    # --- POST ---
    axes[1,0].imshow(post)
    axes[1,0].set_title("Post Image")
    axes[1,0].axis("off")

    axes[1,1].imshow(post)
    axes[1,1].imshow(post_mask, cmap=cmap, alpha=0.5, vmin=0, vmax=4)
    axes[1,1].set_title("Post + Damage Mask")
    axes[1,1].axis("off")

    axes[1,2].imshow(post_mask, cmap=cmap, vmin=0, vmax=4)
    axes[1,2].set_title("Post Mask Only")
    axes[1,2].axis("off")

    # # --- COMPARISON ---
    # axes[2,0].imshow(pre_mask, cmap="gray")
    # axes[2,0].set_title("Buildings")
    # axes[2,0].axis("off")

    # axes[2,1].imshow(post_mask, cmap=cmap, vmin=0, vmax=4)
    # axes[2,1].set_title("Damage Classes")
    # axes[2,1].axis("off")

    # axes[2,2].imshow(pre)
    # axes[2,2].imshow(post_mask, cmap=cmap, alpha=0.5)
    # axes[2,2].set_title("Damage Overlay")
    # axes[2,2].axis("off")

    # --- LEGEND ---
    class_names = [
        "Background",
        "No Damage",
        "Minor Damage",
        "Major Damage",
        "Destroyed"
    ]

    colors = [cmap(i) for i in range(5)]
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(5)]

    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.02)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

sample = dataset[0]
visualize_full(sample)

# %% [markdown]
# ### DataLoader

# %%
base_folder = "/content/data/xview2_jpeg"

train_dataset = XView2Dataset(base_folder+"/tier1/images_jpeg", base_folder+"/tier1/labels")
val_dataset   = XView2Dataset(base_folder+"/test/images_jpeg", base_folder+"/test/labels")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False)

# %% [markdown]
# ## DeepLabv3 Alternate Segmentation Method

# %%
# Five XView2 Classes --> numbers correspond to colors
label_map_xview2 = [
    (0, 0, 0),           # 0: Background (black)
    (0, 255, 0),         # 1: No Damage (green)
    (255, 255, 0),       # 2: Minor Damage (yellow)
    (255, 165, 0),       # 3: Major Damage (orange)
    (255, 0, 0),         # 4: Destroyed (red)
]

# "Utility functions for processing / plotting"
def draw_segmentation_map_xview2(outputs, num_classes=5):
    """
    Converts model logits to RGB segmentation map using xView2 damage classes.

    returns: segmentation_map --> np.ndarray
        RGB image with damage classes colored, shape (H, W, 3), dtype uint8
    """
    # Handle batch dimension if present
    if outputs.dim() == 4:
        outputs = outputs.squeeze(0)

    # Get predicted class for each pixel
    labels = torch.argmax(outputs, dim=0).cpu().numpy()

    # Initialize R, G, B channel arrays
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    # Fill each channel based on predicted class
    for class_id in range(num_classes):
        class_mask = labels == class_id

        R, G, B = label_map_xview2[class_id]

        red_map[class_mask] = R
        green_map[class_mask] = G
        blue_map[class_mask] = B

    # Stack into (H, W, 3) RGB image
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image, alpha=0.7, beta=0.3):
    """
    Blends original image with segmentation map.

    returns: overlay --> np.ndarray
        Blended image in BGR format
    """
    # Convert to numpy if PIL Image
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)

    # Convert RGB → BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    seg_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    # Weighted blend
    overlay = cv2.addWeighted(image_bgr, alpha, seg_bgr, beta, 0)

    return overlay

# %%
# Set up a bunch of different models to try
import PIL
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import (
                                             DeepLabV3_ResNet50_Weights,
                                             DeepLabV3_ResNet101_Weights,
                                             DeepLabV3_MobileNet_V3_Large_Weights
                                             )

def load_model(model_name: str):
    if model_name.lower() not in ("mobilenet", "resnet_50", "resnet_101"):
        raise ValueError("'model_name' should be one of ('mobilenet', 'resnet_50', 'resnet_101')")

    if model_name == "resnet_50":
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    elif model_name == "resnet_101":
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        transforms = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    else:
        model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        transforms = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

    model.eval()

    # Warmup run

    _ = model(torch.randn(1, 3, 520, 520))

    return model, transforms

def modify_for_xview(model, num_classes=5):
    # Modify the first convolutional layer for 6 channels (DeepLab is 3 normally)
    old_conv = model.backbone.conv1

    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # Initialize weights: copy pre-trained 3-channel weights to both halves
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:, :, :] = old_conv.weight

    model.backbone.conv1 = new_conv

    # Modify the classifier heads for the new number of classes
    # DeepLabV3 models have a list of layers in 'classifier', the last is index 4
    in_channels_classifier = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels_classifier, num_classes, kernel_size=1)

    # Upddate "auxiliary classifier" if it exists
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)

    return model

# %%
def compute_metrics_deep(model_output, labels, num_classes=5):
    preds_tensor = model_output['out'] if isinstance(model_output, dict) else model_output
    preds = torch.argmax(preds_tensor, dim=1)

    f1_scores, iou_scores, recalls = [], [], []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)

        tp = (pred_cls & label_cls).sum().item()
        fp = (pred_cls & ~label_cls).sum().item()
        fn = (~pred_cls & label_cls).sum().item()

        # --- F1 Score ---
        f1_denom = (2 * tp + fp + fn)
        f1 = (2 * tp) / f1_denom if f1_denom > 0 else 0

        # --- IoU ---
        iou_denom = (tp + fp + fn)
        iou = tp / iou_denom if iou_denom > 0 else 0

        # --- Recall (Used for Balanced Accuracy) ---
        recall_denom = (tp + fn)
        recall = tp / recall_denom if recall_denom > 0 else 0

        # We skip background (0) for F1 and IoU to focus on damage
        if cls > 0:
            f1_scores.append(f1)
            iou_scores.append(iou)

        # Balanced Accuracy usually includes all classes
        recalls.append(recall)

    return {
        "f1": np.mean(f1_scores) if f1_scores else 0.0,
        "iou": np.mean(iou_scores) if iou_scores else 0.0,
        "bal_acc": np.mean(recalls) if recalls else 0.0
    }

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    metrics_log = {"f1": [], "iou": [], "bal_acc": []}

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["post_mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Get all metrics
        m = compute_metrics_deep(outputs, masks)
        for key in metrics_log: metrics_log[key].append(m[key])

        loss = criterion(outputs['out'], masks)
        if 'aux' in outputs:
          loss += 0.4 * criterion(outputs['aux'], masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", f1=f"{m['f1']:.2f}", iou=f"{m['iou']:.2f}")

    return {
      "loss": running_loss / len(dataloader),
      "f1": np.mean(metrics_log["f1"]),
      "iou": np.mean(metrics_log["iou"]),
      "bal_acc": np.mean(metrics_log["bal_acc"])
    }

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    metrics_log = {"f1": [], "iou": [], "bal_acc": []}

    pbar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["post_mask"].to(device)

            outputs = model(images)
            m = compute_metrics_deep(outputs, masks)
            for key in metrics_log: metrics_log[key].append(m[key])

            loss = criterion(outputs['out'], masks)
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{m['iou']:.2f}")

    return {
        "loss": running_loss / len(dataloader),
        "f1": np.mean(metrics_log["f1"]),
        "iou": np.mean(metrics_log["iou"]),
        "bal_acc": np.mean(metrics_log["bal_acc"])
    }

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 2
best_val_loss = float('inf')

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

torch.manual_seed(42)

base_model, _ = load_model("resnet_50")
model = modify_for_xview(base_model)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

base_folder = "/home/wwg2tp/Downloads"
train_dataset = XView2Dataset(base_folder + "/train/images", base_folder + "/train/labels")
val_dataset = XView2Dataset(base_folder + "/hold/images", base_folder + "/hold/labels")
test_dataset = XView2Dataset(base_folder + "/test/images", base_folder + "/test/labels")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=True)

history = {
    "train_loss": [], "train_f1": [], "train_iou": [], "train_bal_acc": [],
    "val_loss": [], "val_f1": [], "val_iou": [], "val_bal_acc": []
}

for epoch in range(EPOCHS):
    train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_metrics = validate_one_epoch(model, val_loader, criterion, DEVICE)

    # Store everything
    for key in ["loss", "f1", "iou", "bal_acc"]:
        history[f"train_{key}"].append(train_metrics[key])
        history[f"val_{key}"].append(val_metrics[key])

    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    print(f"TRAIN | Loss: {train_metrics['loss']:.4f} | F1: {train_metrics['f1']:.4f} | IoU: {train_metrics['iou']:.4f}")
    print(f"VAL   | Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['f1']:.4f} | IoU: {val_metrics['iou']:.4f} | B_Acc: {val_metrics['bal_acc']:.4f}")

    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        torch.save(model.state_dict(), "deeplab_best_siamese.pth")
        print("New best model saved")

# %%
# Print final losses
print("\n" + "="*50)
print("TRAINING COMPLETE - LOSS HISTORY")
print("="*50)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1:2d} | Train Loss: {history['train_loss'][epoch]:.4f} | Val Loss: {history['val_loss'][epoch]:.4f}")

# %%
# Plot Loss Curve
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# F1 Score
axes[0, 1].plot(history['train_f1'], label='Train F1', linewidth=2)
axes[0, 1].plot(history['val_f1'], label='Val F1', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].set_title('F1 Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# IoU
axes[1, 0].plot(history['train_iou'], label='Train IoU', linewidth=2)
axes[1, 0].plot(history['val_iou'], label='Val IoU', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('IoU')
axes[1, 0].set_title('Intersection over Union')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Balanced Accuracy
axes[1, 1].plot(history['train_bal_acc'], label='Train Bal Acc', linewidth=2)
axes[1, 1].plot(history['val_bal_acc'], label='Val Bal Acc', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Balanced Accuracy')
axes[1, 1].set_title('Balanced Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
import torch.nn.functional as F
import PIL.Image
import matplotlib.patches as mpatches

def test_trained_model(model, image_dir, label_dir, num_images=5, device=None):
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_files = os.listdir(image_dir)
    image_bases = sorted(set(
        f.replace("_pre_disaster.jpg", "").replace("_post_disaster.jpg", "")
        for f in all_files if "_pre_disaster" in f or "_post_disaster" in f
    ))

    selected_bases = np.random.choice(image_bases, min(num_images, len(image_bases)), replace=False)
    results = []
    # Dictionary to track all metrics
    test_metrics_log = {"f1": [], "iou": [], "bal_acc": []}

    for img_base in tqdm(selected_bases, desc="Testing"):
        try:
            pre_path = os.path.join(image_dir, img_base + "_pre_disaster.jpg")
            post_path = os.path.join(image_dir, img_base + "_post_disaster.jpg")
            label_path = os.path.join(label_dir, img_base + "_post_disaster.png")

            pre_img = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            post_img = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            true_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            true_mask_tensor = torch.from_numpy(true_mask).long().to(device).unsqueeze(0)

            # Inference
            stacked = np.concatenate([pre_img, post_img], axis=2)
            input_tensor = torch.from_numpy(stacked).permute(2, 0, 1).float().unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)

                # --- CHANGE: Use the multi-metric function ---
                m = compute_metrics_deep(output, true_mask_tensor, num_classes=5)
                for key in test_metrics_log: test_metrics_log[key].append(m[key])

                mask_np = torch.argmax(output['out'], dim=1).squeeze(0).cpu().numpy()

            results.append({
                "name": img_base,
                "pre": (pre_img * 255).astype(np.uint8),
                "post": (post_img * 255).astype(np.uint8),
                "mask": mask_np,
                "f1": m['f1'],   # Save F1 for display
                "iou": m['iou']   # Save IoU for display
            })

        except Exception as e:
            print(f"Error on {img_base}: {e}")

    # Calculate averages
    final_stats = {key: np.mean(val) for key, val in test_metrics_log.items()}

    print(f"\n--- Test Set Performance ---")
    print(f"Mean F1: {final_stats['f1']:.4f} | Mean IoU: {final_stats['iou']:.4f} | Balanced Acc: {final_stats['bal_acc']:.4f}")

    return results, final_stats

def test_with_dataloader(model, dataloader, criterion, device, num_batches=1):
    model.eval()
    results = []
    metrics_log = {"loss": [], "f1": [], "iou": [], "bal_acc": []}

    with torch.no_grad():
        # We only take a few batches for visualization, but calculate metrics
        for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
            images = batch["image"].to(device)
            masks = batch["post_mask"].to(device) # Ground truth

            # 1. Inference
            outputs = model(images)

            # 2. Calculate Metrics
            loss = criterion(outputs['out'], masks)
            m = compute_metrics_deep(outputs, masks)

            metrics_log["loss"].append(loss.item())
            for key in ["f1", "iou", "bal_acc"]:
                metrics_log[key].append(m[key])

            # 3. Store first few for display
            if i < num_batches:
                # Convert back to numpy for plotting
                # images is (B, 6, H, W). Split back to pre/post (B, 3, H, W)
                pre_imgs = images[:, :3, :, :].cpu().numpy()
                post_imgs = images[:, 3:, :, :].cpu().numpy()
                preds = torch.argmax(outputs['out'], dim=1).cpu().numpy()

                for b in range(images.shape[0]):
                    results.append({
                        "pre": np.transpose(pre_imgs[b], (1, 2, 0)),
                        "post": np.transpose(post_imgs[b], (1, 2, 0)),
                        "mask": preds[b],
                        "f1": m['f1']
                    })

    final_stats = {key: np.mean(val) for key, val in metrics_log.items()}
    return results, final_stats

def display_test_results(results):
    # xView2 Color Map
    colors = {
        0: [0, 0, 0],       # Background
        1: [0, 255, 0],     # No Damage
        2: [255, 255, 0],   # Minor
        3: [255, 165, 0],   # Major
        4: [255, 0, 0]      # Destroyed
    }

    n = len(results)
    if n == 0:
        print("No results to display.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))

    # Ensure axes is always 2D even if n=1
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, res in enumerate(results):
        # 1. Process Mask
        h, w = res["mask"].shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for c, color in colors.items():
            rgb_mask[res["mask"] == c] = color

        # 2. Process Images (Dataloader outputs are usually 0-1 float)
        # We clip to ensure matplotlib doesn't complain about values slightly out of range
        pre_img = np.clip(res["pre"], 0, 1)
        post_img = np.clip(res["post"], 0, 1)

        # 3. Plotting
        axes[i, 0].imshow(pre_img)
        axes[i, 0].set_title(f"Sample {i+1} - Pre")

        axes[i, 1].imshow(post_img)
        axes[i, 1].set_title("Post-Disaster")

        axes[i, 2].imshow(rgb_mask)
        # Pulling metrics we stored in the results list
        f1 = res.get('f1', 0.0)
        ba = res.get('bal_acc', 0.0)
        axes[i, 2].set_title(f"Pred (F1: {f1:.2f}, BalAcc: {ba:.2f})")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

# --- RUN IT ---
# Use 'test' directory
TEST_IMAGES = "/content/data/xview2_jpeg/test/images_jpeg"
TEST_LABELS = "/content/data/xview2_jpeg/test/labels"

# Run it
# OLD --> visuals, stats = test_trained_model(model, TEST_IMAGES, TEST_LABELS, num_images=5)
visuals, stats = test_with_dataloader(model, test_loader, criterion, DEVICE, num_batches=2)

# View Images
#display_test_results(visuals)

# Get Metrics
print(f"Final F1: {stats['f1']:.4f}")
print(f"Final IoU: {stats['iou']:.4f}")

# Get Dice & Accuracy
dice_loss = 1 - stats['f1']
print(f"Calculated Dice Loss: {dice_loss:.4f}")
bal_acc = stats['bal_acc']
print(f"Overall Balanced Accuracy: {bal_acc:.4f}")

# %%
display_test_results(visuals)

# %%


# %%


# %%



