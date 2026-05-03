import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import importlib
import numpy as np
import msgspec
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from tqdm import tqdm


@dataclass
class DetectorData:

    # Initialize tuning values
    learning_rate: float = 1e-4
    num_epochs: int = 20
    batch_size: int = 32
    weight_decay: float = 0.001
    momentum: float = 0.9

    # Scheduler
    # How many epochs between changing learning rate
    step_size: int = 7
    # Factor by which to scale learning rate at each step
    gamma: float = 0.1

    # Specify ImageNet mean and standard deviation
    # Channel-wise, width-wise, and height-wise mean and std respectively
    # ImageNet values, keep if using ResNet (trained on ImageNet)
    image_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    image_crop: tuple = (224, 224)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True if device == "cuda" else False


class FasterRCNNConfig(msgspec.StrictStruct):

    yolo_data_path: Path
    output_path: Path

    training_params: DetectorData    


class CustomTensorDataset(Dataset):
    # initialize the constructor
    def __init__(self, tensors, transforms=None):
        # Store image, label, and bbox coordinates
        self.tensors = tensors
        # A torchvision.transform instance that will be used to process the image
        self.transforms = transforms

    # Override the Dataset __getitem__ method for our data
    # Note: Will need to update for bird data
    def __getitem__(self, index):
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]

        # Transpose image so the channel is first (Channel, Height, Width)
        image = image.permute(2, 0, 1)

        # Apply any image transformations
        if self.transforms:
            image = self.transforms(image)

        # Return data for faster rcnn resnet50 model
        targets = {
            "boxes": bbox.unsqueeze(0),  # (1,4)
            "labels": label.unsqueeze(0),  # (1,)
        }
        
        # Return
        return (image, targets)
    
    # Override the __len__ method to retugn the size of the image dataset tensor
    def __len__(self):
        return self.tensors[0].size(0)


def collate_fn(batch):
    # Default collator tries to stack everything into single tensors,
    # which fails on target dicts. Faster R-CNN expects a list of
    # images and a list of target dicts, so we just unzip and return as lists.
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_tensors_from_yolo(
    yolo_data_path: Path,
    cfg: DetectorData,
    split: Literal["train", "val", "test"],
) -> DataLoader:
    
    images_path = yolo_data_path / "images" / split
    labels_path = yolo_data_path / "labels" / split

    images, labels, bboxes = [], [], []

    # All images must be the same size
    for label_path in tqdm(labels_path.rglob(f"*.txt")):

        with label_path.open("r") as f:
            label = f.read()

        if not label:
            continue
        else:
            label = [float(x) for x in label.split()]

        (category, x0, y0, w, h) = label

        # Image preprocessing
        if split == "test":
            cage = label_path.parent.stem
            video = label_path.parent.parent.stem
            camera = label_path.parent.parent.parent.stem
            image_path = images_path / camera / video / cage
        else:
            image_path = images_path

        data_name = label_path.stem

        # Automatically find the image extension
        first_image = next(image_path.rglob("*"), None)
        image_extension = first_image.suffix

        image = cv2.imread(str(image_path / f"{data_name}{image_extension}"))
        # Rescale the image
        image = cv2.resize(image, cfg.image_crop)

        # Get the image height and width
        im_height, im_width = image.shape[:2]
        x0_not_norm = (x0 * im_width) - (0.5 * w * im_width)
        y0_not_norm = (y0 * im_height) - (0.5 * h * im_height)
        x1_not_norm = (x0 * im_width) + (0.5 * w * im_width)
        y1_not_norm = (y0 * im_height) + (0.5 * h * im_height)

        # Append data
        images.append(image)
        labels.append(int(category))
        bboxes.append((x0_not_norm, y0_not_norm, x1_not_norm, y1_not_norm))

    # Convert to numpy arrays for faster processing
    images = np.array(images, dtype="float32")
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")

    # Convert numpy arrays to pytorch tensors
    tensor_images = torch.tensor(images)
    tensor_labels = torch.tensor(labels)
    tensor_bboxes = torch.tensor(bboxes)

    # Assign transforms to the preprocess the image tensors
    apply_transforms = transforms.Compose([
        # Set images to PIL, ToTensor requires a PIL input
        transforms.ToPILImage(),
        # Normalize to 0-1 pixel scale
        transforms.ToTensor(),
        # Normalize around ImageNet pixel mean and std
        transforms.Normalize(mean=cfg.image_mean, std=cfg.image_std)
    ])

    ds = CustomTensorDataset((tensor_images, tensor_labels, tensor_bboxes), transforms=apply_transforms)

    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=os.cpu_count(), pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )


def select_resnet_model(
    model_name, 
    model_weights,
    weights,
    num_classes=1,
):
    model_module = importlib.import_module("torchvision.models.detection")
    model_class = getattr(model_module, model_name)
    weights_class = getattr(model_module, model_weights)
    model = model_class(weights=weights_class[weights])

    # Modify fully connected layer for the 0/1 task
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1) # bird and background

    return model



def train_one_epoch(
    model: FasterRCNN,
    data_loader: DataLoader,
    optimizer,
    scheduler,
    cfg: DetectorData
):

    model.train()

    tqdm_bar = tqdm(data_loader, total=len(data_loader))

    for images, targets in (tqdm_bar):

        # Move tensors to device
        images = list(img.to(cfg.device) for img in images)
        targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        loss_val = loss.item()

        # Backwards
        loss.backward()
        optimizer.step()

        tqdm_bar.set_description(desc=f"Training Loss: {loss:.3f}")

    scheduler.step()


def run_validation(model, cfg, data_loader):

    model.train()
    val_losses = []
    tqdm_bar = tqdm(data_loader, total=len(data_loader))

    for images, targets in tqdm_bar:

        # Move tensors to device
        images = list(img.to(cfg.device) for img in images)
        targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            losses = model(images, targets)

        loss = sum(loss for loss in losses.values())
        loss_val = loss.item()
        val_losses.append(loss_val)

        tqdm_bar.set_description(desc=f"Validation Loss: {loss:.4f}")

    avg_loss = sum(val_losses) / len(val_losses)

    return avg_loss


# Run training, validation, and test
def run_training(
    cfg: DetectorData,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: FasterRCNN,
    output_path: Path,
):
    #Loss and optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay, 
        momentum=cfg.momentum,
    )  

    # Decay LR by a factor of 0.1 every 7 epochs
    # Consider moving this into the config
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    best_loss = float("inf")

    # Set the model to the device
    model.to(cfg.training_params.device)

    # They also update the lr_scheduler here
    for epoch in range(cfg.num_epochs):

        print("----------Epoch {}----------".format(epoch+1))

        train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            cfg=cfg
        )

        avg_loss = run_validation(
            model=model, 
            data_loader=val_loader,
            cfg=cfg,
        )

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path / "best.pt")
            print(f"Saved new best model (val loss: {avg_loss:.4f})")
        

def load_and_train(cfg: FasterRCNNConfig):

    train_loader = load_tensors_from_yolo(
        yolo_data_path=cfg.yolo_data_path,
        cfg=cfg.training_params,
        split="train",
    )

    val_loader = load_tensors_from_yolo(
        yolo_data_path=cfg.yolo_data_path,
        cfg=cfg.training_params,
        split="val",
    )

    resnet = select_resnet_model(
        model_name="fasterrcnn_resnet50_fpn",
        model_weights="FasterRCNN_ResNet50_FPN_Weights",
        weights="DEFAULT",
    )

    run_training(
        cfg=cfg.training_params,
        train_loader=train_loader,
        val_loader=val_loader,
        model=resnet,
        output_path=cfg.output_path,
    )
