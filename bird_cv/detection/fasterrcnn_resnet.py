from pathlib import Path
from typing import Literal

import csv
import cv2
import importlib
import numpy as np
import msgspec
import time
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from tqdm import tqdm


class DetectorData(msgspec.Struct):
    """Hyperparameters and device settings for Faster R-CNN training.

    Attributes:
        learning_rate: Initial learning rate for SGD.
        num_epochs: Total number of training epochs.
        batch_size: Number of samples per mini-batch.
        weight_decay: L2 regularization coefficient for SGD.
        momentum: Momentum factor for SGD.
        step_size: Epochs between learning-rate reductions.
        gamma: Multiplicative factor applied to the learning rate at each step.
        image_mean: Per-channel ImageNet pixel mean (RGB order).
        image_std: Per-channel ImageNet pixel standard deviation (RGB order).
        image_crop: Target (width, height) to resize every image to.
        device: PyTorch device string; auto-selects CUDA when available.
        pin_memory: Whether to pin DataLoader memory; enabled automatically with CUDA.
    """

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
    # Channel-wise, width-wise, and height-wis[e mean and std respectively
    # ImageNet values, keep if using ResNet (trained on ImageNet)
    image_mean: list = msgspec.field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: list = msgspec.field(default_factory=lambda: [0.229, 0.224, 0.225])
    image_crop: tuple = (224, 224)

    device: str = msgspec.field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory: bool = msgspec.field(
        default_factory=lambda: True if torch.cuda.is_available() else False
    )


class FasterRCNNConfig(msgspec.Struct):
    """Top-level configuration for a Faster R-CNN training run.

    Attributes:
        yolo_data_path: Root directory of the YOLO-formatted dataset, expected
            to contain ``images/`` and ``labels/`` sub-directories.
        output_path: Directory where checkpoints and results are written.
            Created automatically if it does not exist.
        training_params: Hyperparameter bundle forwarded to the training loop.
    """

    yolo_data_path: Path
    output_path: Path

    training_params: DetectorData = msgspec.field(default_factory=DetectorData)

    def __post_init__(self):
        # Create the output path
        self.output_path.mkdir(parents=True, exist_ok=True)


class CustomTensorDataset(Dataset):
    """PyTorch Dataset that wraps pre-loaded image, label, and bounding-box tensors.

    Attributes:
        tensors: Tuple of ``(images, labels, bboxes)`` tensors.
        transforms: Optional torchvision transform applied to each image.
    """

    def __init__(
        self,
        tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        transforms: transforms.Compose | None = None,
    ):
        """Initializes the dataset with tensors and an optional transform.

        Args:
            tensors: Tuple of ``(image_tensor, label_tensor, bbox_tensor)``
                where images have shape ``(N, H, W, C)``, labels ``(N,)``,
                and bboxes ``(N, 4)`` in ``(x0, y0, x1, y1)`` pixel format.
            transforms: A ``torchvision.transforms`` composition applied to
                each image after channel permutation. Defaults to ``None``.
        """
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns a single (image, targets) pair for Faster R-CNN.

        Args:
            index: Integer index into the dataset.

        Returns:
            A tuple ``(image, targets)`` where ``image`` is a float tensor of
            shape ``(C, H, W)`` and ``targets`` is a dict with keys
            ``"boxes"`` (shape ``(1, 4)``) and ``"labels"`` (shape ``(1,)``).
        """
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]

        # Transpose image so the channel is first (Channel, Height, Width)
        image = image.permute(2, 0, 1)

        if self.transforms:
            image = self.transforms(image)

        targets = {
            "boxes": bbox.unsqueeze(0),  # (1,4)
            "labels": label.unsqueeze(0),  # (1,)
        }

        return (image, targets)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            Integer count of samples.
        """
        return self.tensors[0].size(0)


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate a batch of (image, targets) pairs into lists.

    The default PyTorch collator stacks tensors, which fails for variable-size
    target dicts. Faster R-CNN expects a plain list of images and a plain list
    of target dicts, so this function simply unzips the batch.

    Args:
        batch: List of ``(image, targets)`` tuples returned by
            ``CustomTensorDataset.__getitem__``.

    Returns:
        A tuple ``(images, targets)`` where both elements are plain Python
        lists suitable for passing directly to a Faster R-CNN model.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_tensors_from_yolo(
    yolo_data_path: Path,
    cfg: DetectorData,
    split: Literal["train", "val", "test"],
) -> DataLoader:
    """Load a YOLO-formatted dataset split into a PyTorch DataLoader.

    Reads all ``.txt`` label files under ``<yolo_data_path>/labels/<split>/``,
    resolves the corresponding image file, resizes it to ``cfg.image_crop``,
    converts YOLO-normalized box coordinates to absolute pixel coordinates, and
    wraps everything in a ``CustomTensorDataset`` with ImageNet normalization.

    For the ``"test"`` split the image directory is expected to follow a nested
    ``camera/video/cage`` sub-directory structure; for ``"train"`` and ``"val"``
    images live flat under the split directory.

    Args:
        yolo_data_path: Root directory of the YOLO dataset containing
            ``images/`` and ``labels/`` sub-directories.
        cfg: Training hyperparameters supplying ``image_crop``, ``image_mean``,
            ``image_std``, ``batch_size``, and ``pin_memory``.
        split: Dataset partition to load — one of ``"train"``, ``"val"``, or
            ``"test"``.

    Returns:
        A configured ``DataLoader`` that yields ``(images, targets)`` list
        pairs ready for Faster R-CNN.

    Raises:
        FileNotFoundError: If a label file's corresponding image directory
            contains no files.
    """

    images_path = yolo_data_path / "images" / split
    labels_path = yolo_data_path / "labels" / split

    images, labels, bboxes = [], [], []

    # All images must be the same size
    for label_path in tqdm(labels_path.rglob("*.txt")):
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
        if first_image is None:
            raise FileNotFoundError(f"No images found in {image_path}")
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
    apply_transforms = transforms.Compose(
        [
            # Set images to PIL, ToTensor requires a PIL input
            transforms.ToPILImage(),
            # Normalize to 0-1 pixel scale
            transforms.ToTensor(),
            # Normalize around ImageNet pixel mean and std
            transforms.Normalize(mean=cfg.image_mean, std=cfg.image_std),
        ]
    )

    ds = CustomTensorDataset(
        (tensor_images, tensor_labels, tensor_bboxes), transforms=apply_transforms
    )

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=1,  # as requested by medicinebow
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )


def select_resnet_model(
    model_name: str,
    model_weights: str,
    weights: str,
    num_classes: int = 1,
) -> FasterRCNN:
    """Load a torchvision Faster R-CNN variant and replace its head for binary detection.

    Dynamically imports ``model_name`` from ``torchvision.models.detection``,
    initialises it with the requested pretrained weights, and replaces the
    box-predictor fully-connected layer to output ``num_classes + 1`` classes
    (adding a background class).

    Args:
        model_name: Name of the torchvision detection model class, e.g.
            ``"fasterrcnn_resnet50_fpn"``.
        model_weights: Name of the corresponding weights enum class, e.g.
            ``"FasterRCNN_ResNet50_FPN_Weights"``.
        weights: Key into the weights enum, e.g. ``"DEFAULT"``.
        num_classes: Number of foreground object classes. The predictor head
            will be sized to ``num_classes + 1`` to include the background.
            Defaults to ``1``.

    Returns:
        A ``FasterRCNN`` model with a freshly initialised box-predictor head.
    """
    model_module = importlib.import_module("torchvision.models.detection")
    model_class = getattr(model_module, model_name)
    weights_class = getattr(model_module, model_weights)
    model = model_class(weights=weights_class[weights])

    # Modify fully connected layer for the 0/1 task
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes + 1
    )  # bird and background

    return model


def train_one_epoch(
    model: FasterRCNN,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    cfg: DetectorData,
) -> float:
    """Run one full training epoch over the data loader.

    Sets the model to training mode, performs forward and backward passes for
    every mini-batch, updates parameters, and steps the learning-rate
    scheduler at the end of the epoch.

    Args:
        model: Faster R-CNN model to train.
        data_loader: DataLoader yielding ``(images, targets)`` list pairs.
        optimizer: PyTorch optimizer (e.g. ``torch.optim.SGD``).
        scheduler: Learning-rate scheduler stepped once per epoch.
        cfg: Training configuration providing ``device``.

    Returns:
        Average training loss across all batches in the epoch.
    """

    model.train()
    train_losses = []
    tqdm_bar = tqdm(data_loader, total=len(data_loader))

    for images, targets in tqdm_bar:
        # Move tensors to device
        images = list(img.to(cfg.device) for img in images)
        targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        train_loss = loss.item()
        train_losses.append(train_loss)

        # Backwards
        loss.backward()
        optimizer.step()

        tqdm_bar.set_description(desc=f"Training Loss: {loss:.3f}")

    scheduler.step()

    avg_loss = sum(train_losses) / len(train_losses)
    return avg_loss


def run_validation(
    model: FasterRCNN, cfg: DetectorData, data_loader: DataLoader
) -> float:
    """Evaluate the model on a validation split without updating weights.

    Runs the model in training mode so that Faster R-CNN returns losses rather
    than predictions, but wraps each forward pass in ``torch.no_grad()`` to
    skip gradient computation.

    Args:
        model: Faster R-CNN model to evaluate.
        cfg: Training configuration providing ``device``.
        data_loader: DataLoader yielding ``(images, targets)`` list pairs for
            the validation split.

    Returns:
        Average validation loss across all batches.
    """

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


def run_training(
    cfg: DetectorData,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: FasterRCNN,
    output_path: Path,
) -> None:
    """Execute the full training loop with checkpointing and CSV logging.

    Trains for ``cfg.num_epochs`` epochs using SGD with a step learning-rate
    schedule. After each epoch the model is evaluated on the validation set;
    if validation loss improves, the weights are saved to ``output_path/best.pt``.
    Per-epoch metrics (epoch index, wall-clock time, train loss, val loss) are
    appended to ``output_path/results.csv``.

    Args:
        cfg: Hyperparameter bundle controlling the optimizer, scheduler, and
            device placement.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        model: Faster R-CNN model to train; moved to ``cfg.device`` internally.
        output_path: Directory where ``best.pt`` and ``results.csv`` are written.
    """
    # Loss and optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
    )

    # Decay LR by a factor of 0.1 every 7 epochs
    # Consider moving this into the config
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=cfg.gamma
    )

    best_loss = float("inf")

    # Set the model to the device
    model.to(cfg.device)

    # Write training results to a csv
    csv_file = output_path / "results.csv"
    fields = ["epoch", "time", "train_loss", "val_loss"]
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(fields)

        # They also update the lr_scheduler here
        for epoch in range(cfg.num_epochs):
            time_start = time.time()

            print("----------Epoch {}----------".format(epoch + 1))

            avg_train_loss = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=exp_lr_scheduler,
                cfg=cfg,
            )

            avg_val_loss = run_validation(
                model=model,
                data_loader=val_loader,
                cfg=cfg,
            )

            time_end = time.time()
            total_time = time_end - time_start

            # Save results to a csv
            writer.writerow([epoch + 1, total_time, avg_train_loss, avg_val_loss])
            f.flush()

            # Save the best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), output_path / "best.pt")
                print(f"Saved new best model (val loss: {avg_val_loss:.4f})")


def load_and_train(cfg: FasterRCNNConfig) -> None:
    """Orchestrate data loading, model setup, and training for Faster R-CNN.

    Convenience entry point that:

    1. Builds train and validation ``DataLoader``s from the YOLO-formatted
       dataset specified in ``cfg``.
    2. Loads a pretrained ``fasterrcnn_resnet50_fpn`` with ImageNet weights and
       replaces its box-predictor head for single-class detection.
    3. Calls ``run_training`` to execute the full training loop.

    Args:
        cfg: Top-level configuration containing the dataset path, output path,
            and training hyperparameters.
    """

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
