"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import dataloader
import torch
import torch.nn as nn
from torchvision import models
from models.model_3d import I3D
from models.model_2d import ResNet34
import os
import math
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, mode="3D", suppress_logs=False, model_name="LUNA25-3D-lowLR-highWD-noDice-rot=90-3D-20250507"):

        self.size_px = 64
        self.size_mm = 50

        self.model_name = model_name
        self.mode = mode
        self.suppress_logs = suppress_logs

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.mode == "2D":
            self.model_2d = ResNet34(weights=None).to(device)
        elif self.mode == "3D":
            self.model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).to(device)

        self.model_root = "/opt/app/results/"

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch_lung = dataloader.clip_and_scale(patch.copy(), -1000, 400)
        patch_mediastinum = dataloader.clip_and_scale(patch.copy(), 40, 400)
        patch_soft_tissue = dataloader.clip_and_scale(patch.copy(), -160, 240)

        patch_combined = np.stack([patch_lung, patch_mediastinum, patch_soft_tissue], axis=0)
        return patch_combined


    def _process_model(self, mode):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_2d
        else:
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_3d

        nodules = []

        for _coord in self.coords:

            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nodules = torch.from_numpy(nodules).to(device)

        ckpt = torch.load(
            os.path.join(self.model_root, self.model_name, "best_metric_model.pth"),
            map_location=torch.device("cpu")
        )

        model.load_state_dict(ckpt)
        model.eval()

        # Convert grayscale (1-channel) input to 3-channel if needed
        if nodules.dim() == 4 and nodules.shape[1] == 1:
            # For 2D: [B, 1, H, W] -> [B, 3, H, W]
            nodules = nodules.repeat(1, 3, 1, 1)
        elif nodules.dim() == 6 and nodules.shape[2] == 1:
            # For 3D: [B, C, 1, D, H, W] -> [B, C, D, H, W]
            nodules = nodules.squeeze(2)

        logits = model(nodules)
        logits = logits.data.cpu().numpy()

        logits = np.array(logits)
        return logits

    def predict(self):

        logits = self._process_model(self.mode)

        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits