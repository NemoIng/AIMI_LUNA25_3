from pathlib import Path
from loss_functions import ComboLoss, AsymmetricFocalTverskyLoss as AFTLoss
from FocalLoss import FocalLoss
import torch

from models.model_2d import ResNet34, ResNet34_exp
from models.model_3d_base import I3D
from models.model_3d_resnet import ResNet3D
from models.model_3d_densenet import DenseNet3D

class Configuration(object):
    def __init__(self) -> None:
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        # Working directory
        self.WORKDIR = Path(".")
        self.RESOURCES = Path("resources")
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )
        
        # Data parameters
        # Path to the nodule blocks folder provided for the LUNA25 training data. 
        self.DATADIR = Path("luna25_nodule_blocks")

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        self.EXPERIMENT_NAME = "3d_scheduler_test" # Name of the experiment
        self.MODE = "3D" # 2D or 3D
        self.MODEL_3D = "3DRes" # 3D model to use: I3D, 3DRes, or 3DRes

        self.EXPERIMENT_NAME = f"{self.MODE}_{self.EXPERIMENT_NAME}"
        
        self.alpha = 0.3
        self.gamma = 2.0
        self.loss_function = ComboLoss(alpha=self.alpha, gamma=self.gamma, dice_weight=0.3).to(self.device)
        
        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 2
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 32

        self.ROTATION = ((-180, 180), (-180, 180), (-180, 180))
        self.TRANSLATION = True
        self.EPOCHS = 35
        self.PATIENCE = 10
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 4e-5
        self.WEIGHT_DECAY = 3e-3
 
        # Other parameters
        self.DROPOUT = 0.2
        self.BATCHNORM = False
        self.CROSS_VALIDATION = False
        self.CROSS_VALIDATION_FOLDS = 5
       
        self.AUGMENTATIONS = True
        self.AUG_SETTINGS = {
            # 2D
            "horizontal_flip": 0.4,
            "rotation_90": 0.4,
            "brightness_shift": 0.4,
            "gaussian_noise": 0.2,
            "coarse_dropout": 0.2,
            "zoom": 0.2,
            "shear": 0.2,
            # 3D
            "flip_x": 0.3,
            "flip_y": 0.3,
            "flip_z": 0.3,
            "brightness_shift_3d": 0.2,
            "gaussian_noise_3d": 0.2,
            "coarse_dropout_3d": 0.2,
        }
        
        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path("dataset_csv") if not self.CROSS_VALIDATION else Path("dataset_csv/cross_validation")
        
        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid.csv" # Path to the validation CSV
        
        if self.CROSS_VALIDATION: self.EXPERIMENT_NAME = f"{self.EXPERIMENT_NAME}_CV"
        
        # set model
        if self.MODE == "2D":
            self.model = ResNet34_exp(dropout=self.DROPOUT, batchnorm=self.BATCHNORM).to(self.device)
        elif self.MODE == "3D" and self.MODEL_3D == "I3D":
            self.model = I3D(
                num_classes=1,
                input_channels=3,
                pre_trained=True,
                freeze_bn=False,
            ).to(self.device)
        elif self.MODE == '3D' and self.MODEL_3D == "3DRes":
            self.model = ResNet3D(num_classes=1, input_channels=3, pretrained=True, 
                                  freeze_bn=False, dropout=self.DROPOUT).to(self.device)
        elif self.MODE == '3D' and self.MODEL_3D == "3DDense":
            self.model = DenseNet3D(num_classes=1, input_channels=3,
                dropout=self.DROPOUT).to(self.device)
                                           
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.EPOCHS, eta_min=1e-7
        )

config = Configuration()