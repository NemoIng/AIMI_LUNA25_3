from pathlib import Path
import ComboLoss
import torch

from models.model_2d import ResNet34, ResNet34_exp
from models.model_3d import I3D

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

        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path("dataset_csv")
        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid.csv" # Path to the validation CSV

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        # self.EXPERIMENT_NAME = "LUNA25-3D-Combo" # Name of the experiment
        # self.MODE = "3D" # 2D or 3D
        self.EXPERIMENT_NAME = "LUNA25-2D-test-lowLR-highWD-noDice" # Name of the experiment
        self.MODE = "2D" # 2D or 3D

        self.alpha = 0.3
        self.gamma = 2.0
        self.dice_weight = 0.0
        self.loss_function = ComboLoss.ComboLoss(alpha=self.alpha, gamma=self.gamma, dice_weight=self.dice_weight).to(self.device)

        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 2
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 32
        self.ROTATION = ((-180, 180), (-180, 180), (-180, 180)) #((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True
        self.EPOCHS = 30
        self.PATIENCE = 7
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 2e-5
        self.WEIGHT_DECAY = 5e-3
        
        # Model parameters
        self.DROPOUT = [0, 0]# [0.3, 0.3]
        self.BATCHNORM = True 
        
        # set model
        if self.MODE == "2D":
            self.model = ResNet34_exp(dropout=self.DROPOUT, batchnorm=self.BATCHNORM).to(self.device)
        elif self.MODE == "3D":
            self.model = I3D(
                num_classes=1,
                input_channels=3,
                pre_trained=True,
                freeze_bn=True,
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.EPOCHS
        )

config = Configuration()
