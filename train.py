"""
Script for training a ResNet18 or I3D to classify a pulmonary nodule as benign or malignant.
"""

from dataloader import get_data_loader
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import warnings
import random
import pandas
from experiment_config import config
from datetime import datetime
import shutil
from sklearn.metrics import confusion_matrix
import scipy.stats as st
from sklearn.metrics import roc_auc_score, roc_curve
import sys

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

def make_weights_for_balanced_classes(labels):
    """Making sampling weights for the data samples
    :returns: sampling weights for dealing with class imbalance problem

    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))
    return weights


def calculate_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)

    # Bootstrapping for 95% confidence intervals
    n_bootstraps = 1000
    rng = np.random.RandomState(seed=42)
    bootstrapped_aucs = []

    for _ in range(n_bootstraps):
        # Resample the data
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            # Skip this resample if only one class is present
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(score)

    # Calculate the confidence intervals
    ci_lower = np.percentile(bootstrapped_aucs, 2.5)
    ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    return {"auc": auc, "ci_lower": ci_lower, "ci_upper": ci_upper}

def calculate_sensitivity(y_true, y_pred):
    """
    Computes the sensitivity (recall) at 95% specificity for a classifier.
    
    Parameters:
        y_true (array-like): Ground truth binary labels (0 = benign, 1 = malignant).
        y_pred (array-like): Predicted probability scores from the classifier.

    Returns:
        float: Sensitivity (recall) at 95% specificity.
        float: Decision threshold used to achieve 95% specificity.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # Find the threshold corresponding to 95% specificity (FPR = 1 - specificity)
    target_fpr = 1 - 0.95  # 5% false positive rate
    idx = np.where(fpr <= target_fpr)[0][-1]  # Get the last index where FPR <= 5%
    
    # Extract sensitivity (TPR) and threshold
    sensitivity = tpr[idx]

    return {"sensitivity": sensitivity}

def calculate_specificity(y_true, y_pred):
    """
    Computes the specificity at 95% sensitivity for a classifier.
    
    Parameters:
        y_true (array-like): Ground truth binary labels (0 = benign, 1 = malignant).
        y_pred (array-like): Predicted probability scores from the classifier.

    Returns:
        float: Specificity at 95% sensitivity.
        float: Decision threshold used to achieve 95% sensitivity.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # Find the threshold corresponding to 95% sensitivity (TPR = 0.95)
    target_tpr = 0.95  # Sensitivity (TPR) threshold
    idx = np.where(tpr >= target_tpr)[0][0]  # Get first index where TPR >= 95%
    
    # Extract specificity (1 - FPR) and threshold
    specificity = 1 - fpr[idx]
    
    return {"specificity": specificity}


def train(
    train_csv_path,
    valid_csv_path,
    exp_save_root,

):
    """
    Train a ResNet18 or an I3D model
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info(f"Training with {train_csv_path}")
    logging.info(f"Validating with {valid_csv_path}")

    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)

    print()

    logging.info(
        f"Number of malignant training samples: {train_df.label.sum()}"
    )
    logging.info(
        f"Number of benign training samples: {len(train_df) - train_df.label.sum()}"
    )
    print()
    logging.info(
        f"Number of malignant validation samples: {valid_df.label.sum()}"
    )
    logging.info(
        f"Number of benign validation samples: {len(valid_df) - valid_df.label.sum()}\n"
    )

    # create a training data loader
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))

    train_loader = get_data_loader(
        config.DATADIR,
        train_df,
        mode=config.MODE,
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
        augmentations=config.AUGMENTATIONS,
        aug_settings=config.AUG_SETTINGS
    )

    valid_loader = get_data_loader(
        config.DATADIR,
        valid_df,
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
        augmentations=False,
        aug_settings=config.AUG_SETTINGS,
    )

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0

    for epoch in range(epochs):

        if counter > patience:
            logging.info(f"Model not improving for {patience} epochs")
            break

        logging.info("-" * 10)
        logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # train

        config.model.train()

        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"], batch_data["label"]
            labels = labels.float().to(config.device)
            inputs = inputs.float().to(config.device)
            config.optimizer.zero_grad()
            outputs = config.model(inputs)
            loss = config.loss_function(outputs.squeeze(), labels.squeeze())
            loss.backward()
            config.optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_df) // train_loader.batch_size
            if step % 100 == 0:
                logging.info(
                    "{}/{}, train_loss: {:.4f}".format(step, epoch_len, loss.item())
                )
        epoch_loss /= step
        logging.info(
            "epoch {} average train loss: {:.4f}".format(epoch + 1, epoch_loss)
        )

        # validate

        config.model.eval()

        epoch_loss = 0
        step = 0

        with torch.no_grad():

            y_pred = torch.tensor([], dtype=torch.float32, device=config.device)
            y = torch.tensor([], dtype=torch.float32, device=config.device)
            for val_data in valid_loader:
                step += 1
                val_images = val_data["image"].float().to(config.device)
                val_labels = val_data["label"].float().to(config.device)
                val_images = val_images.to(config.device)
                val_labels = val_labels.float().to(config.device)
                outputs = config.model(val_images)
                loss = config.loss_function(outputs.squeeze(), val_labels.squeeze())
                epoch_loss += loss.item()
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)

                epoch_len = len(valid_df) // valid_loader.batch_size

            epoch_loss /= step
            logging.info(
                "epoch {} average valid loss: {:.4f}".format(epoch + 1, epoch_loss)
            )

            y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
            y = y.data.cpu().numpy().reshape(-1)

            fpr, tpr, _ = metrics.roc_curve(y, y_pred)
            auc_metric = metrics.auc(fpr, tpr)

            auc_nums = calculate_auc(y, y_pred)

            auc_metric = auc_nums["auc"]
            auc_ci_low = auc_nums["ci_lower"]
            auc_ci_high = auc_nums["ci_upper"]

            # AUC 95% CI via DeLong method approximation
            def auc_ci(y_true, y_scores, alpha=0.95):
                n = len(y_scores)
                auc = metrics.roc_auc_score(y_true, y_scores)
                q1 = auc / (2 - auc)
                q2 = 2 * auc**2 / (1 + auc)
                se = np.sqrt((auc * (1 - auc) + (n - 1) * (q1 - auc**2) + (n - 1) * (q2 - auc**2)) / (n**2))
                z = st.norm.ppf(1 - (1 - alpha) / 2)
                lower = max(0.0, auc - z * se)
                upper = min(1.0, auc + z * se)
                return lower, upper

            # auc_ci_low, auc_ci_high = auc_ci(y, y_pred)

            y_true_binary = y.astype(int)

            fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh = thresholds[ix]

            y_pred_binary = (y_pred >= best_thresh).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

            # sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            sensitivity = calculate_sensitivity(y, y_pred)["sensitivity"]
            specificity = calculate_specificity(y, y_pred)["specificity"]

            if auc_metric > best_metric:

                counter = 0
                best_metric = auc_metric
                best_metric_epoch = epoch + 1

                torch.save(
                    config.model.state_dict(),
                    exp_save_root / "best_metric_model.pth",
                )

                metadata = {
                    "train_csv": train_csv_path,
                    "valid_csv": valid_csv_path,
                    "config": config,
                    "best_auc": best_metric,
                    "epoch": best_metric_epoch,
                }
                np.save(
                    exp_save_root / "config.npy",
                    metadata,
                )

                logging.info("saved new best metric model")

            logging.info(
                f"epoch {epoch + 1}: AUC = {auc_metric:.4f} (CI 95%: {auc_ci_low:.4f}–{auc_ci_high:.4f}), "
                f"Sensitivity = {sensitivity:.4f}, Specificity = {specificity:.4f}"
            )

        counter += 1

    logging.info(
        "train completed, best_metric: {:.4f} at epoch: {}".format(
            best_metric, best_metric_epoch
        )
    )


if __name__ == "__main__":        
    # Print hyperparameters
    logging.info(f"Experiment name: {config.EXPERIMENT_NAME}")
    logging.info(f"Experiment mode: {config.MODE}")
    logging.info(f"Batch size: {config.BATCH_SIZE}")
    logging.info(f"Epochs: {config.EPOCHS}")
    logging.info(f"Learning rate: {config.LEARNING_RATE}")   
    logging.info(f"Weight decay: {config.WEIGHT_DECAY}")  
    logging.info(f"Dropout: {config.DROPOUT}")
    logging.info(f"Batch normalization: {config.BATCHNORM}")
    logging.info(f"Rotation: {config.ROTATION}")
    logging.info(f"Translation: {config.TRANSLATION}")
    logging.info(f"Patch size: {config.PATCH_SIZE}")
    logging.info(f"Loss function: {config.loss_function}")
    logging.info(f"Alpha: {config.alpha}")
    logging.info(f"Gamma: {config.gamma}")
    
    if config.MODE != "2D":
        logging.info(f"3D model: {config.MODEL_3D}")
    
    model_str_lines = str(config.model).splitlines()
    # Print the last few lines
    logging.info("\n" + "\n".join(model_str_lines[-15:]))
    
    base_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-{datetime.today().strftime('%Y%m%d')}"
    fold_aucs = []  # Store best AUCs for each fold

    if config.CROSS_VALIDATION:
        logging.info("Training 5 folds")
        base_dir = config.EXPERIMENT_DIR / base_name
        base_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(5):  # 5-fold cross-validation
            logging.info(f"Training fold {fold}/5")
            
            config.EXPERIMENT_NAME = f"{base_name}--fold{fold}"
            exp_save_root = base_dir / f"fold{fold}"
            exp_save_root.mkdir(parents=True, exist_ok=True)

            shutil.copyfile("experiment_config.py", f"{exp_save_root}/experiment_config.py")

            # Train and capture best AUC from metadata
            train(
                train_csv_path=config.CSV_DIR / f"train_fold{fold}.csv",
                valid_csv_path=config.CSV_DIR / f"val_fold{fold}.csv",
                exp_save_root=exp_save_root,
            )

            # Load best AUC from saved metadata
            metadata = np.load(exp_save_root / "config.npy", allow_pickle=True).item()
            fold_aucs.append(metadata["best_auc"])

        logging.info(f"Best AUCs for all folds: {fold_aucs}")
        logging.info(f"Mean AUC across folds: {np.mean(fold_aucs):.4f}")
    else:
        config.EXPERIMENT_NAME = base_name
        exp_save_root = config.EXPERIMENT_DIR / config.EXPERIMENT_NAME
        exp_save_root.mkdir(parents=True, exist_ok=True)

        shutil.copyfile("experiment_config.py", f"{exp_save_root}/experiment_config.py")

        # Train and capture best AUC from metadata
        train(
            train_csv_path=config.CSV_DIR_TRAIN,
            valid_csv_path=config.CSV_DIR_VALID,
            exp_save_root=exp_save_root,
        )