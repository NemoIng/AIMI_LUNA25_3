import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def split_csv_stratified_group(csv_file, train_file, val_file, train_size=0.9):
    """
    Splits a CSV file into training and validation sets,
    ensuring stratified distribution of the 'label' column and grouping
    rows based on 'PatientID'.

    Args:
        csv_file (str): Path to the input CSV file.
        train_file (str): Path to the output training CSV file.
        val_file (str): Path to the output validation CSV file.
        train_size (float): Proportion of data to use for training. Must be < 1.0.
    """
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1 (exclusive)")

    # Read CSV
    df = pd.read_csv(csv_file)

    # Validate required columns
    if 'label' not in df.columns:
        raise KeyError("Missing 'label' column.")
    if 'PatientID' not in df.columns:
        raise KeyError("Missing 'PatientID' column.")

    # Prepare group-level stratification
    group_df = df.groupby('PatientID').agg({
        'label': lambda x: x.iloc[0]  # Assume all labels per patient are the same
    }).reset_index()

    # StratifiedGroupKFold to preserve both label balance and group integrity
    sgkf = StratifiedGroupKFold(n_splits=int(1 / (1 - train_size)), shuffle=True, random_state=42)

    # Generate the first (and only needed) split
    for train_idx, val_idx in sgkf.split(group_df, group_df['label'], groups=group_df['PatientID']):
        train_groups = group_df.iloc[train_idx]['PatientID']
        val_groups = group_df.iloc[val_idx]['PatientID']
        break

    # Filter original DataFrame by group
    train_df = df[df['PatientID'].isin(train_groups)]
    val_df = df[df['PatientID'].isin(val_groups)]

    # Check label distributions
    print("Training set label distribution:")
    print(train_df['label'].value_counts(normalize=True))
    print("\nValidation set label distribution:")
    print(val_df['label'].value_counts(normalize=True))

    # Save to CSV
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print(f"\nTraining set saved to {train_file}")
    print(f"Validation set saved to {val_file}")
    
if __name__ == "__main__":
    split_csv_stratified_group(
        csv_file='dataset_csv/LUNA25_original.csv',
        train_file='dataset_csv/train_split2.csv',
        val_file='dataset_csv/valid_split2.csv',
        train_size=0.9
    )