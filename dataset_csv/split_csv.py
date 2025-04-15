import pandas as pd
from sklearn.model_selection import GroupKFold
import numpy as np

def split_csv_stratified_group(csv_file, train_file, val_file, train_size=0.8, test_file=None):
    """
    Splits a CSV file into training, validation, and optionally test sets,
    ensuring stratified distribution of the 'label' column and grouping
    rows based on 'PatientID'.

    Args:
        csv_file (str): Path to the input CSV file.
        train_file (str): Path to the output training CSV file.
        val_file (str): Path to the output validation CSV file.
        train_size (float, optional): The proportion of the dataset to include in the training split. Default is 0.8 (80%).
        test_file (str, optional): Path to the output test CSV file. Defaults to None.
    """
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Check if the 'label' column exists
    if 'label' not in df.columns:
        raise KeyError("The 'label' column is not found in the CSV file.")

    # Check if the 'PatientID' column exists
    if 'PatientID' not in df.columns:
        raise KeyError("The 'PatientID' column is not found in the CSV file.")

    # Determine unique patient IDs
    patient_ids = df['PatientID'].unique()

    # Calculate the number of splits based on the desired train_size
    n_splits = int(1 / (1 - train_size)) if train_size < 1 else 5
    if n_splits < 2:
        raise ValueError("train_size must be less than 1.0")

    # Create a GroupKFold object for splitting while keeping groups (patients) together
    group_kfold = GroupKFold(n_splits=n_splits)
    folds = list(group_kfold.split(patient_ids, groups=patient_ids))

    # Create a dictionary to map PatientIDs to folds
    patient_fold_dict = {}
    for fold_index, (train_index, val_index) in enumerate(folds):
        for val_patient_index in val_index:
            patient_fold_dict[patient_ids[val_patient_index]] = fold_index

    # Add a 'Fold' column to the dataframe using the dictionary
    df['Fold'] = df['PatientID'].map(patient_fold_dict)

    # Separate into train and validation sets.
    train_folds = list(range(n_splits - 1))  # Use all folds except the last one for training
    val_fold = n_splits - 1  # The last fold is used for validation

    train_df = df[df['Fold'].isin(train_folds)].drop(columns=['Fold'])
    val_df = df[df['Fold'] == val_fold].drop(columns=['Fold'])

    # Check the label distribution in the training and validation sets
    print("Training set label distribution:")
    print(train_df['label'].value_counts(normalize=True))
    print("\nValidation set label distribution:")
    print(val_df['label'].value_counts(normalize=True))

    # Save the training and validation sets to CSV files
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    if test_file:
        # For simplicity, let's say the last fold is for testing.
        test_df = df[df['Fold'] == val_fold].drop(columns=['Fold'])
        test_df.to_csv(test_file, index=False)
        print(f"\nTest set saved to {test_file}")
    print(f"\nTraining set saved to {train_file}")
    print(f"Validation set saved to {val_file}")



if __name__ == "__main__":
    # Example usage:
    input_csv_path = 'LUNA25_original.csv'  # Replace with the path to your input CSV file
    train_output_csv_path = 'train.csv'
    val_output_csv_path = 'val.csv'
    train_size = 0.9  # 90% for training, 10% for validation
    split_csv_stratified_group(input_csv_path, train_output_csv_path, val_output_csv_path, train_size=train_size)
