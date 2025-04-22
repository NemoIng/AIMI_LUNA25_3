import numpy as np
import matplotlib.pyplot as plt
import os
import random

def show_npy_image(file_path, slice_index=None):
    """
    Display a slice of a .npy medical image.

    :param file_path: Path to the .npy file
    :param slice_index: Index of the slice to display (default is the middle slice)
    """
    # Load the .npy file
    image_array = np.load(file_path)

    # Determine the slice to display
    if slice_index is None:
        slice_index = image_array.shape[0] // 2  # Default to the middle slice

    # Check if the slice index is valid
    if slice_index < 0 or slice_index >= image_array.shape[0]:
        raise ValueError(f"Slice index {slice_index} is out of range for this image.")

    # Display the selected slice
    plt.imshow(image_array[slice_index], cmap='gray')
    plt.title(f"Slice {slice_index}")
    plt.axis('off')
    plt.show()

def get_random_npy_file(folder_path):
    """
    Get a random .npy file from the specified folder.

    :param folder_path: Path to the folder containing .npy files
    :return: Path to a random .npy file
    """
    # List all .npy files in the folder
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if not npy_files:
        raise FileNotFoundError("No .npy files found in the specified folder.")
    
    # Select a random file
    return os.path.join(folder_path, random.choice(npy_files))

# Example usage
if __name__ == "__main__":
    folder_path = 'luna25_nodule_blocks/image'  # Replace with your folder path
    try:
        random_file_path = get_random_npy_file(folder_path)
        print(f"Opening random file: {random_file_path}")
        show_npy_image(random_file_path)
    except Exception as e:
        print(f"Error: {e}")
