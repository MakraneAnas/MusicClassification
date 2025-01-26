import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import platform


def save_song_mel_spectrograms(
        base_folder="../data/genres_original",
        save_folder="../data/song_data",
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        target_duration=30
):
    """
    Generate and save log mel-spectrograms as .npy files for each full song,
    organizing them into separate folders by genre.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    genres = [genre for genre in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, genre))]

    for genre in tqdm(genres, desc="Processing Genres"):
        genre_folder = os.path.join(base_folder, genre)
        save_genre_folder = os.path.join(save_folder, genre)
        os.makedirs(save_genre_folder, exist_ok=True)

        files = [f for f in os.listdir(genre_folder) if f.endswith('.wav')]

        for file in tqdm(files, desc=f"Processing {genre} files", leave=False):
            file_path = os.path.join(genre_folder, file)
            try:
                # Load the entire audio file
                y, sr = librosa.load(file_path, sr=None)

                target_samples = target_duration * sr

                if len(y) > target_samples:
                    y = y[:target_samples]
                else:
                    y = np.pad(y, (0, target_samples - len(y)))

                # Compute the mel-spectrogram
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels
                )

                # Convert to decibels
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                # Min-max normalization
                min_val = np.min(mel_spectrogram_db)
                max_val = np.max(mel_spectrogram_db)

                if max_val - min_val > 0:
                    normalized_spectrogram = (mel_spectrogram_db - min_val) / (max_val - min_val)
                else:
                    normalized_spectrogram = mel_spectrogram_db

                # Save the spectrogram
                save_path = os.path.join(save_genre_folder, f"{file.split('.')[0]}{file.split('.')[1]}.npy")
                np.save(save_path, normalized_spectrogram)
            except Exception as e:
                print(f"Error processing file '{file_path}': {e}")

    print(f"Mel-spectrograms saved in: {save_folder}")


def load_genre_data(data_dir="../data/song_data",
                    genres=None):
    """
    Load spectrogram data and return the associated labels and song names.
    """

    if genres is None:
        # Automatically detect genres (folders) in the data directory
        genres = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Detected genres: {genres}")

    if not genres:
        raise ValueError("No genres found in the specified directory.")

    data = []
    labels = []
    song_names = []  # List to store song names
    for idx, genre in enumerate(genres):
        genre_folder = os.path.join(data_dir, genre)
        if os.path.exists(genre_folder):
            files = [f for f in os.listdir(genre_folder) if f.endswith('.npy')]
            if not files:
                print(f"Warning: No .npy files found for genre: {genre}")
                continue

            for file in files:
                file_path = os.path.join(genre_folder, file)
                try:
                    spectrogram = np.load(file_path)  # Load individual spectrogram
                    data.append(spectrogram)
                    labels.append(idx)  # Assign label based on genre index
                    song_names.append(os.path.splitext(file)[0])  # Add file name minus extension
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
        else:
            print(f"Warning: Folder not found for genre: {genre}")

    if not data or not labels:
        raise ValueError("No data found. Please check the input directory and genre list.")

    # Convert lists to numpy arrays
    data = np.array(data, dtype=object)  # Keep object type for varied dimensions if needed
    labels = np.array(labels)
    song_names = np.array(song_names)  # Convert to numpy array for consistency

    return data, labels, song_names


def load_dataset_splits(
    data_dir="../data/song_data",
    genres=None,
    test_split_size=0.1,
    validation_split_size=0.1,
    random_state=2003
):
    """
    Load spectrogram dataset and split into training, validation, and test sets,
    including the corresponding song names.
    """
    data, labels, song_names = load_genre_data(data_dir, genres)

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        data, labels, song_names,
        test_size=test_split_size, stratify=labels, random_state=random_state
    )

    # Further split the training data into training and validation sets
    X_train, X_val, Y_train, Y_val, S_train, S_val = train_test_split(
        X_train, Y_train, S_train,
        test_size=validation_split_size, stratify=Y_train, random_state=random_state
    )

    return X_train, Y_train, S_train, X_val, Y_val, S_val, X_test, Y_test, S_test


def slice_songs(X, Y, S,
                sr=22050,
                hop_length=512,
                length_in_seconds=30,
                overlap=0.5):
    """
    Slice spectrograms into smaller splits with overlap.

    Parameters:
        X: Array of spectrograms
        Y: Array of labels
        S: Array of song names
        sr: Sample rate (default: 22050)
        hop_length: Hop length used in spectrogram creation (default: 512)
        length_in_seconds: Length of each slice in seconds (default: 30)
        overlap: Overlap ratio between consecutive slices (default: 0.5 for 50% overlap)
    """
    # Compute the number of frames for the desired slice length
    frames_per_second = sr / hop_length
    slice_length_frames = int(length_in_seconds * frames_per_second)

    # Calculate hop size for overlapping (stride)
    stride = int(slice_length_frames * (1 - overlap))

    # Initialize lists for sliced data
    X_slices = []
    Y_slices = []
    S_slices = []

    # Slice each spectrogram
    for i, spectrogram in enumerate(X):
        num_frames = spectrogram.shape[1]

        # Calculate start positions for all slices
        start_positions = range(0, num_frames - slice_length_frames + 1, stride)

        for start_frame in start_positions:
            end_frame = start_frame + slice_length_frames

            # Extract the slice
            slice_ = spectrogram[:, start_frame:end_frame]

            # Only add if the slice is the expected length
            if slice_.shape[1] == slice_length_frames:
                X_slices.append(slice_)
                Y_slices.append(Y[i])
                S_slices.append(S[i])

    # Convert lists to numpy arrays
    X_slices = np.array(X_slices)
    Y_slices = np.array(Y_slices)
    S_slices = np.array(S_slices)

    return X_slices, Y_slices, S_slices


class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, S=None):
        # Convert the object array to float32 before creating tensor
        self.spectrograms = torch.tensor(np.stack(spectrograms).astype(np.float32), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.S = S

    def __len__(self):
        return len(self.spectrograms)  # Changed from self.X to self.spectrograms

    def __getitem__(self, idx):
        """
        Returns a single data point.
        """
        spectrogram = self.spectrograms[idx]  # Changed from self.X to self.spectrograms
        label = self.labels[idx]  # Changed from self.Y to self.labels

        if self.S is not None:
            song_name = self.S[idx]
            return spectrogram, label, song_name
        return spectrogram, label


def get_data_loaders(
    data_dir="../data/song_data",
    genres=None,
    test_split_size=0.1,
    validation_split_size=0.1,
    slice_length_seconds=30,
    overlap=0.5,
    sr=22050,
    hop_length=512,
    batch_size=32,
    random_state=2003,
    shuffle=True
):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    :param overlap:
    """
    # Load the dataset splits
    X_train, Y_train, S_train, X_val, Y_val, S_val, X_test, Y_test, S_test = load_dataset_splits(
        data_dir=data_dir,
        genres=genres,
        test_split_size=test_split_size,
        validation_split_size=validation_split_size,
        random_state=random_state
    )

    # Slice the spectrograms
    X_train_slices, Y_train_slices, _ = slice_songs(X_train, Y_train, S_train, sr,
                                                    hop_length, slice_length_seconds, overlap)
    X_val_slices, Y_val_slices, _ = slice_songs(X_val, Y_val, S_val, sr,
                                                hop_length, slice_length_seconds, overlap)
    X_test_slices, Y_test_slices, S_test_slices = slice_songs(X_test, Y_test, S_test,
                                                              sr, hop_length, slice_length_seconds, overlap)

    # Create PyTorch Datasets
    train_dataset = SpectrogramDataset(X_train_slices, Y_train_slices)
    val_dataset = SpectrogramDataset(X_val_slices, Y_val_slices)
    test_dataset = SpectrogramDataset(X_test_slices, Y_test_slices, S_test_slices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def save_run_info(models, epochs, learning_rate, batch_size, patience, slice_lengths, overlap_ratio, seeds, device, results_dir):
    """
    Save general run information to a text file in the results directory.
    Args:
        models: List of model names being trained (e.g., ["CNN", "CRNN"]).
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size used for training.
        slice_lengths: List of slice lengths in seconds.
        seeds: List of random seeds used.
        device: The device used (e.g., "cuda" or "cpu").
        results_dir: Directory where the results are saved.
    """
    run_info_path = os.path.join(results_dir, "run_info.txt")
    with open(run_info_path, mode="w") as file:
        file.write("Run Configuration\n")
        file.write("==================\n\n")
        file.write(f"Models: {', '.join(models)}\n")
        file.write(f"Number of Epochs: {epochs}\n")
        file.write(f"Early Stopping Patience: {patience}\n")
        file.write(f"Learning Rate: {learning_rate}\n")
        file.write(f"Batch Size: {batch_size}\n")
        file.write(f"Slice Lengths: {slice_lengths}\n")
        file.write(f"Slice Overlap Ratio: {overlap_ratio}\n")
        file.write(f"Seeds: {seeds}\n")
        file.write(f"Device: {device}\n")
        file.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            file.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
        file.write(f"CPU Info: {platform.processor()}\n")
        file.write(f"OS: {platform.system()} {platform.release()}\n")
        file.write(f"Python Version: {platform.python_version()}\n")
