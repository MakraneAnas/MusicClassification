import torch
from src.trainer import train_and_evaluate
from src.models import MusicCNN, MusicCRNN2D, MusicCRNN1D, MusicRNN
from src.utility import save_run_info

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "data/song_data"
    GENRES = None  # Automatically detect genres in the data directory
    SLICE_LENGTHS = [1, 3, 5, 10]  # Slice lengths in seconds
    HMM = [10]
    SEEDS = [333, 123, 223]  # Seeds for multiple runs and reproducibility
    NUM_CLASSES = 10
    EPOCHS = 300
    EARLY_STOP_PATIENCE = 10
    BATCH_SIZE = 16
    OVERLAP = 0.5
    LEARNING_RATE = 0.001
    RESULTS_DIR = "./results_2.0"
    WEIGHTS_CHECKPOINT = False

    """    save_run_info(
        models=["CNN", "CRNN2D", "CRNN1D", "RNN"],
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        patience=EARLY_STOP_PATIENCE,
        slice_lengths=SLICE_LENGTHS,
        overlap_ratio=OVERLAP,
        seeds=SEEDS,
        device=DEVICE,
        results_dir=RESULTS_DIR,
    )"""

    # Train CNN
    """    train_and_evaluate(model_class=MusicCNN, model_name="MusicCNN", data_dir=DATA_DIR, genres=GENRES,
                       slice_lengths=SLICE_LENGTHS, seeds=SEEDS, num_classes=NUM_CLASSES, device=DEVICE, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, early_stopping_patience=EARLY_STOP_PATIENCE,
                       results_dir=RESULTS_DIR, save_weights=WEIGHTS_CHECKPOINT)

    # Train CRNN2D
    train_and_evaluate(model_class=MusicCRNN2D, model_name="MusicCRNN2D", data_dir=DATA_DIR, genres=GENRES,
                       slice_lengths=SLICE_LENGTHS, seeds=SEEDS, num_classes=NUM_CLASSES, device=DEVICE, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, results_dir=RESULTS_DIR,
                       save_weights=WEIGHTS_CHECKPOINT)"""

    # Train CRNN1D
    train_and_evaluate(model_class=MusicCRNN1D, model_name="MusicCRNN1D", data_dir=DATA_DIR, genres=GENRES,
                       slice_lengths=HMM, seeds=SEEDS, num_classes=NUM_CLASSES, device=DEVICE, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, early_stopping_patience=EARLY_STOP_PATIENCE,
                       results_dir=RESULTS_DIR, save_weights=WEIGHTS_CHECKPOINT)

    # Train RNN
    train_and_evaluate(model_class=MusicRNN, model_name="MusicRNN", data_dir=DATA_DIR, genres=GENRES,
                       slice_lengths=SLICE_LENGTHS, seeds=SEEDS, num_classes=NUM_CLASSES, device=DEVICE, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, early_stopping_patience=EARLY_STOP_PATIENCE,
                       results_dir=RESULTS_DIR, save_weights=WEIGHTS_CHECKPOINT)
