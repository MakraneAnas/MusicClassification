import torch
from src.trainer import train_and_evaluate
from src.models import MusicCNN, MusicCRNN2D, MusicCRNN1D, MusicRNN
from src.utility import save_run_info
import config

if __name__ == "__main__":
    if torch.cuda.is_available() and config.DEVICE == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        DEVICE = torch.device("cuda")
    else:
        print("CUDA not available or not requested, using CPU")
        DEVICE = torch.device("cpu")

    # Configuration
    DATA_DIR = config.DATA_DIR
    GENRES = config.GENRES
    SLICE_LENGTHS = config.SLICE_LENGTHS
    SEEDS = config.SEEDS
    NUM_CLASSES = config.NUM_CLASSES
    EPOCHS = config.EPOCHS
    EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
    BATCH_SIZE = config.BATCH_SIZE
    OVERLAP = config.OVERLAP
    LEARNING_RATE = config.LEARNING_RATE
    RESULTS_DIR = config.RESULTS_DIR
    WEIGHTS_CHECKPOINT = config.WEIGHTS_CHECKPOINT

    save_run_info(
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
    )

    # Train CNN
    train_and_evaluate(model_class=MusicCNN, model_name="MusicCNN", data_dir=DATA_DIR, genres=GENRES,
                       slice_lengths=SLICE_LENGTHS, seeds=SEEDS, num_classes=NUM_CLASSES, device=DEVICE, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, early_stopping_patience=EARLY_STOP_PATIENCE,
                       results_dir=RESULTS_DIR, save_weights=WEIGHTS_CHECKPOINT)

    # Train CRNN2D
    train_and_evaluate(model_class=MusicCRNN2D, model_name="MusicCRNN2D", data_dir=DATA_DIR, genres=GENRES,
                       slice_lengths=SLICE_LENGTHS, seeds=SEEDS, num_classes=NUM_CLASSES, device=DEVICE, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, results_dir=RESULTS_DIR,
                       save_weights=WEIGHTS_CHECKPOINT)

    # Train CRNN1D
    train_and_evaluate(model_class=MusicCRNN1D, model_name="MusicCRNN1D", data_dir=DATA_DIR, genres=GENRES,
                       slice_lengths=SLICE_LENGTHS, seeds=SEEDS, num_classes=NUM_CLASSES, device=DEVICE, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, early_stopping_patience=EARLY_STOP_PATIENCE,
                       results_dir=RESULTS_DIR, save_weights=WEIGHTS_CHECKPOINT)

    # Train RNN
    train_and_evaluate(model_class=MusicRNN, model_name="MusicRNN", data_dir=DATA_DIR, genres=GENRES,
                       slice_lengths=SLICE_LENGTHS, seeds=SEEDS, num_classes=NUM_CLASSES, device=DEVICE, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, early_stopping_patience=EARLY_STOP_PATIENCE,
                       results_dir=RESULTS_DIR, save_weights=WEIGHTS_CHECKPOINT)
