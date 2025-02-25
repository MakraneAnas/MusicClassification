DEVICE = "cuda"  # Use "cuda" or "cpu"
DATA_DIR = "data/song_data"
GENRES = None
SLICE_LENGTHS = [1, 3, 5, 10]
SEEDS = [333, 123, 223]
NUM_CLASSES = 10
EPOCHS = 300
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 16
OVERLAP = 0.5
LEARNING_RATE = 0.001
RESULTS_DIR = "./results"
WEIGHTS_CHECKPOINT = False