# 🎵 Music Classification Using Deep Learning 🎵
This repository contains a deep learning pipeline for music genre classification using spectrogram-based models, including CNNs and CRNNs. The models are trained on mel spectrograms generated from audio files.

## Dependencies

The experiment code is writen in Python 3.10.15 and built on a number of Python packages including (but not limited to):

- torch~=2.3.1
- numpy~=1.26.4
- pandas~=2.2.2
- scikit-learn~=1.5.2
- scipy~=1.14.1
- librosa~=0.10.2.post1

Batch installation is possible using the supplied "requirements.txt" with pip or conda.

````cmd
pip install -r requirements.txt
````

Additional install details (recommended for replication and strong performance):
- Python: 3.10.15
- GPU: NVIDIA GeForce RTX 4050 (Driver: 566.03)
- CUDA : 12.7
- CUDNN: 9.1.0
- [ffmpeg](http://ffmpeg.org/download.html) is required by Librosa to convert audio files into spectrograms. 

## Dataset
This study primarily uses the GTZAN Music Genre Classification Dataset, a widely-used dataset for music genre classification research. The dataset is available for download from Kaggle:

**Access the dataset here** : [GTZAN Dataset - Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

The main characteristics of the dataset can be summarized as:

| Property          | Value  |
|-------------------|--------|
| # of Tracks       | 1000   |
| # of genres       | 10     |
| Tracks per Genre | 100    | 
| Duration per Track           | 30 sec |

The figure below visualizes three seconds of the mel-scaled audio spectrogram for a randomly sampled song from each genre, the primary data representation used in training.

![assets/output.png](img%2Foutput.png)

## Preparing the Dataset
Once downloaded, extract the dataset and locate the audio files in:
````cmd
path_to_downloaded_dataset/Data/genres_original
````
Then, generate **mel spectrograms** using:
````cmd
from src.utility import save_song_mel_spectrograms
save_song_mel_spectrograms(base_folder="path_to_downloaded_dataset/Data/genres_original")
````

This will create a structured dataset in **data/song_data/**, organized as:
````cmd
data/song_data/
   ├── blues/
   ├── classical/
   ├── country/
   ├── disco/
   ├── hiphop/
   ├── jazz/
   ├── metal/
   ├── pop/
   ├── reggae/
   ├── rock/
````


## Model Training
### Configuring Hyperparameters
All training hyperparameters are defined in _config.py_, making it easier to modify settings without changing the main script.

Latest run config :

````cmd
DEVICE = "cuda"
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
````

### Running Training
To train and evaluate models, simply run:

````cmd
python main.py
````
This script will:

- Read the configuration file.
- Train and evaluate CNN, CRNN-1D, CRNN-2D, and RNN models.
- Save logs, model weights, and performance metrics in the `results/` directory


## Results
Classification performance was evaluated using the test F1-score across three independent trials for varying audio clip durations {1s, 3s, 5s, 10s}. Both the average and best F1-scores were reported to provide insights into the model's overall performance and its peak capabilities. Additionally, Table 2 complements this analysis with metrics such as slice accuracy, song accuracy, and execution times for each model and duration. These results are discussed below:


**Test F1 Scores for Song-level Audio Features (3 runs):**

| <br/>Model | <br/>Type | 1s<br/> | 3s<br/> | 5s<br/> | 10s<br/> |
|:-----------| :--- | :--- | :--- | :--- | :--- |
| CNN        | Average | **0.854827** | 0.805952 | 0.806792 | 0.791403 |
| CNN        | Best | **0.892154** | 0.853420 | 0.843851 | 0.866250 |
| CRNN2D     | Average | **0.901206** | 0.860250 | 0.848177 | 0.812321 |
| CRNN2D     | Best | **0.933698** | 0.878168 | 0.879795 | 0.835102 |
| CRNN1D     | Average | **0.653674** | 0.590523 | 0.626817 | 0.522722 |
| CRNN1D     | Best | 0.682419 | 0.614971 | **0.697611** | 0.555610 |
| RNN        | Average | **0.757967** | 0.685237 | 0.438738 | 0.394573 |
| RNN        | Best | **0.778231** | 0.709431 | 0.605512 | 0.483631 |


**Model Performance Metrics by Audio Slice Length**:


| Model | Split Size | Slice Accuracy | Song Accuracy | Execution Time | Epoch |
| :--- | :--- |:---------------| :--- |:---------------| :--- |
| MusicCNN | 1s | 0.799167       | 0.9 | 21m 1s         | 106 |
| MusicCNN | 3s | 0.820527       | 0.86 | 24m 59s        | 82 |
| MusicCNN | 5s | 0.837273       | 0.85 | 24m 19s        | 84 |
| MusicCNN | 10s | 0.836000       | 0.88 | 38m 37s        | 133 |
| MusicCRNN2D | 1s | 0.833334       | 0.94 | 13m 12s        | 48 |
| MusicCRNN2D | 3s | 0.807895       | 0.89 | 11m 48s        | 43 |
| MusicCRNN2D | 5s | 0.819091       | 0.89 | 13m 50s        | 68 |
| MusicCRNN2D | 10s | 0.778000       | 0.85 | 18m 24s        | 66 |

 **Key Findings**:
1. **CRNN2D consistently outperforms CNN and other models for shorter audio clips**, with the highest scores for both average and best cases across most durations. This indicates that the 2D convolution layers are highly effective in capturing temporal and frequency-domain features from mel spectrograms.
2. **All models, particularly CRNN2D and CNN, perform better with shorter audio clips** (e.g., 1s). Longer clips may dilute temporal precision, leading to decreased classification accuracy. 
3. **MusicCRNN2D offers a good balance of accuracy and efficiency**, achieving competitive results with lower execution times compared to MusicCNN, especially for shorter audio clips. 
4. **Song-level accuracy remains higher than slice-level accuracy**, reflecting the benefit of aggregating predictions across multiple slices for robust classification.

## Live Demo on Hugging Face 🤗
Want to test the CNN and CRNN-2D models without setting up the project locally?
Try them out directly on Hugging Face Spaces:

Simply upload an audio file, and the model will predict its genre!

🔗[MusicGenrePulse - Live Demo](https://huggingface.co/spaces/Skynova/MusicGenrePulse)