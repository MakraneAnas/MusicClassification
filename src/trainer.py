import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from src.utility import get_data_loaders
import numpy as np
from collections import defaultdict
from scipy.stats import mode
import csv
import time
import pandas as pd


def train_one_epoch(model, loader, criterion, optimizer, device="cuda"):
    model = model.to(device)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device="cuda", song_level=False):
    """
    Evaluate the model on a dataset. Supports slice-level and optional song-level evaluations.

    Args:
        model: The trained model to evaluate.
        loader: DataLoader for evaluation data.
        criterion: Loss function.
        device: The device (e.g., 'cuda' or 'cpu').
        song_level: If True, computes song-level predictions using voting (requires song_names in loader).

    Returns:
        A dictionary containing:
        - Slice-level loss and accuracy.
        - Song-level metrics if song_level=True and loader provides song_names.
    """
    model = model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    slice_predictions = []
    slice_labels = []
    song_predictions = defaultdict(list)
    song_actuals = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation/Test"):
            # Unpack batch depending on the loader type
            if song_level and len(batch) == 3:
                inputs, labels, song_names = batch
            else:
                inputs, labels = batch
                song_names = None

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            slice_predictions.extend(preds.cpu().numpy())
            slice_labels.extend(labels.cpu().numpy())

            # Accumulate predictions for song-level evaluation if applicable
            if song_names is not None:
                for song_name, pred, label in zip(song_names, preds.cpu().numpy(), labels.cpu().numpy()):
                    song_predictions[song_name].append(pred)
                    if song_name not in song_actuals:
                        song_actuals[song_name] = label

    # Compute slice-level metrics
    slice_accuracy = correct / total
    slice_loss = running_loss / total

    results = {
        "slice_loss": slice_loss,
        "slice_accuracy": slice_accuracy,
        "slice_predictions": slice_predictions,
        "slice_labels": slice_labels,
    }

    if song_level and song_predictions:
        song_preds = []
        song_labels = []

        for song_name, preds in song_predictions.items():
            if preds:  # Ensure preds is not empty
                # Handle majority voting using `mode`
                majority_vote = mode(preds, keepdims=True).mode[0]
                song_preds.append(majority_vote)
                song_labels.append(song_actuals[song_name])
            else:
                print(f"Warning: No predictions for song {song_name}")
                continue

        song_accuracy = np.mean(np.array(song_preds) == np.array(song_labels))
        classification_metrics = classification_report(song_labels, song_preds, output_dict=True)
        confusion_mat = confusion_matrix(song_labels, song_preds)

        results.update({
            "song_accuracy": song_accuracy,
            "song_classification_report": classification_metrics,
            "song_confusion_matrix": confusion_mat,
            "song_predictions": song_preds,
            "song_labels": song_labels,
        })

    return results


def train_and_evaluate(
        model_class, model_name, data_dir, genres, slice_lengths, seeds,
        num_classes, device="cuda", overlap_ratio=0.5, epochs=10, batch_size=32, learning_rate=0.001, early_stopping_patience=8,
        results_dir="./results", save_weights=False
):
    for slice_length in slice_lengths:
        for seed in seeds:
            print(f"\n\n\nTraining {model_name} on slice length {slice_length}s with seed {seed}")

            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Define paths for results
            seed_results_dir = os.path.join(results_dir, model_name, f"seed_{seed}", f"{slice_length}s")
            os.makedirs(seed_results_dir, exist_ok=True)

            # Data loaders
            train_loader, val_loader, test_loader = get_data_loaders(data_dir=data_dir, genres=genres,
                                                                     slice_length_seconds=slice_length,
                                                                     overlap=overlap_ratio, batch_size=batch_size,
                                                                     random_state=seed)

            # Initialize model, criterion, and optimizer
            model = model_class(num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_val_loss = float('inf')
            patience_counter = 0

            # Training history
            history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
            start_time = time.time()

            # Train for the specified number of epochs
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")

                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_results = evaluate(model, val_loader, criterion, device, song_level=False)

                val_loss = val_results["slice_loss"]
                val_acc = val_results["slice_accuracy"]

                history["epoch"].append(epoch + 1)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)

                print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"\nVal Loss: {val_results['slice_loss']:.4f}, Val Acc: {val_results['slice_accuracy']:.4f}")

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Optionally save the best model weights
                    best_weights_path = os.path.join(seed_results_dir, "best_model_weights.pth")
                    torch.save(model.state_dict(), best_weights_path)
                else:
                    patience_counter += 1
                    print(f"Early stopping patience counter: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

            # Test the model
            test_results = evaluate(model, test_loader, criterion, device, song_level=True)

            # Extract metrics
            slice_loss = test_results["slice_loss"]
            slice_accuracy = test_results["slice_accuracy"]
            song_accuracy = test_results.get("song_accuracy", None)
            classification_metrics = test_results.get("song_classification_report", None)
            confusion_mat = test_results.get("song_confusion_matrix", None)

            end_time = time.time()
            total_time = end_time - start_time

            print(f"\nTest Slice-Level Accuracy: {slice_accuracy:.4f}")
            if song_accuracy is not None:
                print(f"\nTest Song-Level Accuracy: {song_accuracy:.4f}")

            # Save training history
            history_df = pd.DataFrame(history)
            history_path = os.path.join(seed_results_dir, "training_history.csv")
            history_df.to_csv(history_path, index=False)

            # Save classification report (mapped to genre names)
            class_report_path = os.path.join(seed_results_dir, "classification_report.csv")
            class_report_df = pd.DataFrame(classification_metrics).transpose()

            # Map genre names explicitly to indices
            # THIS IS NO GOOD ANAS

            genre_names = [genre for genre in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, genre))]
            if genre_names is not None:
                genre_indices = [str(i) for i in range(len(genre_names))]
                mapping = {str(idx): name for idx, name in enumerate(genre_names)}
                class_report_df.rename(index=mapping, inplace=True)

            class_report_df.to_csv(class_report_path)

            # Save confusion matrix
            confusion_matrix_path = os.path.join(seed_results_dir, "confusion_matrix.csv")
            confusion_df = pd.DataFrame(confusion_mat, index=genre_names, columns=genre_names)
            confusion_df.to_csv(confusion_matrix_path)

            # Save overall metrics in CSV
            overall_metrics_path = os.path.join(seed_results_dir, "overall_metrics.csv")
            with open(overall_metrics_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Slice Loss", slice_loss])
                writer.writerow(["Slice Accuracy", slice_accuracy])
                if song_accuracy is not None:
                    writer.writerow(["Song Accuracy", song_accuracy])
                writer.writerow(["Training Time (seconds)", total_time])

            # Save model weights (conditionally)
            if save_weights:
                weights_path = os.path.join(seed_results_dir, "model_weights.pth")
                torch.save(model.state_dict(), weights_path)
