
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score, classification_report
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import plot_confusion_matrix, plot_predictions
from utils import *
import json
from season_based_arctic_Seaice import return_season

def load_model_weights(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print("Epoch of the saved model:", epoch)
    return model

def calculate_metrics(targets_flat, predictions_flat):
    f1 = f1_score(targets_flat, predictions_flat, average='weighted')
    accuracy = accuracy_score(targets_flat, predictions_flat)
    precision = precision_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
    recall = recall_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
    jaccard = jaccard_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
    report = classification_report(targets_flat, predictions_flat, zero_division=1)
    return f1, accuracy, precision, recall, jaccard, report

def save_metrics(metrics, model_name):
    np.save(f'inference/metrics_{model_name}.npy', metrics)

def test_segmentation(model, test_loader, criterion, model_name, device, hparams_data, model_path, load=True):
      """
    Perform segmentation testing on the given model.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function.
        model_name (str): Name of the model.
        device (torch.device): Device to run the model on.
        hparams_data (dict): Hyperparameters and data configuration.
        model_path (str): Path to the saved model.
        load (bool): Whether to load a saved model or not.

    Returns:
        tuple: F1 score, accuracy, precision, and recall of the model on the test set.
    """
    print("\n Testing")
    os.makedirs('inference', exist_ok=True)
    
    if load:
        model = load_model_weights(model, model_path)

    model.to(device)
    model.eval()

    predictions_flat, targets_flat = [], []

    for batch_x, batch_y, masks, name in tqdm(test_loader, total=len(hparams_data['test_list']), colour='yellow', position=0):
        torch.cuda.empty_cache()
        with torch.no_grad(), torch.cuda.amp.autocast():
            print("Testing file:", name)

            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device)
            outputs = model(batch_x)
                    
            predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            predictions_nonmasked = predictions[~masks]
            targets_nonmasked = batch_y[~masks].cpu().numpy()

            predictions_flat.extend(predictions_nonmasked)
            targets_flat.extend(targets_nonmasked)

    f1, accuracy, precision, recall, jaccard, report = calculate_metrics(targets_flat, predictions_flat)
    metrics = {'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'jaccard': jaccard}
    save_metrics(metrics, model_name)

    print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Jaccard: {jaccard:.4f}")
    print(f"Segmentation Report: \n{report}")

    return f1, accuracy, precision, recall



def test_segmentation_data_params(model, test_loader, model_name, device, hparams_data, model_path, load=True):
    """
    Perform segmentation testing on the given model with detailed data parameters.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        model_name (str): Name of the model.
        device (torch.device): Device to run the model on.
        hparams_data (dict): Hyperparameters and data configuration.
        model_path (str): Path to the saved model.
        load (bool): Whether to load a saved model or not.

    Returns:
        tuple: F1 score, accuracy, precision, and recall of the model on the test set.
    """
    print("\nTesting with detailed data parameters")
    os.makedirs('inference', exist_ok=True)
    
    if load:
        model = load_model_weights(model, model_path)

    model.to(device)
    model.eval()

    predictions_flat = []
    targets_flat = []

    for batch_x, batch_y, masks, name in tqdm(iterable=test_loader, total=len(hparams_data['test_list']), colour='yellow', position=0):
        torch.cuda.empty_cache()
        print(f"File test name is: {name}")

        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            
            predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            
            predictions_nonmasked = predictions[~masks]
            targets_nonmasked = batch_y[~masks].cpu().numpy()
            predictions_flat.extend(predictions_nonmasked)
            targets_flat.extend(targets_nonmasked)

            print("Unique classes and pixel counts in prediction:", np.unique(predictions_nonmasked, return_counts=True))
            print("Unique classes and pixel counts in target:", np.unique(targets_nonmasked, return_counts=True))

            f1, accuracy, precision, recall, jaccard, report = calculate_metrics(targets_nonmasked, predictions_nonmasked)
            print(f"File metrics - F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Jaccard: {jaccard:.4f}")
            print(f"File Segmentation Report:\n{report}")

    overall_f1, overall_accuracy, overall_precision, overall_recall, overall_jaccard, overall_report = calculate_metrics(targets_flat, predictions_flat)
    
    print("\nOverall results:")
    print(f"F1 score: {overall_f1:.4f}, Accuracy: {overall_accuracy:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, Jaccard: {overall_jaccard:.4f}")
    print(f"Overall Segmentation Report:\n{overall_report}")
    
    metrics = {'f1': overall_f1, 'accuracy': overall_accuracy, 'precision': overall_precision, 'recall': overall_recall, 'jaccard': overall_jaccard}
    save_metrics(metrics, model_name)

    return overall_f1, overall_accuracy, overall_precision, overall_recall

def test_segmentation_per_season_location(model, test_loader, model_name, device, hparams_data, model_path, load=True, season_prediction=True, location_prediction=True):
    """
    Perform segmentation testing on the given model, with results broken down by regular season and location.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        model_name (str): Name of the model.
        device (torch.device): Device to run the model on.
        hparams_data (dict): Hyperparameters and data configuration.
        model_path (str): Path to the saved model.
        load (bool): Whether to load a saved model or not.
        season_prediction (bool): Whether to calculate metrics per season.
        location_prediction (bool): Whether to calculate metrics per location.

    Returns:
        tuple: Overall F1 score, accuracy, precision, and recall of the model on the test set.
    """
    print("\nTesting per season and location")
    os.makedirs('inference', exist_ok=True)
    
    if load:
        model = load_model_weights(model, model_path)

    model.to(device)
    model.eval()

    predictions_flat = []
    targets_flat = []
    seasons_predictions = {'winter': [], 'spring': [], 'summer': [], 'fall': []}
    seasons_targets = {'winter': [], 'spring': [], 'summer': [], 'fall': []}
    location_predictions = {'cat1': [], 'cat2': [], 'cat3': [], 'cat4': []}
    location_targets = {'cat1': [], 'cat2': [], 'cat3': [], 'cat4': []}

    for batch_x, batch_y, masks, name in tqdm(iterable=test_loader, total=len(hparams_data['test_list']), colour='yellow', position=0):
        torch.cuda.empty_cache()
        season_file = get_season(name)
        location_file = get_location(name)

        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            
            predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            
            predictions_nonmasked = predictions[~masks]
            targets_nonmasked = batch_y[~masks].cpu().numpy()
            predictions_flat.extend(predictions_nonmasked)
            targets_flat.extend(targets_nonmasked)

            seasons_predictions[season_file].extend(predictions_nonmasked)
            seasons_targets[season_file].extend(targets_nonmasked)
            location_predictions[location_file].extend(predictions_nonmasked)
            location_targets[location_file].extend(targets_nonmasked)

    overall_f1, overall_accuracy, overall_precision, overall_recall, overall_jaccard, overall_report = calculate_metrics(targets_flat, predictions_flat)
    
    print("\nOverall results:")
    print(f"F1 score: {overall_f1:.4f}, Accuracy: {overall_accuracy:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, Jaccard: {overall_jaccard:.4f}")
    print(f"Overall Segmentation Report:\n{overall_report}")
    
    metrics = {'f1': overall_f1, 'accuracy': overall_accuracy, 'precision': overall_precision, 'recall': overall_recall, 'jaccard': overall_jaccard}
    save_metrics(metrics, model_name)

    if season_prediction:
        for season in seasons_predictions:
            if seasons_targets[season] and seasons_predictions[season]:
                f1, accuracy, precision, recall, jaccard, report = calculate_metrics(seasons_targets[season], seasons_predictions[season])
                print(f"\nResults for {season} season:")
                print(f"F1 score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Jaccard: {jaccard:.4f}")
                print(f"Segmentation Report:\n{report}")
            else:
                print(f"\nNo data for {season} season")

    if location_prediction:
        for location in location_predictions:
            if location_targets[location] and location_predictions[location]:
                f1, accuracy, precision, recall, jaccard, report = calculate_metrics(location_targets[location], location_predictions[location])
                print(f"\nResults for {location} location:")
                print(f"F1 score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Jaccard: {jaccard:.4f}")
                print(f"Segmentation Report:\n{report}")
            else:
                print(f"\nNo data for {location} location")

    return overall_f1, overall_accuracy, overall_precision, overall_recall

def test_segmentation_for_two_seasons(model, test_loader, model_name, device, hparams_data, model_path, load=True, season_prediction=True, location_prediction=True):
    """
    Perform segmentation testing on the given model for two specific seasons: Melt and Freeze.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        model_name (str): Name of the model.
        device (torch.device): Device to run the model on.
        hparams_data (dict): Hyperparameters and data configuration.
        model_path (str): Path to the saved model.
        load (bool): Whether to load a saved model or not.
        season_prediction (bool): Whether to calculate metrics per season.
        location_prediction (bool): Whether to calculate metrics per location.

    Returns:
        tuple: Overall F1 score, accuracy, precision, and recall of the model on the test set.
    """
    print("\nTesting for Melt and Freeze seasons")
    os.makedirs('inference', exist_ok=True)
    
    if load:
        model = load_model_weights(model, model_path)

    model.to(device)
    model.eval()

    predictions_flat = []
    targets_flat = []
    seasons_predictions = {'Melt': [], 'Freeze': []}
    seasons_targets = {'Melt': [], 'Freeze': []}
    location_predictions = {'cat1': [], 'cat2': [], 'cat3': [], 'cat4': []}
    location_targets = {'cat1': [], 'cat2': [], 'cat3': [], 'cat4': []}

    for batch_x, batch_y, masks, name in tqdm(iterable=test_loader, total=len(hparams_data['test_list']), colour='yellow', position=0):
        torch.cuda.empty_cache()
        season_file = return_season(hparams_data['dir_test_with_icecharts'], name)
        location_file = get_location(name)
        
        if season_file is None:
            print(f"Skipping file with name {name} as no season could be determined.")
            continue

        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            
            predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            
            predictions_nonmasked = predictions[~masks]
            targets_nonmasked = batch_y[~masks].cpu().numpy()
            predictions_flat.extend(predictions_nonmasked)
            targets_flat.extend(targets_nonmasked)

            seasons_predictions[season_file].extend(predictions_nonmasked)
            seasons_targets[season_file].extend(targets_nonmasked)
            location_predictions[location_file].extend(predictions_nonmasked)
            location_targets[location_file].extend(targets_nonmasked)

    overall_f1, overall_accuracy, overall_precision, overall_recall, overall_jaccard, overall_report = calculate_metrics(targets_flat, predictions_flat)
    
    print("\nOverall results:")
    print(f"F1 score: {overall_f1:.4f}, Accuracy: {overall_accuracy:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, Jaccard: {overall_jaccard:.4f}")
    print(f"Overall Segmentation Report:\n{overall_report}")
    
    metrics = {'f1': overall_f1, 'accuracy': overall_accuracy, 'precision': overall_precision, 'recall': overall_recall, 'jaccard': overall_jaccard}
    save_metrics(metrics, model_name)

    if season_prediction:
        for season in seasons_predictions:
            if seasons_targets[season] and seasons_predictions[season]:
                f1, accuracy, precision, recall, jaccard, report = calculate_metrics(seasons_targets[season], seasons_predictions[season])
                print(f"\nResults for {season} season:")
                print(f"F1 score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Jaccard: {jaccard:.4f}")
                print(f"Segmentation Report:\n{report}")
            else:
                print(f"\nNo data for {season} season")

    if location_prediction:
        for location in location_predictions:
            if location_targets[location] and location_predictions[location]:
                f1, accuracy, precision, recall, jaccard, report = calculate_metrics(location_targets[location], location_predictions[location])
                print(f"\nResults for {location} location:")
                print(f"F1 score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Jaccard: {jaccard:.4f}")
                print(f"Segmentation Report:\n{report}")
            else:
                print(f"\nNo data for {location} location")

    return overall_f1, overall_accuracy, overall_precision, overall_recall
