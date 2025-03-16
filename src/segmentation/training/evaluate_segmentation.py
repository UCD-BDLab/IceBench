
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score, classification_report
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import plot_confusion_matrix, plot_predictions
from utils.utils import *
import json
from season_based_arctic_Seaice import return_season



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
    print(" ")
    print("testing")
    os.makedirs('inference', exist_ok=True)
    
    # Load the best model weights
    if load:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print("Epoch of the saved model:", epoch)

    model.to(device)
    model.eval()
   
    predictions_flat = []
    targets_flat = []

    for batch_x, batch_y, masks, name in tqdm(iterable=test_loader, total=len(hparams_data['test_list']), colour='yellow', position=0):
        torch.cuda.empty_cache()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            print("file test name is : ", name)

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
                    
            predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            predictions_nonmasked = predictions[~masks]
            targets_nonmasked = batch_y[~masks].cpu().numpy()

            del batch_x, batch_y, outputs

            predictions_flat = np.append(predictions_flat, predictions_nonmasked)
            targets_flat = np.append(targets_flat, targets_nonmasked)

    # Calculate metrics
    f1 = f1_score(targets_flat, predictions_flat, average='weighted')
    print("F1 score: ", f1)
    accuracy = accuracy_score(targets_flat, predictions_flat)
    print("Accuracy score: ", accuracy)
    recall = recall_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
    print("Recall score: ", recall)
    jaccard = jaccard_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
    print("Jaccard score: ", jaccard)
    precision = precision_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
    print("Precision score: ", precision)

    print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Jaccard: {jaccard:.4f}")

    return f1, accuracy, precision, recall



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
    print(" ")
    print("testing")
    os.makedirs('inference' , exist_ok=True)
    season_ = hparams_data ['train_data_options']['season'].strip("'")
    curr_loc = hparams_data ['train_data_options']['location'][0].strip("'")

    # Load the best model weights
    if load:
        #model_path = 'models_per_loc_per_seasons/best_model_{}_{}_{}_test'.format(curr_loc , season,model_name)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])


    model.to(device)
    model.eval()

   
    predictions_flat = []
    targets_flat = []

    season_scores = {'winter': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'spring': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'summer': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'fall': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0}}
    seasons_predictions = {'winter': [], 'spring': [], 'summer': [], 'fall': []}
    seasons_targets = {'winter': [], 'spring': [], 'summer': [], 'fall': []}

    location_scores = {'cat1': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'cat2': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'cat3': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'cat4': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0}}
    location_predictions = {'cat1': [], 'cat2': [], 'cat3': [], 'cat4': []}
    location_targets = {'cat1': [], 'cat2': [], 'cat3': [], 'cat4': []}

    for batch_x, batch_y, masks, name in tqdm(iterable=test_loader, total=len(hparams_data['test_list']), colour='yellow',  position=0):
        torch.cuda.empty_cache()
        #print("file test name is : ", name)
        season_file = get_season(name)
        location_file = get_location(name)

        with torch.no_grad() , torch.cuda.amp.autocast():

            batch_x = batch_x.to(device , non_blocking=True)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
                    
            predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            
            predictions_nonmasked = predictions[~masks]
            targets_nonmasked = batch_y[~masks].cpu().numpy()
            predictions_flat = np.append(predictions_flat, predictions_nonmasked)
            targets_flat = np.append(targets_flat,targets_nonmasked)

            seasons_predictions[season_file].extend(predictions_nonmasked)
            seasons_targets[season_file].extend(targets_nonmasked)

            location_predictions[location_file].extend(predictions_nonmasked)
            location_targets[location_file].extend(targets_nonmasked)


            # print("F1 score: ", f1_score( targets_nonmasked , predictions_nonmasked , average='weighted' ))
            # print("Accuracy score: ", accuracy_score( targets_nonmasked , predictions_nonmasked ))
            # print("Precision score: ", precision_score(targets_nonmasked ,predictions_nonmasked,average='weighted' , zero_division=1))
            # print("Recall score: ", recall_score(targets_nonmasked ,predictions_nonmasked, average='weighted' , zero_division=1))
            # print("Jaccard score: ", jaccard_score(targets_nonmasked ,predictions_nonmasked, average='weighted' , zero_division=1))
            # print("Segmentation report: \n", classification_report(targets_nonmasked ,predictions_nonmasked , zero_division=1))                                       


    # Calculate metrics
    f1 = f1_score( targets_flat , predictions_flat , average='weighted' )
    accuracy = accuracy_score( targets_flat , predictions_flat )
    precision = precision_score(targets_flat ,predictions_flat,average='weighted' , zero_division=1)
    recall = recall_score(targets_flat ,predictions_flat, average='weighted' , zero_division=1)
    jaccard = jaccard_score(targets_flat ,predictions_flat, average='weighted' , zero_division=1)
    report = classification_report(targets_flat ,predictions_flat , zero_division=1)
    print(f"The overall result for trained model on {season_} and {curr_loc}" )
    print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} , Jaccard: {jaccard:.4f}")
    print(f"Segmentation Report: \n {report}")
    
    # save the model name and the metrics
    # make  a dictionary OF METRIC THEN SAVE IT
    dic_metrics = {'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall , 'jaccard': jaccard}
    np.save('inference/metrics_{}.npy'.format(model_name), dic_metrics)

    # plot the prediction and target for each season
    #plot_4in1_loc_season(hparams_data , seasons_predictions , seasons_targets)
    if season_prediction:
        for season in seasons_predictions:
            f1 = f1_score( seasons_targets[season] , seasons_predictions[season] , average='weighted' )
            accuracy = accuracy_score( seasons_targets[season] , seasons_predictions[season] )
            precision = precision_score(seasons_targets[season] ,seasons_predictions[season],average='weighted' , zero_division=1)
            recall = recall_score(seasons_targets[season] ,seasons_predictions[season], average='weighted' , zero_division=1)
            jaccard = jaccard_score(seasons_targets[season] ,seasons_predictions[season], average='weighted' , zero_division=1)
            report = classification_report(seasons_targets[season] ,seasons_predictions[season] , zero_division=1)
            print(f"The result for {season} season in test files")
            
            #plot_season_prediction_target(hparams_data , seasons_predictions[season] , seasons_targets[season] , season)
            print(" ")
            print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} , Jaccard: {jaccard:.4f}")
            print(f"Segmentation Report: \n {report}")
            season_scores[season]['f1'] = f1
            season_scores[season]['accuracy'] = accuracy
            season_scores[season]['precision'] = precision
            season_scores[season]['recall'] = recall
            season_scores[season]['jaccard'] = jaccard
    
    if location_prediction:
        for location in location_predictions:
            if location_targets[location] and location_predictions[location]:
                f1 = f1_score( location_targets[location] , location_predictions[location] , average='weighted' )
                accuracy = accuracy_score( location_targets[location] , location_predictions[location] )
                precision = precision_score(location_targets[location] ,location_predictions[location],average='weighted' , zero_division=1)
                recall = recall_score(location_targets[location] ,location_predictions[location], average='weighted' , zero_division=1)
                jaccard = jaccard_score(location_targets[location] ,location_predictions[location], average='weighted' , zero_division=1)
                report = classification_report(location_targets[location] ,location_predictions[location] , zero_division=1)
                print(f"The result for {location} location in test files")
                
                #plot_season_prediction_target(hparams_data , seasons_predictions[season] , seasons_targets[season] , season)
                print(" ")
                print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} , Jaccard: {jaccard:.4f}")
                print(f"Segmentation Report: \n {report}")
                location_scores[location]['f1'] = f1
                location_scores[location]['accuracy'] = accuracy
                location_scores[location]['precision'] = precision
                location_scores[location]['recall'] = recall
                location_scores[location]['jaccard'] = jaccard
            else:
                print(f"The result for {location} location in test files")
                print(f"No test files for {location} lcoation")
                location_scores[location]['f1'] = 0
                location_scores[location]['accuracy'] = 0
                location_scores[location]['precision'] = 0
                location_scores[location]['recall'] = 0
                location_scores[location]['jaccard'] = 0
                print("test F1: 0, Accuracy: 0, Precision: 0, Recall: 0 , Jaccard: 0")

    return f1, accuracy, precision, recall

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
    print(" ")
    print("testing")
    os.makedirs('inference' , exist_ok=True)
    season_ = hparams_data ['train_data_options']['season'].strip("'")
    curr_loc = hparams_data ['train_data_options']['location'][0].strip("'")
    dir_test = hparams_data['dir_test_with_icecharts']

    # Load the best model weights
    if load:
        #model_path = 'models_per_loc_per_seasons/best_model_{}_{}_{}_test'.format(curr_loc , season,model_name)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])


    model.to(device)
    model.eval()

   
    predictions_flat = []
    targets_flat = []

    

    season_scores = {'Melt': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                    'Freeze': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0}}
    
    seasons_predictions = {'Melt': [], 'Freeze': []}
    seasons_targets = {'Melt': [], 'Freeze': []}

    location_scores = {'cat1': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'cat2': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'cat3': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0},
                        'cat4': {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0 , 'jaccard': 0}}
    location_predictions = {'cat1': [], 'cat2': [], 'cat3': [], 'cat4': []}
    location_targets = {'cat1': [], 'cat2': [], 'cat3': [], 'cat4': []}
    for batch_x, batch_y, masks, name in tqdm(iterable=test_loader, total=len(hparams_data['test_list']), colour='yellow',  position=0):
        torch.cuda.empty_cache()
        #print("file test name is : ", name)
        season_file = return_season(hparams_data['dir_test_with_icecharts'] , name)
        location_file = get_location(name)
        
        if season_file is None:
            print(f"Skipping file with name {name} as no season could be determined.")
            continue
        with torch.no_grad() , torch.cuda.amp.autocast():

            batch_x = batch_x.to(device , non_blocking=True)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
                    
            predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            
            predictions_nonmasked = predictions[~masks]
            targets_nonmasked = batch_y[~masks].cpu().numpy()
            predictions_flat = np.append(predictions_flat, predictions_nonmasked)
            targets_flat = np.append(targets_flat,targets_nonmasked)

            seasons_predictions[season_file].extend(predictions_nonmasked)
            seasons_targets[season_file].extend(targets_nonmasked)

            location_predictions[location_file].extend(predictions_nonmasked)
            location_targets[location_file].extend(targets_nonmasked)

    # Calculate metrics
    f1 = f1_score( targets_flat , predictions_flat , average='weighted' )
    accuracy = accuracy_score( targets_flat , predictions_flat )
    precision = precision_score(targets_flat ,predictions_flat,average='weighted' , zero_division=1)
    recall = recall_score(targets_flat ,predictions_flat, average='weighted' , zero_division=1)
    jaccard = jaccard_score(targets_flat ,predictions_flat, average='weighted' , zero_division=1)
    report = classification_report(targets_flat ,predictions_flat , zero_division=1)
    print(f"The overall result for trained model on {season_} and {curr_loc}" )
    print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} , Jaccard: {jaccard:.4f}")
    print(f"Segmentation Report: \n {report}")
    
    # save the model name and the metrics
    # make  a dictionary OF METRIC THEN SAVE IT
    dic_metrics = {'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall , 'jaccard': jaccard}
    np.save('inference/metrics_{}.npy'.format(model_name), dic_metrics)

    # plot the prediction and target for each season
    #plot_4in1_loc_season(hparams_data , seasons_predictions , seasons_targets)
    if season_prediction:
        for season in seasons_predictions:
            f1 = f1_score( seasons_targets[season] , seasons_predictions[season] , average='weighted' )
            accuracy = accuracy_score( seasons_targets[season] , seasons_predictions[season] )
            precision = precision_score(seasons_targets[season] ,seasons_predictions[season],average='weighted' , zero_division=1)
            recall = recall_score(seasons_targets[season] ,seasons_predictions[season], average='weighted' , zero_division=1)
            jaccard = jaccard_score(seasons_targets[season] ,seasons_predictions[season], average='weighted' , zero_division=1)
            report = classification_report(seasons_targets[season] ,seasons_predictions[season] , zero_division=1)
            print(f"The result for {season} season in test files")
            
            #plot_season_prediction_target(hparams_data , seasons_predictions[season] , seasons_targets[season] , season)
            print(" ")
            print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} , Jaccard: {jaccard:.4f}")
            print(f"Segmentation Report: \n {report}")
            season_scores[season]['f1'] = f1
            season_scores[season]['accuracy'] = accuracy
            season_scores[season]['precision'] = precision
            season_scores[season]['recall'] = recall
            season_scores[season]['jaccard'] = jaccard
    
    if location_prediction:
        for location in location_predictions:
            if location_targets[location] and location_predictions[location]:
                f1 = f1_score( location_targets[location] , location_predictions[location] , average='weighted' )
                accuracy = accuracy_score( location_targets[location] , location_predictions[location] )
                precision = precision_score(location_targets[location] ,location_predictions[location],average='weighted' , zero_division=1)
                recall = recall_score(location_targets[location] ,location_predictions[location], average='weighted' , zero_division=1)
                jaccard = jaccard_score(location_targets[location] ,location_predictions[location], average='weighted' , zero_division=1)
                report = classification_report(location_targets[location] ,location_predictions[location] , zero_division=1)
                print(f"The result for {location} location in test files")
                
                #plot_season_prediction_target(hparams_data , seasons_predictions[season] , seasons_targets[season] , season)
                print(" ")
                print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} , Jaccard: {jaccard:.4f}")
                print(f"Segmentation Report: \n {report}")
                location_scores[location]['f1'] = f1
                location_scores[location]['accuracy'] = accuracy
                location_scores[location]['precision'] = precision
                location_scores[location]['recall'] = recall
                location_scores[location]['jaccard'] = jaccard
            else:
                print(f"The result for {location} location in test files")
                print(f"No test files for {location} lcoation")
                location_scores[location]['f1'] = 0
                location_scores[location]['accuracy'] = 0
                location_scores[location]['precision'] = 0
                location_scores[location]['recall'] = 0
                location_scores[location]['jaccard'] = 0
                print("test F1: 0, Accuracy: 0, Precision: 0, Recall: 0 , Jaccard: 0")

    return f1, accuracy, precision, recall