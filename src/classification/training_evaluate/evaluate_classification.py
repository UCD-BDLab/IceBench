
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , classification_report , confusion_matrix , jaccard_score
from functions import plot_confusion_matrix
import numpy as np
import torch.nn as nn


def test_classification( model, model_name, criterion, test_loader, model_path,device, load = True):
    print(" ")
    print("testing")
    os.makedirs('inference' , exist_ok=True)
    
    # Load the best model weights
    if load:
        #model_path = 'best_model_{}'.format(model_name)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print("Epoch of the saved model:", epoch)
    
  
    model = model.to(device)
    model.eval()  # Set the model in evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for idx, (test_data, test_target) in enumerate(tqdm(iterable=test_loader,  total= len(test_loader) ,colour='green', position=0)):

            test_data = test_data.float().to(device)
            test_data[torch.isnan(test_data)] = 0
            test_target = test_target.long().view(-1).to(device)

            output = model(test_data)
            loss = criterion(output, test_target)

            test_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(test_target.cpu().numpy())
            test_correct += torch.sum(preds == test_target).item()
    
    accuracy = accuracy_score(test_targets, test_preds )
    precision = precision_score(test_targets, test_preds, average='weighted', zero_division=1)
    recall = recall_score(test_targets, test_preds, average='weighted' , zero_division=1)
    f1 = f1_score(test_targets, test_preds, average='weighted'  , zero_division=1)
    classification_report_test = classification_report(test_targets, test_preds , zero_division=1)
    confusion_matrix_test = confusion_matrix(test_targets, test_preds)
    # Print the test results

    print(f"Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(" classification report :" , classification_report_test)
    print(" confusion matrix :" , confusion_matrix_test)

    # save the model name and the metrics
    dic_metrics = {'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall }
    #np.save('inference/metrics_{}.npy'.format(model_name), dic_metrics)

    return accuracy, precision, recall, f1



