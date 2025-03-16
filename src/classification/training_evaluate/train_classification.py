import torch
import torch.nn as nn
import os 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def train_and_evaluate_classification( hparams_model , model, model_name, criterion, optimizer,scheduler, train_loader, val_loader ,model_path, device ):

    # training part
    num_epochs = int(hparams_model['train']['epochs'])
    best_val_loss = float('inf')
    best_model_state_dict = None
    early_stopping_patience = int(hparams_model['train']['patience'])
    best_f1 = 0.0
    best_model_state_dict = None

    save_metric = hparams_model['train']['save_metric']
    # Validate save_metric parameter
    if save_metric not in ['f1', 'loss']:
        raise ValueError("save_metric must be either 'f1' or 'loss'")
    patience_counter = 0
    

    torch.backends.cudnn.benchmark = True

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    criterion.to(device)
    
    total_train_files = len(train_loader.dataset) // int(hparams_model['train']['batch_size'])
    total_val_files = len(val_loader.dataset)
    for epoch in tqdm(iterable=range(num_epochs), position=0):
        # Train the model
        train_loss = 0.0
        train_correct = 0
        improved = False
        for batch_idx, (data, target) in enumerate(tqdm(iterable=train_loader,colour='white', position=0)):
            optimizer.zero_grad()

            data = data.to(torch.float32).to(device)
            
            data[torch.isnan(data)] = 0
            target = target.long().view(-1).to(device)
            #target = target.to(torch.long).to(device) 
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # Get the predictions
            preds = torch.argmax(out, dim=1)

            # Count the number of correct predictions
            train_correct += torch.sum(preds == target).item()
        
        accuracy_train = train_correct / len(train_loader.dataset) * 100
        # Validation
        val_loss = 0.0
        val_correct = 0
        val_preds = []
        val_targets = []
        model.eval()
        print('Validation ...')
        with torch.no_grad():
            for idx , (val_data, val_target) in enumerate(tqdm(iterable=val_loader ,colour='green', position=0)):

                val_data = val_data.float().to(device)
                val_data[torch.isnan(val_data)] = 0 
                val_target = val_target.long().view(-1).to(device)
                out = model(val_data)
                loss = criterion(out, val_target)

                val_loss += loss.item()

                preds = torch.argmax(out, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(val_target.cpu().numpy())
                val_correct += torch.sum(preds == val_target).item()
        
        loss_val = val_loss / len(val_loader)
        scheduler.step(loss_val)
        
        accuracy = accuracy_score(val_targets, val_preds)
        precision = precision_score(val_targets, val_preds, average='weighted' ,zero_division=1)
        recall = recall_score(val_targets, val_preds, average='weighted' , zero_division=1)
        f1 = f1_score(val_targets, val_preds, average='weighted' , zero_division=1)
        val_accuracy = val_correct / len(val_loader) * 100
        

        print("epoch : ", epoch + 1)
        print(f"Training Loss: {train_loss / len(train_loader):.4f} , Training Accuracy: {accuracy_train:.4f}")
        print(f"Validation F1: {f1 * 100:.4f}, Accuracy: {accuracy * 100:.4f}, Precision: {precision * 100:.4f}, Recall: {recall* 100:.4f}")
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    
        # Save model based on the chosen metric
        
        if save_metric == 'loss' and loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model_state_dict = model.state_dict()
            print(f"Saving model based on validation loss: {loss_val:.4f}")
            torch.save({
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'f1_score': f1
            }, model_path)
            improved = True
            
        elif save_metric == 'f1' and f1 > best_f1:
            best_f1 = f1
            best_model_state_dict = model.state_dict()
            print(f"Saving model based on F1 score: {f1 * 100:.4f}")
            torch.save({
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': loss_val,
                'f1_score': best_f1
            }, model_path)
            improved = True

        # Update patience counter
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} as {save_metric} hasn't improved for {early_stopping_patience} epochs.")
            break

    return best_val_loss, best_f1

