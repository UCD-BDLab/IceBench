import torch
import torch.nn as nn
import os 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def train_and_validate_classification( hparams_model , model, model_name, criterion, optimizer,scheduler, train_loader, val_loader ,model_path, device ):
    


    # training part
    num_epochs = int(hparams_model['train']['epochs'])
    best_val_loss = float('inf')
    best_model_state_dict = None
    early_stopping_patience = int(hparams_model['train']['patience'])
    patience_counter = 0
    best_f1 = 0.0
    best_model_state_dict = None

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
    
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model_state_dict = model.state_dict()
            print("Saving model based on Val loss...")
            torch.save({
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, model_path)
            
        if f1 > best_f1:
            best_f1 = f1
            print(f"New best F1: {f1 * 100:.4f}, saving model... ")
            
            torch.save({
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, model_path.replace('best_val_loss' ,'best_f1'))
        
        # Early stopping
        if loss_val >= best_val_loss:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} as validation loss didn't improve for {early_stopping_patience} epochs.")
            break
    return best_val_loss , best_f1

