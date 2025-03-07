import torch
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score , jaccard_score
from sklearn.metrics import mean_squared_error, r2_score
import gc
import numpy as np
import os

def train_and_validate_segmentation(hparams_data, hparams_model, model, model_name, criterion, optimiser, scheduler, scheduler_name, train_loader, val_loader, device, model_path, load=False):
    torch.backends.cudnn.benchmark = True
   
    train_losses = []
    season = hparams_data['train_data_options']['season'].strip("'")
    patch_size = hparams_data['data_preprocess_options']['patch_size']
    best_val_loss = float('inf')
    best_f1_score = 0
    early_stopping_counter = 0
    patience = int(hparams_model['train']['patience'])
    downscale_factor = int(hparams_data['data_preprocess_options']['downsampling_factor'])
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    criterion.to(device)

    # training part
    epochs = int(hparams_model['train']['epochs'])
    epoch_len = int(hparams_model['datamodule']['epoch_len'])
    start_time = time.perf_counter()
    
    for epoch in tqdm(iterable=range(epochs), position=0):
        gc.collect()  # Collect garbage to free memory.
        model.train()  
        running_loss = 0
        epoch_start_time = time.time() 
        
        # Loops though batches in queue.
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=train_loader, total=epoch_len, colour='white', position=0)):
            torch.cuda.empty_cache()

            inputs = batch_x.to(device, non_blocking=True)
            targets = batch_y.to(device, dtype=torch.long)
               
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.detach().item()
        
        torch.cuda.empty_cache()

        loss_epoch = torch.true_divide(running_loss, i + 1).detach().item()
        epoch_end_time = time.time()
        epoch_time_seconds = epoch_end_time - epoch_start_time
        epoch_time_minutes = epoch_time_seconds / 60
        print(f"Epoch {epoch+1} took {epoch_time_minutes:.2f} minutes to complete.")
        print('Mean training loss: ' + f'{loss_epoch:.3f}')
        train_losses.append(loss_epoch)
        
        del batch_x, batch_y, inputs, outputs 

        # Validation        
        print("Validating...")
        predictions_flat = []
        targets_flat = []
        val_losses = []
        model.eval()
        
        for inf_x, inf_y, masks, name in tqdm(iterable=val_loader, total=len(hparams_data['validation_list']), colour='green', position=0):            
            torch.cuda.empty_cache()
            with torch.no_grad(), torch.cuda.amp.autocast():
                inf_x = inf_x.to(device, non_blocking=True)
                inf_y = inf_y.to(device)
                outputs = model(inf_x)
                del inf_x 

                loss_val = criterion(outputs, inf_y.unsqueeze(0))
                val_loss = loss_val.item()
                val_losses.append(val_loss)
                
                predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
                predictions_nonmasked = predictions[~masks]
                targets_nonmasked = inf_y[~masks].cpu().numpy()
                predictions_flat = np.append(predictions_flat, predictions_nonmasked)
                targets_flat = np.append(targets_flat, targets_nonmasked)
            
            # Explicitly delete variables to free memory
            del inf_y, masks, outputs, loss_val, predictions, predictions_nonmasked, targets_nonmasked
            torch.cuda.empty_cache()
        
        # Calculate metrics
        r2 = r2_score(targets_flat, predictions_flat)
        jaccard = jaccard_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
        f1 = f1_score(targets_flat, predictions_flat, average='weighted')
        accuracy = accuracy_score(targets_flat, predictions_flat)
        precision = precision_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
        recall = recall_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
        
        mean_val_loss = sum(val_losses) / len(val_loader)
        
        print("epoch : ", epoch + 1)
        print(f"Validation F1: {f1 * 100:.4f}, Accuracy: {accuracy * 100:.4f}, Precision: {precision * 100:.4f}, Recall: {recall* 100:.4f}, Jaccard: {jaccard:.4f}")
        print(f"Validation Loss: {mean_val_loss:.4f}")
        
        # Step the scheduler appropriately based on type
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(mean_val_loss)  # Using validation loss instead of training loss
        else:
            scheduler.step()
        
        # Check for early stopping based on validation loss
        if mean_val_loss < best_val_loss: 
            best_val_loss = mean_val_loss
            early_stopping_counter = 0
            # Save the model when validation loss improves
            print("Saving model based on validation loss improvement...")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimiser.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),  # Also save scheduler state
                        'epoch': epoch},
                        model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping. No improvement in validation loss.")
                break
                
        if f1 > best_f1_score:
            best_f1_score = f1
            
            # Save the model when F1 score improves
            print(f"New best F1: {f1 * 100:.4f}, saving model...")
            model_path_f1 = model_path.replace('best_model', 'best_model_f1')
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimiser.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),  # Also save scheduler state
                        'epoch': epoch},
                        model_path_f1)
        
        del targets_flat, predictions_flat
    
    end_time = time.perf_counter()
    print(f'{(end_time-start_time)/60:.2f} minutes for model training...', end=' ')


import torch
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
from sklearn.metrics import mean_squared_error, r2_score
import gc
import numpy as np
import os

def train_segmentation(
    hparams_data,
    hparams_model,
    model,
    model_name,
    criterion,
    optimiser,
    scheduler,
    scheduler_name,
    train_loader,
    val_loader,
    device,
    model_path,
    load=False,
    save_best_f1=True,
    save_best_loss=True,
    track_epoch_time=True
):
    """
    Unified training function for segmentation models.
    
    Parameters:
    -----------
    hparams_data : dict
        Data hyperparameters
    hparams_model : dict
        Model hyperparameters
    model : torch.nn.Module
        Model to train
    model_name : str
        Name of the model (used for saving)
    criterion : torch.nn.Module
        Loss function
    optimiser : torch.optim.Optimizer
        Optimizer
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    scheduler_name : str
        Name of the scheduler type (e.g., 'ReduceLROnPlateau', 'CosineAnnealingLR')
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    device : torch.device
        Device to train on
    model_path : str
        Path to save model
    load : bool, default=False
        Whether to load a pre-trained model
    save_best_f1 : bool, default=True
        Whether to save model with best F1 score
    save_best_loss : bool, default=True
        Whether to save model with best validation loss
    track_epoch_time : bool, default=True
        Whether to track and print epoch time
    """
    torch.backends.cudnn.benchmark = True
   
    train_losses = []
    
    # Extract data parameters if available
    season = hparams_data.get('train_data_options', {}).get('season', '')
    if isinstance(season, str):
        season = season.strip("'")
    
    location = None
    if 'train_data_options' in hparams_data and 'location' in hparams_data['train_data_options']:
        if isinstance(hparams_data['train_data_options']['location'], list) and len(hparams_data['train_data_options']['location']) > 0:
            location = hparams_data['train_data_options']['location'][0].strip("'")
    
    patch_size = hparams_data.get('data_preprocess_options', {}).get('patch_size', 0)
    downscale_factor = int(hparams_data.get('data_preprocess_options', {}).get('downsampling_factor', 1))
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_f1_score = 0
    early_stopping_counter = 0
    patience = int(hparams_model['train']['patience'])
    
    # Move model to device
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    criterion.to(device)

    # Training settings
    epochs = int(hparams_model['train']['epochs'])
    
    # Determine epoch length based on what's available in the config
    if 'epoch_len' in hparams_model.get('datamodule', {}):
        epoch_len = int(hparams_model['datamodule']['epoch_len'])
    elif 'data_size' in hparams_model.get('datamodule', {}):
        epoch_len = int(hparams_model['datamodule']['data_size'])
    else:
        epoch_len = len(train_loader)
    
    # Create model_path_f1 for saving best F1 model
    model_path_f1 = model_path.replace('best_model', 'best_model_f1')
    
    start_time = time.perf_counter()
    
    for epoch in tqdm(iterable=range(epochs), position=0):
        gc.collect()  # Collect garbage to free memory
        model.train()  
        running_loss = 0
        
        # Track epoch time if requested
        if track_epoch_time:
            epoch_start_time = time.time()
        
        # Training loop
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=train_loader, total=epoch_len, colour='white', position=0)):
            torch.cuda.empty_cache()
            
            inputs = batch_x.to(device, non_blocking=True)
            targets = batch_y.to(device, dtype=torch.long)
               
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.detach().item()

        torch.cuda.empty_cache()
        
        # Calculate average loss for the epoch
        loss_epoch = torch.true_divide(running_loss, i + 1).detach().item()
        
        # Report epoch time if tracking
        if track_epoch_time:
            epoch_end_time = time.time()
            epoch_time_seconds = epoch_end_time - epoch_start_time
            epoch_time_minutes = epoch_time_seconds / 60
            print(f"Epoch {epoch+1} took {epoch_time_minutes:.2f} minutes to complete.")
        
        print(f"Epoch {epoch+1}/{epochs}, Mean training loss: {loss_epoch:.3f}")
        train_losses.append(loss_epoch)
        
        # Step the scheduler based on the scheduler type
        if scheduler is not None:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(loss_epoch)
            else:
                scheduler.step()
        
        # Clean up to save memory
        del batch_x, batch_y, inputs, outputs
        
        # Validation
        print("Validating...")
        predictions_flat = []
        targets_flat = []
        val_losses = []
        model.eval()
        
        # Determine validation data structure
        has_name_in_loader = True
        for val_batch in val_loader:
            if len(val_batch) == 3:  # (inf_x, inf_y, masks)
                has_name_in_loader = False
            break
        
        for val_batch in tqdm(iterable=val_loader, total=len(hparams_data.get('validation_list', [])), colour='green', position=0):
            torch.cuda.empty_cache()
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                if has_name_in_loader:
                    inf_x, inf_y, masks, name = val_batch
                else:
                    inf_x, inf_y, masks = val_batch
                
                inf_x = inf_x.to(device, non_blocking=True)
                inf_y = inf_y.to(device)
                outputs = model(inf_x)
                del inf_x
                
                # Handle different shapes for validation targets
                if inf_y.dim() == 2:  # 2D tensor needs unsqueeze
                    loss_val = criterion(outputs, inf_y.unsqueeze(0))
                else:  # Already has batch dimension
                    loss_val = criterion(outputs, inf_y)
                
                val_loss = loss_val.item()
                val_losses.append(val_loss)
                
                # Process predictions
                predictions = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
                predictions_nonmasked = predictions[~masks]
                targets_nonmasked = inf_y[~masks].cpu().numpy()
                
                predictions_flat = np.append(predictions_flat, predictions_nonmasked)
                targets_flat = np.append(targets_flat, targets_nonmasked)
            
            # Clean up to save memory
            del inf_y, masks, outputs, loss_val, predictions, predictions_nonmasked, targets_nonmasked
            torch.cuda.empty_cache()
        
        # Calculate metrics
        avg_val_loss = sum(val_losses) / len(val_loader)
        r2 = r2_score(targets_flat, predictions_flat)
        jaccard = jaccard_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
        f1 = f1_score(targets_flat, predictions_flat, average='weighted')
        accuracy = accuracy_score(targets_flat, predictions_flat)
        precision = precision_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
        recall = recall_score(targets_flat, predictions_flat, average='weighted', zero_division=1)
        
        print(f"Epoch {epoch+1}, Validation F1: {f1*100:.4f}, Accuracy: {accuracy*100:.4f}, "
              f"Precision: {precision*100:.4f}, Recall: {recall*100:.4f}, Jaccard: {jaccard:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save model based on validation loss if enabled
        if save_best_loss and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            
            print("Saving model with best validation loss...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'f1_score': f1
            }, model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping. No improvement in validation metrics.")
                break
        
        # Save model based on F1 score if enabled
        if save_best_f1 and f1 > best_f1_score:
            best_f1_score = f1
            
            print(f"New best F1: {f1*100:.4f}, saving model...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'f1_score': best_f1_score
            }, model_path_f1)
        
        # Clean up
        del targets_flat, predictions_flat
    
    # Report total training time
    end_time = time.perf_counter()
    training_minutes = (end_time - start_time) / 60
    print(f"{training_minutes:.2f} minutes for model training.")
    
    return {
        'best_val_loss': best_val_loss,
        'best_f1_score': best_f1_score,
        'train_losses': train_losses,
        'training_time_minutes': training_minutes
    }