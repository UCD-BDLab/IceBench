from sklearn.metrics import f1_score , recall_score, precision_score, accuracy_score
import pytorch_lightning as pl
import numpy as np
import torch

class MyModel(pl.LightningModule):
    def __init__(self, model, criterion, optimiser, scheduler, class_weights=None):
        super().__init__()
        self.model = model
        if isinstance(criterion, nn.CrossEntropyLoss) and class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = criterion 
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.test_predictions = []
        self.test_true_labels = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.view(-1)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss ,prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=1).cpu().numpy(), average='weighted')
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy ,prog_bar=True, logger=True)
        self.log('val_f1', f1 ,prog_bar=True, logger=True)
        
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)  # Reshape the target tensor to 1D
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        accuracy = accuracy_score(y.cpu().numpy(), y_hat.argmax(dim=1).cpu().numpy())
        f1 = f1_score(y.cpu().numpy(), y_hat.argmax(dim=1).cpu().numpy(), average='weighted' , zero_division=1)
        recall = recall_score(y.cpu().numpy(), y_hat.argmax(dim=1).cpu().numpy(), average='weighted', zero_division=1)
        precision = precision_score(y.cpu().numpy(), y_hat.argmax(dim=1).cpu().numpy(), average='weighted' ,    zero_division=1)
        
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        self.log('test_f1', f1)
        self.log('test_recall', recall)
        self.log('test_precision', precision)
        self.test_predictions.extend(y_hat.argmax(dim=1).cpu().numpy())
        self.test_true_labels.extend(y.cpu().numpy())
        
        return {'test_loss': loss, 'test_accuracy': accuracy, 'test_f1': f1, 'test_recall': recall, 'test_precision': precision}
    
    def configure_optimizers(self):
        return {
            'optimizer': self.optimiser,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
