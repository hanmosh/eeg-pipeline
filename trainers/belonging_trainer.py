import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np
from utils.log import logger, model_tracker


class BelongingTrainer:
    def __init__(self, trainer_params, model, data, metadata):
        self.trainer_params = trainer_params
        self.model = model
        self.data = data
        self.metadata = metadata
        
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        
    def run(self):
        trained_model = self.train()
        
        if self.data['test_loader'] is not None:
            self.evaluate(self.data['test_loader'], split_name='test')
        
        # Plot metrics
        if model_tracker.save_model:
            model_tracker.plot_metrics(["train_loss", "val_loss"])
            model_tracker.plot_metrics(["train_accuracy", "val_accuracy"])
            model_tracker.plot_metric("train_auc")
            if self.data['val_loader'] is not None:
                model_tracker.plot_metric("val_auc")
        
        return trained_model
    
    def train(self):
        logger.log_dict(self.trainer_params)
        
        num_epochs = self.trainer_params.get('num_epochs', 50)
        learning_rate = self.trainer_params.get('learning_rate', 1e-3)
        weight_decay = self.trainer_params.get('weight_decay', 1e-4)
        patience = self.trainer_params.get('patience', 10)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        
        train_loader = self.data['train_loader']
        val_loader = self.data['val_loader']
        
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []
            train_preds = []
            train_labels = []
            train_probs = []
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                train_probs.extend(probs[:, 1].detach().cpu().numpy())  # Prob of positibe class
            
            avg_train_loss = np.mean(train_losses)
            train_accuracy = accuracy_score(train_labels, train_preds)
            
            # Calculate AUC
            if len(np.unique(train_labels)) > 1:
                train_auc = roc_auc_score(train_labels, train_probs)
            else:
                train_auc = 0.0
            
            model_tracker.track_metric('train_loss', avg_train_loss)
            model_tracker.track_metric('train_accuracy', train_accuracy)
            model_tracker.track_metric('train_auc', train_auc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_accuracy, val_auc = self.validate(val_loader, criterion)
                
                model_tracker.track_metric('val_loss', val_loss)
                model_tracker.track_metric('val_accuracy', val_accuracy)
                model_tracker.track_metric('val_auc', val_auc)
                
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train AUC: {train_auc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
                
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Restored best model from validation")
        
        logger.log('final_train_loss', avg_train_loss)
        logger.log('final_train_accuracy', train_accuracy)
        logger.log('final_train_auc', train_auc)
        
        if val_loader is not None:
            logger.log('best_val_loss', best_val_loss)
        
        return self.model
    
    def validate(self, val_loader, criterion):
        """Validation phase"""
        self.model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        val_accuracy = accuracy_score(val_labels, val_preds)
        
        # Calculate AUC
        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_probs)
        else:
            val_auc = 0.0
        
        return avg_val_loss, val_accuracy, val_auc
    
    def evaluate(self, test_loader, split_name='test'):
        """Final evaluation on test set"""
        self.model.eval()
        test_preds = []
        test_labels = []
        test_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, test_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='binary', zero_division=0
        )
        
        # Calculate AUC
        if len(np.unique(test_labels)) > 1:
            auc = roc_auc_score(test_labels, test_probs)
        else:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        
        logger.log(f'{split_name}_accuracy', accuracy)
        logger.log(f'{split_name}_precision', precision)
        logger.log(f'{split_name}_recall', recall)
        logger.log(f'{split_name}_f1', f1)
        logger.log(f'{split_name}_auc', auc)
        
        print(f"\n{split_name.capitalize()} Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return self.model
