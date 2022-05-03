import torch                    
import torch.nn as nn            
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from source.utils.label_mapper import LabelMapper
from sklearn.metrics import recall_score, precision_score, f1_score


@torch.no_grad()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def recall(outputs, labels, pos_label=0):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(recall_score(labels, preds, pos_label=pos_label, zero_division=1))  


@torch.no_grad()
def precision(outputs, labels, pos_label=0):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(precision_score(labels, preds, pos_label=pos_label, zero_division=1)) 


@torch.no_grad()
def F1Score(outputs, labels, pos_label=0):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(f1_score(labels, preds, pos_label=pos_label, zero_division=1)) 


class ImageClassificationBase(nn.Module):

    def __init__(self, binary_classification=False, pos_label=1):
        super(ImageClassificationBase, self).__init__()
        self.binary_classification = binary_classification
        self.pos_label = pos_label


    def training_step(self, batch, criterion, device):
        images, labels = self.__get_inputs(batch, device)
        out = self(images)   # Generate predictions
        if len(out.shape) and out.shape[1] == 1: # for 1d conv
            out = torch.squeeze(out, dim=1)             
        loss = criterion(out, labels) # Calculate loss
        accu = accuracy(out, labels)

        return loss, accu
    

    def validation_step(self, batch, criterion, device):
        images, labels = self.__get_inputs(batch, device)
        out = self(images)                    # Generate predictions
        if len(out.shape) and out.shape[1] == 1: # for 1d conv
            out =  torch.squeeze(out, dim=1)   
        loss = criterion(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        if self.binary_classification:
            r = recall(out, labels, self.pos_label)
            p = precision(out, labels, self.pos_label)
            f1 = F1Score(out, labels, self.pos_label)
            return {'Loss': loss.detach(), 'Accuracy': acc,
                'Recall': r, 'Precision': p, 'F1Score': f1}

        return {'Loss': loss.detach(), 'Accuracy': acc}
        

    def validation_epoch_end(self, outputs):
        batch_losses = [x['Loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['Accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        if self.binary_classification:
            batch_r = [x['Recall'] for x in outputs]
            epoch_r = torch.stack(batch_r).mean() 

            batch_p = [x['Precision'] for x in outputs]
            epoch_p = torch.stack(batch_p).mean() 

            batch_f1 = [x['F1Score'] for x in outputs]
            epoch_f1 = torch.stack(batch_f1).mean() 

            return {
                'Loss': epoch_loss.item(), 'Accuracy': epoch_acc.item(),
                'Recall': epoch_r.item(), 'Precision': epoch_p.item(), 'F1Score': epoch_f1.item()
            }

        return {'Loss': epoch_loss.item(), 'Accuracy': epoch_acc.item()}
    

    def epoch_end(self, epoch, result):
        print("Epoch :",epoch + 1)
        print(f'Train Accuracy:{result["train_accuracy"]*100:.2f}% Validation Accuracy:{result["Accuracy"]*100:.2f}%')
        print(f'Train Loss:{result["train_loss"]:.4f} Validation Loss:{result["Loss"]:.4f}')

    
    def __get_inputs(self, data, device):
        """get the inputs; data is a list of [inputs, labels]"""
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        return inputs, labels


    @torch.no_grad()
    def predict(self, data, device):
        self.eval()
        result = []
        for i, batch in enumerate(tqdm(data)):
            images, _ = self.__get_inputs(batch, device)
            out = self(images)  

            if len(out.shape) and out.shape[1] == 1: # for 1d conv
                out = torch.squeeze(out, dim=1) 

            _, preds = torch.max(out, dim=1)
            result += preds.tolist()
        return result
