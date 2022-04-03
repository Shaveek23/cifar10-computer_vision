import torch                    
import torch.nn as nn            
import torch.nn.functional as F

@torch.no_grad()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch, criterion, device):
        images, labels = self.__get_inputs(batch, device)
        out = self(images)                  # Generate predictions
        loss = criterion(out, labels) # Calculate loss
        accu = accuracy(out,labels)
        return loss, accu
    

    def validation_step(self, batch, criterion, device):
        images, labels = self.__get_inputs(batch, device)
        out = self(images)                    # Generate predictions
        loss = criterion(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'Loss': loss.detach(), 'Accuracy': acc}
        

    def validation_epoch_end(self, outputs):
        batch_losses = [x['Loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['Accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
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