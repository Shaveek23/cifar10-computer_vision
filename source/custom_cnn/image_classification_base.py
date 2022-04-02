import torch                    
import torch.nn as nn            
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt 


@torch.no_grad()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch, device):
        images, labels = self.__get_inputs(batch, device)
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        accu = accuracy(out,labels)
        return loss,accu
    

    def validation_step(self, batch, device):
        images, labels = self.__get_inputs(batch, device)
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
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

    def predict(self,data,device):
        number_to_label_dict = dict(zip(list(range(10)),
        ['airplane','automobile','bird','cat','deer',
        'dog','frog','horse','ship','truck']))
        result = []
        for i,batch in enumerate(tqdm(data)):
            images, _ = self.__get_inputs(batch, device)
            out = self(images)  
            print(out)
            _, out = torch.max(out, 1)
            result.append(number_to_label_dict[out[0].item()])
        return result

  