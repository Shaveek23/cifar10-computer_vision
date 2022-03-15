from cProfile import label
import os
from random import sample
from source.utils.config_manager import ConfigManager
import torch
import torch.nn as nn
import ray
from ray import tune
from source.utils.data_loading import get_data
from source.utils.data_loading import get_data
import matplotlib.pyplot as plt


class HyperparameteresTunner:
    
    def tune(self, dataset, local_dir, config, scheduler, reporter, num_samples=1, resources_per_trial={"cpu": 1, "gpu": 1}, device = "cpu"):
        
        self.dataset = dataset
        self.device = device
        self.config = config

        result = tune.run(
            tune.with_parameters(self.__train_validate), 
            trial_dirname_creator=lambda trial: f"HyperparametersTunner_{config['tuning_id']}_{trial.trial_id}",
            resources_per_trial=resources_per_trial,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            checkpoint_score_attr='accuracy',
            local_dir=local_dir
            )

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))

        best_model, best_optimizer, best_criterion = self.__get_objects_from_config(best_trial.config)
 
        # if torch.cuda.is_available():
        #     if self.device == "cuda:0":
        #         if gpus_per_trial > 1:
        #             best_trained_model = nn.DataParallel(best_trained_model)
        # best_trained_model.to(self.device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state, criterion_state = torch.load(os.path.join(
            best_checkpoint_dir, self.config['tuning_id']))

        best_model.load_state_dict(model_state)
        best_optimizer.load_state_dict(optimizer_state)
        best_criterion.load_state_dict(criterion_state)

        return best_model, best_optimizer, best_criterion

    def __train_validate(self, config, checkpoint_dir=None):

        self.config = config
        self.data_loaders_factory = self.config['data_loaders_factory']
        self.net, self.optimizer, self.criterion = self.__get_objects_from_config(self.config)

        self.train_iter = self.data_loaders_factory.get_train_loader(self.config['batch_size'])
        self.train_valid_iter = self.data_loaders_factory.get_train_valid_loader(self.config['batch_size'])
        self.valid_iter = self.data_loaders_factory.get_valid_loader(self.config['batch_size'])
        self.test_iter = self.data_loaders_factory.get_test_loader(self.config['batch_size'])
    
        self.__reset_history_plot()
        
        if torch.cuda.is_available():
            self.device = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net)
        self.net.to(self.device)

        if checkpoint_dir:
            model_state, optimizer_state, criterion_state = torch.load(
                os.path.join(checkpoint_dir, self.config['tuning_id']))
            self.net.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            self.criterion.load_state_dict(criterion_state)
        
        for epoch in range(self.config["epochs"]):  # loop over the dataset multiple times
            self.__train(self.train_iter, epoch)
            val_loss, val_steps, correct, total = self.__validate(self.valid_iter)
            self.x_epoch.append((int)(epoch + 1))
            self.__save_model_checkpoint(epoch, self.config)
            tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
            
        print("Finished Tuning")

    def __train(self, train_iter, epoch):
        running_loss = 0.0
        running_corrects = 0.0 # the number of predicted IDs that match the actual ID label.
        epoch_steps = 0
        for i, data in enumerate(train_iter, 0):
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            loss, outputs, labels =  self.__forward(data)  
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item() # loss.item() <- gives the average loss of the batch
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                self.__print_train_statistics(epoch, i, running_loss, epoch_steps)
            
            _, predicted = torch.max(outputs.data, 1)
            running_corrects += (predicted == labels).sum().item()
        
        self.__update_loss_error_history('train', running_loss, running_corrects, train_iter.batch_size, len(train_iter.dataset.samples))
      
    def __validate(self, valid_iter):
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valid_iter, 0):
            with torch.no_grad():
                loss, outputs, labels = self.__forward(data)

                val_loss += loss.cpu().numpy()
                val_steps += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.__update_loss_error_history('val', val_loss, correct, valid_iter.batch_size, len(valid_iter.dataset.samples))

        return val_loss, val_steps, correct, total

    def __get_inputs(self, data):
        """get the inputs; data is a list of [inputs, labels]"""
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        return inputs, labels

    def __forward(self, data):
        inputs, labels = self.__get_inputs(data)
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        return loss, outputs, labels

    def __print_train_statistics(self, epoch, i, running_loss, epoch_steps):
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))

    def __save_model_checkpoint(self, epoch, config):
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, config['tuning_id'])
            torch.save((self.net.state_dict(), self.optimizer.state_dict(), self.criterion.state_dict()), path)
            self.__draw_and_save_curve(checkpoint_dir, epoch)

    def __get_objects_from_config(self, config):
        net_config = config['net']
        type, optimizer_config, criterion_config, net_args = \
            self.__unpack_net_config(**net_config)
        net = type(**net_args)
        optimizer = self.__get_optimizer_from_config(net.parameters(), **optimizer_config)
        criterion = self.__get_criterion_from_config(**criterion_config)

        return net, optimizer, criterion

    def __unpack_net_config(self, type, optimizer, criterion, **kwargs):
        return type, optimizer, criterion, kwargs

    def __get_optimizer_from_config(self, params, type, **kwargs):
        return type(params, **kwargs)
    
    def __get_criterion_from_config(self, type, **kwargs):
        return type(**kwargs)

    def __update_loss_error_history(self, phase, running_loss, running_corrects, batch_size, samples_count):
        
        # average loss per image in the epoch
        epoch_loss = running_loss * batch_size / samples_count
       
        # the fraction of correct ID predictions from the dataset in the epoch
        epoch_acc = running_corrects / samples_count
       
        self.y_loss[phase].append(epoch_loss)
        self.y_err[phase].append(1 - epoch_acc)

    def __reset_history_plot(self):
        self.y_loss = {}  # loss history
        self.y_loss['train'] = []
        self.y_loss['val'] = []
        self.y_err = {} # err history
        self.y_err['train'] = []
        self.y_err['val'] = []
        self.x_epoch = []

        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, title="loss")
        self.ax1 = self.fig.add_subplot(122, title="top1err")


    def __draw_and_save_curve(self, path, i):
        self.ax0.plot(self.x_epoch, self.y_loss['train'], 'bo-', label='train')
        self.ax0.plot(self.x_epoch, self.y_loss['val'], 'ro-', label='val')
        self.ax1.plot(self.x_epoch, self.y_err['train'], 'bo-', label='train')
        self.ax1.plot(self.x_epoch, self.y_err['val'], 'ro-', label='val')
        if i == 0:
            self.ax0.legend()
            self.ax1.legend()
                   
        self.fig.savefig(os.path.join(path, 'loss_curve.jpg'))
    
