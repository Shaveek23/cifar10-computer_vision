import os
from source.utils.config_manager import ConfigManager
import torch
import torch.nn as nn
import ray
from ray import tune
from source.utils.data_loading import get_data
from source.utils.data_loading import get_data


class HyperparameteresTunner:

    def __init__(self):
        self.checkpoint_dir = ConfigManager.get_checkpoints_path()
    
    def tune(self, dataset, config, scheduler, reporter, num_samples=1, resources_per_trial={"cpu": 1, "gpu": 1}, device = "cpu"):
        
        self.dataset = dataset
        self.device = device
        self.config = config

        result = tune.run(
            tune.with_parameters(self.__train_validate), 
            trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            resources_per_trial=resources_per_trial,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter)

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))

        # to do: add a static method to type method to get a new instance of class from config
        # best_trained_model = type(self.net)(best_trial.config["l1"], best_trial.config["l2"])
        best_trained_model = None

        return best_trial

    def __train_validate(self, config):

        self.config = config
        self.data_loaders_factory = self.config['data_loaders_factory']
        self.net, self.optimizer, self.criterion = self.__get_objects_from_config()

        self.train_iter = self.data_loaders_factory.get_train_loader(self.config['batch_size'])
        self.train_valid_iter = self.data_loaders_factory.get_train_valid_loader(self.config['batch_size'])
        self.valid_iter = self.data_loaders_factory.get_valid_loader(self.config['batch_size'])
        self.test_iter = self.data_loaders_factory.get_test_loader(self.config['batch_size'])
    
        
        if torch.cuda.is_available():
            self.device = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net)
        self.net.to(self.device)

        # TO DO: implement checkpoints

        # if self.checkpoint_dir:
        #     model_state, optimizer_state = torch.load(
        #         os.path.join(self.checkpoint_dir, "checkpoint"))
        #     self.net.load_state_dict(model_state)
        #     self.optimizer.load_state_dict(optimizer_state)
        
        for epoch in range(config["epochs"]):  # loop over the dataset multiple times
            self.__train(self.train_iter, epoch)
            val_loss, val_steps, correct, total = self.__validate(self.valid_iter)
            #self.__save_model_checkpoint(epoch)
            tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

        print("Finished Tuning")

    def __train(self, train_iter, epoch):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_iter, 0):
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            loss, _, _ =  self.__forward(data)  
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                self.__print_train_statistics(epoch, i, running_loss, epoch_steps)
                running_loss = 0.0

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

    def __save_model_checkpoint(self, epoch):
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((self.net.state_dict(), self.optimizer.state_dict()), path)

    def __get_objects_from_config(self):
        net_config = self.config['net']
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




