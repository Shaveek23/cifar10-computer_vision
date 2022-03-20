import os
import torch
import torch.nn as nn
from ray import tune
import matplotlib.pyplot as plt
from source.training import epoch_step, plot_result


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

        data_loaders_factory = config['data_loaders_factory']
        net, optimizer, criterion = self.__get_objects_from_config(config)

        train_iter = data_loaders_factory.get_train_loader(config['batch_size'])
        valid_iter = data_loaders_factory.get_valid_loader(config['batch_size'])
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        net.to(device)

        if checkpoint_dir:
            model_state, optimizer_state, criterion_state = torch.load(
                os.path.join(checkpoint_dir, config['tuning_id']))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            criterion.load_state_dict(criterion_state)
        
        history = []
        for epoch in range(config["epochs"]):  # loop over the dataset multiple times
            epoch_result = epoch_step(net, train_iter, valid_iter, optimizer, epoch, device)
            self.__save_model_checkpoint(net, optimizer, epoch, config)
            tune.report(loss=epoch_result['Loss'], accuracy=epoch_result['Accuracy'], train_loss=epoch_result['train_loss'], train_accuracy=epoch_result['train_accuracy'])
            history.append(epoch_result)
            self.__save_acc_loss_plot(history, epoch)

        print("Finished Tuning")


    def __save_model_checkpoint(self, net, optimizer, epoch, config):
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, config['tuning_id'])
            torch.save((net.state_dict(), optimizer.state_dict()), path)
    

    def __save_acc_loss_plot(self, history, epoch):
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            plot_result(history, checkpoint_dir)
            

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

    
