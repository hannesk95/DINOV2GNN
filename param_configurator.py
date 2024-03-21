import configparser
import torch
import os


class ParamConfigurator:
    """Parameter configurator class for deep learning pipeline."""

    def __init__(self):
        """# TODO: Docstring"""
        config = configparser.ConfigParser()
        config.read('/home/johannes/Code/DINOV2GNN/config.ini')

        # Global
        self.seed = config['global'].getint('seed')
        self.device = config['global']['device']

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Data
        self.data_dir = config['data']['data_directory']
        self.artifact_dir = config['data']['artifact_directory']

        if not os.path.exists(self.artifact_dir):
            os.mkdir(self.artifact_dir)

        # Architecture
        self.model_name = config['architecture']['model_name']
        self.model_output = config['architecture']['model_output']
        assert self.model_output in ['cls', 'max', 'mean']
        self.hidden_channels = config['architecture'].getint('hidden_channels')
        self.gnn_type = config['architecture']['gnn_type']

        # Training
        self.batch_size = config['training'].getint('batch_size')
        self.epochs = config['training'].getint('epochs')
        self.num_workers = config['training'].getint('num_workers')        

        # Optimizer
        self.learning_rate = config['optimizer'].getfloat('learning_rate')        
        self.optimizer = config['optimizer']['optimizer']
        self.nesterov = config['optimizer'].getboolean('nesterov')
        self.momentum = config['optimizer'].getfloat('momentum')
        self.weight_decay = config['optimizer'].getfloat('weight_decay')
        self.scheduler_gamma = config['optimizer'].getfloat('scheduler_gamma')
        self.scheduler_step = config['optimizer'].getint('scheduler_step')


