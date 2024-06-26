import configparser
import torch
import os
import uuid


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
        self.dataset = config['data']['dataset']
        self.num_slices = config['data'].getint('num_slices')
        self.fraction = config['data'].getfloat('fraction')
        self.topology = config['data']['topology']
        self.artifact_dir = config['data']['artifact_directory']
        self.k = config['data'].getint("k")
        # self.similarity_metric = config['data']['similarity_metric']

        if not os.path.exists(self.artifact_dir):
            os.mkdir(self.artifact_dir)
        
        self.run_uuid = uuid.uuid4().hex
        self.run_dir = os.path.join(os.path.abspath(self.artifact_dir), self.run_uuid)        

        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)
        else:
            raise ValueError("Run dir does already exist!")

        # Architecture
        self.model_name = config['architecture']['model_name']
        # self.model_output = config['architecture']['model_output']
        # assert self.model_output in ['cls', 'max', 'mean']
        self.hidden_channels = config['architecture'].getint('hidden_channels')
        self.gnn_type = config['architecture']['gnn_type']
        self.gnn_readout = config['architecture']['gnn_readout']
        self.mlp_aggregation = config['architecture']['mlp_aggregation']
        self.mlp_conditional = config['architecture'].getboolean('mlp_conditional')

        match self.dataset:
            case "fracture":
                self.n_classes = 3 
            case "organ":
                self.n_classes = 11
            case _:
                self.n_classes = 2

        # Training
        self.batch_size = config['training'].getint('batch_size')
        self.epochs = config['training'].getint('epochs')
        self.num_workers = config['training'].getint('num_workers')        
        self.ce_gamma = config['training'].getfloat('ce_gamma')

        # Optimizer
        self.learning_rate = config['optimizer'].getfloat('learning_rate')        
        self.optimizer = config['optimizer']['optimizer']
        self.nesterov = config['optimizer'].getboolean('nesterov')
        self.momentum = config['optimizer'].getfloat('momentum')
        self.weight_decay = config['optimizer'].getfloat('weight_decay')
        self.scheduler_gamma = config['optimizer'].getfloat('scheduler_gamma')
        self.scheduler_step = config['optimizer'].getint('scheduler_step')
