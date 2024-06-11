import argparse
import collections
import torch
import numpy as np
import subprocess
import pickle
import gc
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import MOSADTrainer
from utils import prepare_device

torch.cuda.empty_cache()

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    data_loader_sampling = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=True,
        validation_split=config['data_loader']['args']['validation_split'],
        num_workers=config['data_loader']['args']['num_workers'])
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # print additional notes for the log
    if config['notes'] is not None:
        message = config['notes']
        log = "Notes: " + message
        logger.info(log)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    metrics_eval = [getattr(module_metric, met) for met in config['metrics_eval']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = MOSADTrainer(model, criterion, metrics, metrics_eval, optimizer,
                    config=config,
                    device=device,
                    data_loader=data_loader,
                    data_loader_sampling=data_loader_sampling,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler,)
    
    trainer.train()

    trainer._resume_checkpoint(trainer.checkpoint_dir / 'model_best.pth')
    minmaxscaler = trainer.scalar_fitting()
    with open(config._save_dir / 'minmaxscaler.pkl', 'wb') as f:
        pickle.dump(minmaxscaler, f)

    del data_loader.dataset
    del data_loader
    del data_loader_sampling.dataset
    del data_loader_sampling
    gc.collect()

    # "Begin testing on test set..."
    subprocess.run(["python", "test.py", "-r", config._save_dir / 'model_best.pth'])

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--hidden_channels'], type=int, target='arch;args;hidden_channels'),
        CustomArgs(['--embedding_dim'], type=int, target='arch;args;embedding_dim'),
        CustomArgs(['--projection_dim'], type=int, target='arch;args;projection_dim'),
        CustomArgs(['--dropout_rate'], type=float, target='arch;args;dropout_rate'),
        CustomArgs(['--encoder_decoder_layers'], type=int, target='arch;args;encoder_decoder_layers'),
        CustomArgs(['--architecture'], type=str, target='arch;args;architecture'),
        CustomArgs(['--graph_type'], type=str, target='graph_type'),
        CustomArgs(['--lambda_c_max'], type=float, target='lamda_c_max'),
        CustomArgs(['--masked_recon'], type=bool, target='arch;args;masked_recon'),
        CustomArgs(['--model'], type=str, target='arch;type'),
        CustomArgs(['--num_heads'], type=int, target='arch;args;num_heads'),
        CustomArgs(['-s','--save_dir'], type=str, target='trainer;save_dir'),
        CustomArgs(['--gamma'], type=float, target='arch;args;gamma'),
        CustomArgs(['--con_inf_batch_size'], type=int, target='arch;args;con_inf_batch_size'),
        CustomArgs(['--shots'], type=int, target='iso_anom_shots'),
        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
    ]
    config = ConfigParser.from_args(args, options)

    main(config)
