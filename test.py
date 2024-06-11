import argparse
import collections
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import pickle
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from parse_config import ConfigParser

torch.cuda.empty_cache()

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader_eval']['type'])(
        config['data_loader_eval']['args']['data_dir'],
        batch_size=config['data_loader_eval']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        num_workers=config['data_loader_eval']['args']['num_workers']
    )
    data_loader_sampling = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=config['data_loader']['args']['validation_split'],
        num_workers=config['data_loader']['args']['num_workers']
    )
    data_loader_iterator = iter(data_loader_sampling)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics_eval']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    checkpoint_path = Path(config.resume)
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # total_loss = torch.tensor([0])
    total_metrics = torch.zeros(len(metric_fns))

    # total_loss = total_loss.to(device)
    total_metrics = total_metrics.to(device)

    # Constants
    num_vars = config['num_vars']
    
    all_outputs = []
    all_targets = []

    # Begin test
    n_samples = 0
    with torch.no_grad():
        # print("Loading min-max scaler skipped...")
        with open(config.resume.parent / "minmaxscaler.pkl",'rb') as f:
            delta = pickle.load(f)
        print("Loading data...")
        for batch_idx, (data,target) in tqdm(enumerate(data_loader)):
            target = torch.squeeze(target)
            target = target.cuda(device)
            data = torch.reshape(data, (-1, data.shape[-1]))
            data = data.cuda(device)
            output, _, data_loader_iterator, embeddings = model.inference(data,data_loader_iterator,
                                                                            data_loader_sampling, num_vars, scaler=delta)  

            all_outputs.append(output)
            all_targets.append(target)

            # computing loss, metrics on test set
            batch_size = target.shape[0]
            n_samples += batch_size

            for i, metric in enumerate(metric_fns):
                metric_val = metric(output, target)
                if type(metric_val) != float:
                    metric_val = metric_val
                total_metrics[i] = total_metrics[i] + metric_val * batch_size

    log = {}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    all_targets = torch.cat(all_targets)
    all_targets = all_targets.detach().cpu().numpy()

    all_outputs = torch.cat(all_outputs).detach().cpu().numpy()
    all_outputs = np.squeeze(all_outputs)

    if (config["data_loader"]["type"]=="PTBXLDataLoader" or config["data_loader"]["type"]=="TUSZDataLoader")and data_loader.dataset.seen_unseen_labels is not None:
        seen_normal_idx = np.logical_or(np.equal(data_loader.dataset.seen_unseen_labels,0),np.equal(data_loader.dataset.seen_unseen_labels,1))
        unseen_normal_idx = np.logical_or(np.equal(data_loader.dataset.seen_unseen_labels,0),np.equal(data_loader.dataset.seen_unseen_labels,2))
        seen_auc = roc_auc_score(all_targets[seen_normal_idx], all_outputs[seen_normal_idx])
        unseen_auc = roc_auc_score(all_targets[unseen_normal_idx], all_outputs[unseen_normal_idx])
        seen_p, seen_r, _ = precision_recall_curve(all_targets[seen_normal_idx], all_outputs[seen_normal_idx])
        seen_prc = auc(seen_r, seen_p)
        unseen_p, unseen_r, _ = precision_recall_curve(all_targets[unseen_normal_idx], all_outputs[unseen_normal_idx])
        unseen_prc = auc(unseen_r, unseen_p)
        logger.info({'seen_auc': seen_auc})
        logger.info({'seen_prc': seen_prc})
        logger.info({'unseen_auc': unseen_auc})
        logger.info({'unseen_prc': unseen_prc})

    normal_auc = roc_auc_score(1-all_targets, 1-all_outputs)
    abnormal_auc = roc_auc_score(all_targets, all_outputs)
    normal_p, normal_r, _ = precision_recall_curve(1-all_targets, 1-all_outputs)
    abnormal_p, abnormal_r, thresholds_prc = precision_recall_curve(all_targets, all_outputs)
    normal_prc = auc(normal_r, normal_p)
    abnormal_prc = auc(abnormal_r, abnormal_p)

    logger.info({'normal_auc': normal_auc})
    logger.info({'normal_prc': normal_prc})
    logger.info({'abnormal_auc': abnormal_auc})
    logger.info({'abnormal_prc': abnormal_prc})

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-s','--save_dir'], type=str, target='trainer;save_dir')
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
