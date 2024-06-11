import numpy as np
import torch
import wandb
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils import getnormals, coe_batch, mixup_batch
from sklearn.preprocessing import MinMaxScaler

class MOSADTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, metric_eval_fns, optimizer, config, device,
                 data_loader, data_loader_sampling, valid_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, metric_eval_fns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.data_loader_sampling = data_loader_sampling
        self.data_loader_iterator = iter(self.data_loader_sampling)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.test_logger = self.config.get_logger('test')
        self.num_vars = self.config['num_vars']
        self.node_level = self.config['arch']['args']['node_level']

        # Loss weights
        self.lamda_c = self.config['lamda_c_max']
        self.lamda_d = self.config['lamda_d_max']

        # Cached anomaly data
        self.anom_data = self.data_loader.dataset.X_anom
        self.iso_anom_shots = self.config['iso_anom_shots']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, item in enumerate(self.data_loader):
            if len(item) == 2:
                data, target = item
            elif len(item) == 3:
                data, target, normal_seen = item
            data = data.cuda(self.device)
            target = torch.squeeze(target)
            target = target.cuda(self.device)

            if self.iso_anom_shots is not None: #if isolated anomalies are used
                data = data[target==0]
                target = target[target==0]
                if self.iso_anom_shots > 0: # randomly sample shots from isolated anomalies
                    anom_data = self.anom_data[torch.randint(0, self.anom_data.shape[0], (self.iso_anom_shots,))]
                    anom_data = anom_data.cuda(self.device)
                    anom_target = torch.ones(self.iso_anom_shots, dtype=torch.long).cuda(self.device)
                    data = torch.cat((data, anom_data), dim=0)
                    target = torch.cat((target, anom_target), dim=0)
                elif self.iso_anom_shots == 0: # unsupervised
                    pass
                elif self.iso_anom_shots == -1: # use all isolated anomalies
                    anom_data = self.anom_data.cuda(self.device)
                    anom_target = torch.ones(anom_data.shape[0], dtype=torch.long).cuda(self.device)
                    data = torch.cat((data, anom_data), dim=0)
                    target = torch.cat((target, anom_target), dim=0)

            data = data.transpose(1,2)
            
            if self.model.coe_rate > 0:
                x_oe, y_oe = coe_batch(
                    x=data,
                    y=target,
                    coe_rate=self.model.coe_rate,
                    suspect_window_length=self.config['arch']['args']['num_node_features'],
                )
                # Add COE to training batch
                data = torch.cat((data, x_oe), dim=0)
                target = torch.cat((target, y_oe), dim=0)

            if self.model.mixup_rate > 0.0:
                x_mixup, y_mixup = mixup_batch(
                    x=data,
                    y=target,
                    mixup_rate=self.model.mixup_rate,
                )
                # Add Mixup to training batch
                data = torch.cat((data, x_mixup), dim=0)
                target = torch.cat((target, y_mixup), dim=0)
            
            data = data.transpose(1,2)
            data = torch.reshape(data, (-1, data.shape[-1]))

            self.optimizer.zero_grad()
            
            output_ncd = self.model(data, target, self.num_vars)
            data_n = getnormals(data, target, self.num_vars, node_labels=self.node_level)  
            loss = self.criterion(output_ncd, data_n, target, lamda_c=self.lamda_c, lamda_d=self.lamda_d)
            loss.backward()
            self.optimizer.step()
            
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, item in enumerate(self.valid_data_loader):
                if len(item) == 2:
                    data, target = item
                elif len(item) == 3:
                    data, target, normal_seen = item
                data = data.cuda(self.device)
                target = torch.squeeze(target)
                target = target.cuda(self.device)

                if self.iso_anom_shots is not None: #if isolated anomalies are used
                    data = data[target==0]
                    target = target[target==0]
                    if self.iso_anom_shots > 0: # randomly sample shots from isolated anomalies
                        anom_data = self.anom_data[torch.randint(0, self.anom_data.shape[0], (self.iso_anom_shots,))]
                        anom_data = anom_data.cuda(self.device)
                        anom_target = torch.ones(self.iso_anom_shots, dtype=torch.long).cuda(self.device)
                        data = torch.cat((data, anom_data), dim=0)
                        target = torch.cat((target, anom_target), dim=0)
                    elif self.iso_anom_shots == 0: # unsupervised
                        pass
                    elif self.iso_anom_shots == -1: # use all isolated anomalies
                        anom_data = self.anom_data.cuda(self.device)
                        anom_target = torch.ones(anom_data.shape[0], dtype=torch.long).cuda(self.device)
                        data = torch.cat((data, anom_data), dim=0)
                        target = torch.cat((target, anom_target), dim=0)

                data = data.transpose(1,2)

                if self.model.coe_rate > 0:
                    x_oe, y_oe = coe_batch(
                        x=data,
                        y=target,
                        coe_rate=self.model.coe_rate,
                        suspect_window_length=self.config['arch']['args']['num_node_features'],
                    )
                    # Add COE to training batch
                    data = torch.cat((data, x_oe), dim=0)
                    target = torch.cat((target, y_oe), dim=0)

                if self.model.mixup_rate > 0.0:
                    x_mixup, y_mixup = mixup_batch(
                        x=data,
                        y=target,
                        mixup_rate=self.model.mixup_rate,
                    )
                    # Add Mixup to training batch
                    data = torch.cat((data, x_mixup), dim=0)
                    target = torch.cat((target, y_mixup), dim=0)

                data = data.transpose(1,2)
                data = torch.reshape(data, (-1, data.shape[-1]))

                output_ncd = self.model(data, target, self.num_vars)
                data_n = getnormals(data, target, self.num_vars, node_labels=self.node_level)
                loss = self.criterion(output_ncd, data_n, target, lamda_c=self.lamda_c, lamda_d=self.lamda_d)

                self.valid_metrics.update('loss', loss.item())

        return self.valid_metrics.result()

    def scalar_fitting(self):
        """
        Fit a mix max scalar for the reconstruction score based on the validation set

        :param None
        :return: Fitted scalar
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            print("Begin threshold finding...")
            print("Fitting min-max scaler...")
            delta = MinMaxScaler()
            # delta1 = MinMaxScaler()
            for batch_idx, item in tqdm(enumerate(self.valid_data_loader)):
                if len(item) == 3:
                    data, target, normal_seen = item
                else:
                    data, target = item

                target = torch.squeeze(target)

                # remove abnormal data
                data = data[target==0]
                target = target[target==0]

                data = data.cuda(self.device)
                target = target.cuda(self.device)

                data = torch.reshape(data, (-1, data.shape[-1]))

                _, mse, self.data_loader_iterator, _ = self.model.inference(data, self.data_loader_iterator, 
                                                                            self.data_loader_sampling, self.num_vars)
                delta.partial_fit(mse.detach().cpu().reshape(-1, 1))
                print("Fitted min and max= ", delta.data_min_, delta.data_max_) 

        return delta

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
