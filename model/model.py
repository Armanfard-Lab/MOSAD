import torch
import torch.nn.functional as F
from base import BaseModel
from utils import getnormals, get_mask_per_node
from .mosad import  RecHead, DevHead, SupconHead1D

from .network import TCNEncoder

class MOSAD(BaseModel):
    def __init__(self, num_vars,
                 num_node_features,
                 embedding_dim,
                 projection_dim,
                 dropout_rate,
                 encoder_decoder_layers,
                 projection_layers,
                 detector_layers,
                 tcn_kernel_size,
                 tcn_out_channels,
                 tcn_maxpool_out_channels,
                 normalize_embedding,
                 masked_recon,
                 con_inf,
                 dev_inf,
                 node_level,
                 coe_rate,
                 mixup_rate,
                 con_inf_batch_size):
        super().__init__()
        self.encoder = TCNEncoder(in_channels=num_vars, out_channels=embedding_dim, kernel_size=tcn_kernel_size,
                                  tcn_channels=tcn_out_channels, tcn_layers=encoder_decoder_layers, 
                                  tcn_out_channels=tcn_out_channels, maxpool_out_channels=tcn_maxpool_out_channels,
                                  normalize_embedding=normalize_embedding)

        self.rechead = RecHead(in_channels=num_vars, kernel_size=tcn_kernel_size, tcn_channels=tcn_out_channels,
                                  tcn_layers=encoder_decoder_layers, input_length=num_node_features)
        self.conhead = SupconHead1D(embedding_dim, projection_dim, dropout_rate, projection_layers, con_inf_batch_size, 
                                    node_level)
        self.devhead = DevHead(embedding_dim, detector_layers, dropout_rate)

        self.con_inf = con_inf
        self.dev_inf = dev_inf
        self.masked = masked_recon
        self.coe_rate = coe_rate
        self.mixup_rate = mixup_rate
        self.node_level = node_level

    def forward(self, x, targets, num_vars):
        if self.masked:
            x_n = getnormals(x, targets, num_vars, self.node_level)
            x_hat_all = torch.zeros((x_n.shape[0],x.shape[1]),device=x.device)
            for var in range(num_vars):
                mask, neg_mask = get_mask_per_node(x_n.shape[0], num_vars, x.shape[1], x_n.shape[1], x.device,
                                                    parallel=False, var=var)
                x_hat = x_n * mask
                x_hat = x_hat.reshape(-1, num_vars, x_hat.shape[-1])
                x_hat = self.encoder(x_hat)
                x_hat = self.rechead(x_hat)
                x_hat = x_hat.reshape(-1, x_hat.shape[-1])
                x_hat_all = torch.add(x_hat_all, x_hat * neg_mask)
            x_n = x_hat_all
            x = x.reshape(-1, num_vars, x.shape[-1])
            x = self.encoder(x)
        else:
            x = x.reshape(-1, num_vars, x.shape[-1])
            x = self.encoder(x)
            
            # REMOVES REC HEAD (comment next 3 lines) 1/7
            x_n = x[torch.round(targets)==0]
            x_n = self.rechead(x_n)
            x_n = x_n.reshape(-1, x_n.shape[-1])
            # REMOVES REC HEAD (uncomment next 3 lines) 2/7
            # x_n = torch.zeros(x_shape, device=x.device)
            # x_n = x_n[torch.round(targets)==0]
            # x_n = x_n.reshape(-1, x_n.shape[-1])

        x_c = self.conhead(x)
        x_d = self.devhead(x)

        return (x_n, x_c, x_d)

    def inference(self, x, data_loader_iterator, data_loader_sampling, num_vars, scaler=None):
        if self.masked:
            x_hat_all = torch.zeros((x.shape[0],x.shape[1]),device=x.device)
            for var in range(num_vars):
                mask, neg_mask = get_mask_per_node(x.shape[0], num_vars, x.shape[1], x.shape[1], x.device,
                                                    parallel=False, var=var)
                x_hat = x * mask
                x_hat = x_hat.reshape(-1, num_vars, x_hat.shape[-1])
                x_hat = self.encoder(x_hat)
                x_hat = self.rechead(x_hat)
                x_hat = x_hat.reshape(-1, x_hat.shape[-1])
                x_hat_all = torch.add(x_hat_all, x_hat * neg_mask)
            x_hat = x_hat_all.reshape(-1, num_vars, x_hat_all.shape[-1])
            x = x.reshape(-1, num_vars, x.shape[-1])
            x_embed = self.encoder(x)
        else:
            x = x.reshape(-1, num_vars, x.shape[-1])
            x_embed = self.encoder(x)
            x_hat = self.rechead(x_embed) # REMOVES REC HEAD (comment this line) 3/7 
            # x_hat = x # REMOVES REC HEAD (uncomment this line) 4/7

        mse = F.mse_loss(x_hat,x,reduction='none')
        mse = torch.mean(mse, (1,2))
        mse = mse
        # mse = mse * 0  # REMOVES REC HEAD (Uncomment this line) 5/7

        # REMOVES REC HEAD (comment next 3 lines) 6/7
        if scaler is not None:
            mse = scaler.transform(mse.detach().cpu().reshape(-1, 1))
            mse = torch.squeeze(torch.from_numpy(mse).to(x.device))

        score = mse 
    
        if self.dev_inf:
            dev = self.devhead(x_embed)
            dev = torch.squeeze(dev)
            score = score + dev
    
        if self.con_inf:
            sim_score, data_loader_iterator = self.conhead.inference(x_embed, data_loader_iterator, 
                                                                     data_loader_sampling, self.encoder)
            score = score + sim_score

        return score, mse, data_loader_iterator, x_embed
