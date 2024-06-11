import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import TCN


class RecHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        tcn_channels: int,
        tcn_layers: int,
        input_length: int,
    ):
        super(RecHead, self).__init__()
        # (154, 1, 120)
        self.upsample = torch.nn.Upsample(size=input_length)
        # (154, 1, 200)

        dilations = [2**i for i in range(tcn_layers+1)] 
        dilations.sort(reverse=True)
        filters = [tcn_channels] * tcn_layers + [in_channels]

        self.tcn = TCN(
            input_dim=1,
            nb_filters=filters,
            kernel_size=kernel_size,
            nb_stacks=1,
            dilations=dilations,
            padding='causal',
            use_skip_connections=False,
            dropout_rate=0.0,
            return_sequences=True,
            activation=torch.nn.LeakyReLU(),
            use_batch_norm=False,
            use_layer_norm=False
        )
        # (154, 38, 200)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.upsample(x)
        x = self.tcn(x)
        return x
    
class DevHead(nn.Module):
    def __init__(self, embedding_dim, 
                 detector_layers, 
                 dropout_rate):
        super().__init__()

        if detector_layers == 1:
            self.detector = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, 1)
            )
        elif detector_layers == 2:
            self.detector = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.PReLU(),
                nn.BatchNorm1d(embedding_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, 1)
            )
        else:
            raise NotImplementedError
        
    def forward(self, x):
        x = self.detector(x)
        return torch.abs(x)
    

class SupconHead1D(nn.Module):
    def __init__(self, embedding_dim,
                 projection_dim,
                 dropout_rate,
                 projection_layers,
                 inf_batch_size,
                 node_level):
        super().__init__()

        if projection_layers == 1:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim)
            )
        elif projection_layers == 2:
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.PReLU(),
                nn.BatchNorm1d(embedding_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, projection_dim)
            )
        else:
            raise NotImplementedError("Projection layers must be 1 or 2")

        self.inf_batch_size = inf_batch_size
        self.node_level = node_level

    def forward(self, x):
        x = self.projector(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def inference(self, x_embed, data_loader_iterator, data_loader_sampling, encoder):
        # create sample data
        norm_samples = []
        while len(norm_samples) < self.inf_batch_size:
            try:
                item = next(data_loader_iterator)
                rand_sample = item[0]
                rand_y = item[1]
            except StopIteration:
                data_loader_iterator = iter(data_loader_sampling)
                item = next(data_loader_iterator)
                rand_sample = item[0]
                rand_y = item[1]
            if not sum(rand_y) > 0 and len(norm_samples) < self.inf_batch_size:
                norm_samples.append(rand_sample)

        norm_x = torch.cat(norm_samples, dim=0)
        norm_x = norm_x.to(x_embed.device)
        norm_embed = encoder(norm_x)

        # project
        norm_embed = self.forward(norm_embed)
        x_embed = self.forward(x_embed)

        # calculate cosine similarity 
        x_embed = x_embed / x_embed.norm(dim=1)[:, None]
        norm_embed = norm_embed / norm_embed.norm(dim=1)[:, None]

        sim_norm = torch.mm(x_embed, norm_embed.transpose(0, 1))
        sim_norm = torch.mean(sim_norm, 1)

        # calculate sim anomaly score
        sim_score = 1 - sim_norm

        return sim_score, data_loader_iterator
    