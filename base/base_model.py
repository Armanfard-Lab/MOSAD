import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import Iterator, Tuple
from torch.nn import Parameter


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError
    
    @abstractmethod
    def inference(self, *inputs):
        """
        Inference logic

        :return: Anomaly score and embeddings
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def grouped_parameters(self) -> Tuple[Iterator[Parameter], ...]:
        """
        Method from https://github.com/wagner-d/TimeSeAD/tree/master/timesead
        """
        return self.parameters(),