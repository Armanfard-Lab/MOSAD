import os
from base import BaseDataLoader
from utils import TUSZ, SMD, PTBXL
from definitions import ROOT_DIR

class SMDDataLoader(BaseDataLoader):
    """
    Load and process SMD data (non-graph)
    """
    def __init__(self, data_dir,
                 batch_size,
                 shuffle,
                 validation_split,
                 num_workers):
        self.data_dir = data_dir.split('/')

        self.dataset = SMD(
            input_dir=os.path.join(ROOT_DIR,self.data_dir[0],self.data_dir[1]),
            input_file=self.data_dir[2]+'.h5',
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class PTBXLDataLoader(BaseDataLoader):
    """
    Load and process PTBXL data (non-graph)
    """
    def __init__(self, data_dir,
                 batch_size,
                 shuffle,
                 validation_split,
                 num_workers):
        self.data_dir = data_dir.split('/')

        self.dataset = PTBXL(
            input_dir=os.path.join(ROOT_DIR,self.data_dir[0],self.data_dir[1]),
            input_file=self.data_dir[2]+'.h5',
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class TUSZDataLoader(BaseDataLoader):
    """
    Load and process TUSZ data (non-graph)
    """
    def __init__(self, data_dir,
                 batch_size,
                 shuffle,
                 validation_split,
                 num_workers):
        self.data_dir = data_dir.split('/')

        self.dataset = TUSZ(
            input_dir=os.path.join(ROOT_DIR,self.data_dir[0],self.data_dir[1]),
            input_file=self.data_dir[2]+'.h5',
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)