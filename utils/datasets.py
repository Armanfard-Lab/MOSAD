from torch.utils.data import Dataset
import torch
import h5py
import os

class SMD(Dataset):
    def __init__(
            self,
            input_dir: object = "ServerMachineDataset",
            input_file: object = None,
            data_augment: object = False):
        """
        Args: 
            input_dir: dir to preprocessed data
            input_file: file of dataset to be loaded
            data_augment: 
        Returns:
            x: tensor of input features
            y: tensor of label
            adj_mat: adjacency matrix for graph
        """        
        self.input_dir = input_dir
        self.input_file = os.path.join(input_dir, input_file)
        self.data_augment = data_augment

        with h5py.File(self.input_file, "r") as f:
            self.X = f['X'][()] 
            self.targets = f['y'][()]
            try:
                self.X_anom = f['X_anom'][()]
                self.num_iso_anom = self.X_anom.shape[0]
                self.X_anom = torch.FloatTensor(self.X_anom)
            except:
                self.X_anom = None
                self.num_iso_anom = 0   

    def __len__(self):
        return self.targets.size
    
    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (x, y, adj_mat)
        """

        # data augmentation
        # Not implemented
        swap_nodes = None
        
        x = torch.FloatTensor(self.X[idx,:,:])
        y = self.targets[idx]
        y = torch.LongTensor([y])

        return x, y
    
class PTBXL(Dataset):
    def __init__(
            self,
            input_dir: object = "PTBXL",
            input_file: object = None,
            data_augment: object = False):
        """
        Args: 
            input_dir: dir to preprocessed data
            input_file: file of dataset to be loaded
            data_augment: 
        Returns:
            x: tensor of input features
            y: tensor of label
            adj_mat: adjacency matrix for graph
        """        
        self.input_dir = input_dir
        self.input_file = os.path.join(input_dir, input_file)
        self.data_augment = data_augment

        with h5py.File(self.input_file, "r") as f:
            self.X = f['X'][()] 
            self.targets = f['y'][()]
            try:
                self.X_anom = f['X_anom'][()]
                self.X_anom = torch.FloatTensor(self.X_anom)
                self.X_anom = torch.transpose(self.X_anom, 1, 2)
                self.num_iso_anom = self.X_anom.shape[0]
            except:
                self.X_anom = None
                self.num_iso_anom = None  
            try:
                self.normal_seen_labels = f['normal_seen_labels'][()]
            except:
                self.normal_seen_labels = None
            try:
                self.seen_unseen_labels = f['seen_unseen_labels'][()]
            except:
                self.seen_unseen_labels = None
            try:
                self.cls_labels = f['cls_labels'][()]
            except:
                self.cls_labels = None

    def __len__(self):
        return self.targets.size
    
    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (x, y, adj_mat)
        """
        
        x = torch.FloatTensor(self.X[idx,:,:])
        x = torch.transpose(x, 0, 1)
        y = self.targets[idx]
        y = torch.LongTensor([y])

        if self.normal_seen_labels is not None:
            z = torch.LongTensor([self.normal_seen_labels[idx]])
            return x, y, z
        else:
            return x, y
    
    

class TUSZ(Dataset):
    def __init__(
            self,
            input_dir: object = "TUSZ",
            input_file: object = None,
            data_augment: object = False):
        """
        Args: 
            input_dir: dir to preprocessed data
            input_file: file of dataset to be loaded
            data_augment: 
        Returns:
            x: tensor of input features
            y: tensor of label
            adj_mat: adjacency matrix for graph
        """        
        self.input_dir = input_dir
        self.input_file = os.path.join(input_dir, input_file)
        self.data_augment = data_augment

        with h5py.File(self.input_file, "r") as f:
            self.X = f['X'][()] 
            self.targets = f['y'][()]
            try:
                self.X_anom = f['X_anom'][()]
                self.num_iso_anom = self.X_anom.shape[0]
                self.X_anom = torch.FloatTensor(self.X_anom)
            except:
                self.X_anom = None
                self.num_iso_anom = 0  
            try:
                self.seen_unseen_labels = f['seen_unseen_labels'][()]
            except:
                self.seen_unseen_labels = None  

    def __len__(self):
        return self.targets.size
    
    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (x, y, adj_mat)
        """

        # data augmentation
        # Not implemented
        swap_nodes = None
        
        x = torch.FloatTensor(self.X[idx,:,:])
        y = self.targets[idx]
        y = torch.LongTensor([y])

        return x, y