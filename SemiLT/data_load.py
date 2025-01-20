import scanpy as sc
import numpy as np
import os
from setting import Setting
import torch
import torch.utils.data as data
import random
random.seed(1)
from scipy.sparse import csr_matrix

def normalize_csr_matrix_columns(csr_matrix):
    max_values = csr_matrix.max(axis=0).toarray().flatten()
    min_values = csr_matrix.min(axis=0).toarray().flatten()
    csr_matrix.data = (csr_matrix.data - min_values[csr_matrix.indices]) / (max_values[csr_matrix.indices] - min_values[csr_matrix.indices])
    return csr_matrix
    
def h5ad_read(file_name):
    print('load h5ad matrix:', file_name)
    data = sc.read(file_name)
    # data.X = normalize_csr_matrix_columns(data.X)
    # sc.pp.normalize_total(data,exclude_highly_expressed=True)
    # sc.pp.log1p(data)
    # sc.pp.scale(data, max_value=10) 
    # data.X = csr_matrix(data.X)
    return data
    
def data_read(file_name):
    print('load h5ad matrix:', file_name)
    data = sc.read(file_name)
    sc.pp.normalize_total(data, target_sum = 1e4, exclude_highly_expressed=True)
    sc.pp.log1p(data)
    return data

def str_convert_to_numeric(cell_types):
    cell_type_mapping = {}
    unique_cell_types = sorted(set(cell_types))
    for index, cell_type in enumerate(unique_cell_types):
        cell_type_mapping[cell_type] = index
    numeric_values = [cell_type_mapping[cell_type] for cell_type in cell_types]
    return cell_type_mapping, np.array(numeric_values)

def reverse_mapping(mapping):
    reversed_mapping = {v: k for k, v in mapping.items()}
    return reversed_mapping
    
def numeric_convert_to_str(mapping, numeric_values):
    numeric_mapping = reverse_mapping(mapping)
    cell_types = [numeric_mapping[value] for value in numeric_values]
    return cell_types

def read_from_file(data_path, protien_path = None ,peak_path = None,label = False):
    data_path = os.path.join(os.path.realpath('.'), data_path)

    labels = None
    label_index = None
    input_size, input_size_protein = 0, 0
    
    data_reader = h5ad_read(data_path)
    protein_reader = None
    peak_reader = None
    if label is True:          
        label_index, labels = str_convert_to_numeric(data_reader.obs['cell_type'].tolist())
    if protien_path is not None:
        protien_path = os.path.join(os.path.realpath('.'), protien_path)    
        protein_reader = h5ad_read(protien_path).X
    if peak_path is not None:
        peak_path = os.path.join(os.path.realpath('.'), peak_path)    
        peak_reader = h5ad_read(peak_path).X
    data_reader = data_reader.X
    return data_reader, labels, label_index, protein_reader, peak_reader


class Dataloader(data.Dataset):
    def __init__(self, train = True, data_reader = None, labels = None, protein_reader = None):
        self.train = train        
        self.data_reader, self.labels, self.protein_reader = data_reader, labels, protein_reader
        self.input_size = self.data_reader.shape[1]
        self.sample_num = self.data_reader.shape[0]
        self.input_size_protein = None
        if protein_reader is not None:
            self.input_size_protein = self.protein_reader.shape[1]
    def __getitem__(self, index):
        if self.train:
            # get atac data            
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data_reader[rand_idx].todense())
            sample = sample.reshape((1, self.input_size))
            # result = np.zeros_like(sample)
            # result[sample > 0] = 0.5
            # result[sample > 0.4] = 1
            in_data = (sample>0).astype(np.float)  # binarize data #sample.astype(np.float)  #
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[rand_idx].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
            in_label = self.labels[rand_idx]
            return in_data, in_label

        else:
            sample = np.array(self.data_reader[index].todense())
            sample = sample.reshape((1, self.input_size))
            # result = np.zeros_like(sample)
            # result[sample > 0] = 0.5
            # result[sample > 0.4] = 1
            in_data = (sample>0).astype(np.float)  # binarize data #sample.astype(np.float)  #
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[index].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
            in_label = self.labels[index]
            return in_data, in_label
    def __len__(self):
        return self.sample_num


class DataloaderWithoutLabel(data.Dataset):
    def __init__(self, train = True, data_reader = None, labels = None, protein_reader = None, peak_reader = None):
        self.train = train
        self.data_reader, self.labels, self.protein_reader ,self.peak_reader = data_reader, labels, protein_reader,peak_reader
        self.input_size = self.data_reader.shape[1]
        self.sample_num = self.data_reader.shape[0]        
        self.input_size_protein = None
        self.input_size_peak = None        
        if protein_reader is not None:
            self.input_size_protein = self.protein_reader.shape[1]
        if peak_reader is not None:
            self.input_size_peak = self.peak_reader.shape[1] 
            
    def __getitem__(self, index):
        if self.train:
            # get atac data
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data_reader[rand_idx].todense())
            sample = sample.reshape((1, self.input_size))
            # result = np.zeros_like(sample)
            # result[sample > 0] = 0.5
            # result[sample > 0.4] = 1
            in_data = (sample>0).astype(np.float)  # binarize data #sample.astype(np.float)  #
            peak_data = None
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[rand_idx].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
            if self.input_size_peak is not None:
                sample_peak = np.array(self.peak_reader[rand_idx].todense())
                sample_peak = sample_peak.reshape((1, self.input_size_peak))
                # sample_peak = (sample_peak>0).astype(np.float) # binarize
                in_data = np.concatenate((in_data, sample_peak), 1)
            return in_data
        else:
            sample = np.array(self.data_reader[index].todense())
            sample = sample.reshape((1, self.input_size))
            # result = np.zeros_like(sample)
            # result[sample > 0] = 0.5
            # result[sample > 0.4] = 1
            in_data = (sample>0).astype(np.float)  # binarize data #sample.astype(np.float)  #
            peak_data = None
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[index].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
            if self.input_size_peak is not None:
                sample_peak = np.array(self.peak_reader[index].todense())
                sample_peak = sample_peak.reshape((1, self.input_size_peak))
                # sample_peak = (sample_peak>0).astype(np.float) # binarize
                in_data = np.concatenate((in_data, sample_peak), 1)
            return in_data
    def __len__(self):
        return self.sample_num


class PrepareDataloader():
    def __init__(self, setting):
        self.setting = setting
        # hardware constraint
        num_workers = self.setting.threads - 1
        if num_workers < 0:
            num_workers = 0
        print('num_workers:', num_workers)
        kwargs = {'num_workers': num_workers, 'pin_memory': False} # 0: one thread, 1: two threads ...
        
        # load data with peak

        train_rna_loaders = []
        test_rna_loaders = []
        #load RNA+ADT
        if len(setting.rna_paths) == len(setting.rna_protein_paths):
            for rna_path, rna_protein_path in zip(setting.rna_paths, setting.rna_protein_paths):
                data_reader, labels, label_index, protein_reader, _ = read_from_file(rna_path, rna_protein_path,label = True)
                # train loader
                trainset = Dataloader(True, data_reader, labels, protein_reader)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                setting.batch_size, shuffle=True, **kwargs)                        
                train_rna_loaders.append(trainloader)
                # test loader
                trainset = Dataloader(False, data_reader, labels, protein_reader)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                setting.batch_size, shuffle=False, **kwargs)                        
                test_rna_loaders.append(trainloader)
        #load RNA
        else:
            for rna_path in zip(setting.rna_paths):  
                data_reader, labels, label_index, _, _ = read_from_file(rna_path[0], label= True)
                # train loader 
                trainset = Dataloader(True, data_reader, labels)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                setting.batch_size, shuffle=True, **kwargs)                        
                train_rna_loaders.append(trainloader)        
                # test loader 
                trainset = Dataloader(False, data_reader, labels)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                setting.batch_size, shuffle=False, **kwargs)                        
                test_rna_loaders.append(trainloader)
                
  
        train_atac_loaders = []
        test_atac_loaders = []
        self.num_of_atac = 0
        if len(setting.atac_paths) == len(setting.atac_protein_paths):
            for atac_path, atac_protein_path,peak_path in zip(setting.atac_paths, setting.atac_protein_paths,setting.peak_paths):  
                data_reader, _,_, protein_reader, peak_reader = read_from_file(atac_path, atac_protein_path,peak_path)
                # train loader
                trainset = DataloaderWithoutLabel(True, data_reader, None, protein_reader,peak_reader)
                self.num_of_atac += len(trainset)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                setting.batch_size, shuffle=True, **kwargs)                        
                train_atac_loaders.append(trainloader)                  
                # test loader
                trainset = DataloaderWithoutLabel(False, data_reader, None, protein_reader)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                setting.batch_size, shuffle=False, **kwargs)                        
                test_atac_loaders.append(trainloader)
        else:
            for atac_path ,peak_path in zip(setting.atac_paths,setting.peak_paths):   
                data_reader, _, _, _, peak_reader = read_from_file(atac_path,None,peak_path)
                # train loader
                trainset = DataloaderWithoutLabel(True, data_reader,None,None,peak_reader)
                self.num_of_atac += len(trainset)
                
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                setting.batch_size, shuffle=True, **kwargs)                        
                train_atac_loaders.append(trainloader)                   
                # test loader
                trainset = DataloaderWithoutLabel(False, data_reader)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                setting.batch_size, shuffle=False, **kwargs)                        
                test_atac_loaders.append(trainloader)    
                                                                
        self.train_rna_loaders = train_rna_loaders
        self.test_rna_loaders = test_rna_loaders
        self.train_atac_loaders = train_atac_loaders
        self.test_atac_loaders = test_atac_loaders

    def getloader(self):
        return self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders,self.test_atac_loaders, int(self.num_of_atac/self.setting.batch_size)

        