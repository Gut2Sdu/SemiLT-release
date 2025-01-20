import torch
import os

class Setting(object):
    def __init__(self):
        DB = 'demo'
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
            
        if DB == "demo":
            self.number_of_class = 7 # Number of cell types in demo data
            self.input_size = 17668 # Number of common genes and proteins between reference data and target data
            self.rna_paths = ['data_demo/adata_ref_rna.h5ad'] # GEM from reference data
            self.atac_paths = ['data_demo/adata_tar_atac.h5ad'] # GAM from target data
            self.rna_protein_paths = ['data_demo/adata_ref_adt.h5ad'] # Protein expression from reference data
            self.atac_protein_paths = ['data_demo/adata_tar_adt.h5ad'] # Protein expression from target data
            self.peak_paths = ['data_demo/adata_tar_peak.h5ad']
            
            # Training setting            
            self.batch_size = 256
            self.Ir = 0.008
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 