import torch
import os

class Setting(object):
    def __init__(self):
        DB = 'ms' #MCA_subset #CITE_ASAP #ms #PBMC #brain 
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
            
        if DB == "BMMC_1":
            self.number_of_class = 22 # Number of cell types in scRNA-seq
            self.input_size = 11362
            self.rna_paths = ['data_site1/rna_1_filter.h5ad'] # scRNA-seq data
            self.atac_paths = ['data_site1/atac_1_filter.h5ad'] # scATAC-seq data
            self.rna_protein_paths = [] # Protein expression from reference data
            self.atac_protein_paths = [] # Protein expression from target data
            self.peak_paths = ['data_site1/peak_1_PCA50.h5ad'] #peak_1_tfidf
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
            
        if DB == "BMMC_2":
            self.number_of_class = 20 
            self.input_size = 11407
            self.rna_paths = ['data_site2/rna_2_filter.h5ad']
            self.atac_paths = ['data_site2/atac_2_filter.h5ad']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.peak_paths = ['data_site2/peak_2_PCA50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
            
        if DB == "BMMC_3":
            self.number_of_class = 21 
            self.input_size = 11428 
            self.rna_paths = ['data_site3/rna_3_filter.h5ad'] 
            self.atac_paths = ['data_site3/atac_3_filter.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['data_site3/peak_3_PCA50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008 
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
            
        if DB == "BMMC_4":
            self.number_of_class = 20
            self.input_size = 11450
            self.rna_paths = ['data_site4/rna_4_filter.h5ad']
            self.atac_paths = ['data_site4/atac_4_filter.h5ad']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.peak_paths = ['data_site4/peak_4_PCA50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
        
        if DB == "CITE_ASAP":
            self.number_of_class = 7 
            self.input_size = 17219 # Number of common genes and proteins between reference data and target data
            self.rna_paths = ['data_demo/adata_ref_rna.h5ad']
            self.atac_paths = ['data_demo/adata_tar_atac.h5ad']
            self.rna_protein_paths = ['data_demo/adata_ref_adt.h5ad']
            self.atac_protein_paths = ['data_demo/adata_tar_adt.h5ad']
            self.peak_paths = ['data_demo/adata_tar_peak.h5ad'] #adt
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
        if DB == "MCA_subset":
            self.number_of_class = 14 
            self.input_size = 17057 
            self.rna_paths = ['data_MCA/adata_mca_gem.h5ad']
            self.atac_paths = ['data_MCA/adata_mca_gam.h5ad']
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['data_MCA/adata_mca_peak.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256 
            self.lr = 0.008
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
        if DB == "ms":
            self.number_of_class = 10 
            self.input_size = 11055 
            self.rna_paths = ['data_ms/ms_rna.h5ad']
            self.atac_paths = ['data_ms/ms_atac.h5ad']
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['data_ms/ms_PCA50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008 
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
            
        # if DB == "brain":
        #     self.number_of_class = 23
        #     self.input_size = 17403
        #     self.rna_paths = ['data_brain/brain_rna.h5ad']
        #     self.atac_paths = ['data_brain/brain_atac.h5ad']
        #     self.rna_protein_paths = [] 
        #     self.atac_protein_paths = [] 
        #     self.peak_paths = ['data_brain/brain_PCA50.h5ad']
        #     self.atac_labels = False
            
        #     # Training setting            
        #     self.batch_size = 256
        #     self.lr = 0.008
        #     self.lr_decay_epoch = 20
        #     self.epochs = 20
        #     self.embedding_size = 64
        #     self.momentum = 0.9
        #     self.seed = 1
        #     self.checkpoint = '' 
            
        # if DB == "hf":
        #     self.number_of_class = 15
        #     self.input_size = 17187
        #     self.rna_paths = ['data_hf/hf_rna.h5ad'] 
        #     self.atac_paths = ['data_hf/hf_atac_data_1k.h5ad'] 
        #     self.rna_protein_paths = [] 
        #     self.atac_protein_paths = [] 
        #     self.peak_paths = ['data_hf/hf_PCA50.h5ad']
        #     self.atac_labels = False
            
        #     # Training setting            
        #     self.batch_size = 256
        #     self.lr = 0.008
        #     self.lr_decay_epoch = 20
        #     self.epochs = 20
        #     self.embedding_size = 64
        #     self.momentum = 0.9
        #     self.seed = 1
        #     self.checkpoint = ''
            
        if DB == "hf2":
            self.number_of_class = 8
            self.input_size = 11387
            self.rna_paths = ['data_hf/hf2/hf1_data/hf_rna_1k.h5ad'] 
            self.atac_paths = ['data_hf/hf2/hf1_data/hf_atac_1k.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['data_hf/hf2/hf1_data/hf_atac_1k_PCA50.h5ad'] #
            self.atac_labels = False
            
            # Training setting            
            self.batch_size = 300
            self.lr = 0.008
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = '' 
            
        if DB == "PBMC":
            self.number_of_class = 14
            self.input_size = 13786
            self.rna_paths = ['data_PBMC/pbmc_rna.h5ad'] 
            self.atac_paths = ['data_PBMC/pbmc_atac.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['data_PBMC/pbmc_PCA50.h5ad']
            self.atac_labels = False
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 20
            self.epochs = 20
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = '' 