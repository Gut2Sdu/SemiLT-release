from sklearn.neighbors import KNeighborsClassifier
import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import math
from setting import Setting
import os
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import recall_score,precision_score


def getIdx(a):
    co = a.unsqueeze(1)-a.unsqueeze(0)
    uniquer = co.unique(dim=0)
    out = []
    for r in uniquer:
        cover = torch.arange(a.size(0))
        mask = r==0
        idx = cover[mask]
        out.append(idx)
    return out

def str_convert_to_numeric(cell_types):
    cell_type_mapping = {}
    unique_cell_types = sorted(set(cell_types), key=lambda x: (x[0].isupper(), x))
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

def compute_Kscore(top10_neighbors,rna_label_knn):
    atac_K = np.zeros(top10_neighbors.shape)
    atac_K_score = np.zeros([top10_neighbors.shape[0],len(set(rna_label_knn))])
    for i in range(top10_neighbors.shape[0]):
        for j in range(top10_neighbors.shape[1]):
            atac_K[i][j] = rna_label_knn[top10_neighbors[i][j]]
            k=int(rna_label_knn[top10_neighbors[i][j]])
            atac_K_score[i][k] +=1
    atac_K_score = atac_K_score/top10_neighbors.shape[1]
    return atac_K_score


def compute_Dscore(rna_label_knn,atac_embeddings,rna_embedding_knn):
    #CEN
    rna_sort_label = getIdx(torch.from_numpy(rna_label_knn))
    out = []
    rna_embedding = torch.from_numpy(rna_embedding_knn)
    rna_num = torch.zeros(len(rna_sort_label))

    for i in range(len(rna_sort_label)):
        rna_id = rna_sort_label[i]
        rna_embedding_id = rna_embedding[rna_id,:]
        # out.append(rna_embedding_id)
        rna_mean = torch.mean(rna_embedding_id,dim= 0)
        out.append(rna_mean)
        rna_mean = out
        rna_num[i]= len(rna_id)
        
    #atac,rna到类中心距离
    atac_Dist = torch.zeros(atac_embeddings.shape[0],len(rna_sort_label))
    rna_Distp = torch.zeros(len(rna_sort_label))
    pdist = nn.PairwiseDistance(p=2) #定义欧氏距离
    #atac距离
    for i in range(atac_embeddings.shape[0]):
        for j in range(len(rna_sort_label)):
            atac_array = torch.from_numpy(atac_embeddings[i,:])
            atac_Dist[i][j] = pdist(atac_array,rna_mean[j])
    #rna半径
    for i in range(len(rna_sort_label)):
        rna_Dist = torch.zeros(rna_sort_label[i].size()[0])
        for j in range(rna_sort_label[i].size()[0]):
            rna_array = torch.from_numpy(rna_embedding_knn[j,:])
            rna_Dist[j] = pdist(rna_array,rna_mean[i])
        rna_Distp[i] = rna_Dist.max()
        
    #计算分数
    atac_D_score = torch.zeros(atac_embeddings.shape[0],len(rna_sort_label))
    for i in range(atac_embeddings.shape[0]):
        for j in range(len(rna_sort_label)):
            atac_D_score[i][j] = math.exp(-atac_Dist[i][j])
    atac_D_score = atac_D_score.numpy()
    return atac_D_score


def Transfer(setting, neighbors = 30, knn_rna_samples = 50000):    
    # read rna embeddings and predictions    
    print('[Label transfer] Read RNA data')
    db_name = os.path.basename(setting.rna_paths[0]).split('.')[0]
    rna_embeddings = np.loadtxt('./output/' + db_name + '_embeddings.txt')
    rna_labels_str = sc.read((setting.rna_paths[0])).obs['cell_type'].tolist()  
    rna_index, rna_labels = str_convert_to_numeric(rna_labels_str)
    for i in range(1, len(setting.rna_paths)):
        db_name = os.path.basename(setting.rna_paths[i]).split('.')[0]
        rna_embeddings = np.concatenate((rna_embeddings, np.loadtxt('./output/' + db_name + '_embeddings.txt')), 0)
        rna_labels = np.concatenate((rna_labels, np.loadtxt(setting.rna_labels[i])), 0)

    
    rna_embedding_knn = []
    rna_label_knn = []
    rna_prediction_knn = []
    
    num_of_rna = rna_embeddings.shape[0]
    if num_of_rna > knn_rna_samples:
        sampling_interval = num_of_rna*1./knn_rna_samples
        subsampled_rna_embeddings = []
        subsampled_rna_labels = []
        subsampled_rna_data_prediction = []
        
        i = 0
        while i < num_of_rna:        
            rna_embedding_knn.append(rna_embeddings[i])
            rna_label_knn.append(int(rna_labels[i]))
            i = int(i + sampling_interval)
        rna_label_knn=np.array(rna_label_knn)
        rna_embedding_knn=np.array(rna_embedding_knn)
    else: 
        rna_embedding_knn = rna_embeddings
        rna_label_knn = rna_labels
        
    
    
    # read rna embeddings and predictions
    print('[Label transfer] Read ATAC data')
    db_names = []
    db_sizes = []
    db_name = os.path.basename(setting.atac_paths[0]).split('.')[0]    
    atac_embeddings = np.loadtxt('./output/' + db_name + '_embeddings.txt')
    db_names.append(db_name)
    db_sizes.append(atac_embeddings.shape[0])
    for i in range(1, len(setting.atac_paths)):
        db_name = os.path.basename(setting.atac_paths[i]).split('.')[0]        
        em = np.loadtxt('./output/' + db_name + '_embeddings.txt')
        atac_embeddings = np.concatenate((atac_embeddings, em), 0)       
        db_names.append(db_name)
        db_sizes.append(em.shape[0])
        


    # label transfer start
    print('[Label transfer] Build Space')
    neigh = KNeighborsClassifier(n_neighbors=neighbors)
    neigh.fit(rna_embedding_knn, rna_label_knn)
    
    
    top10_Dist, top10_neighbors = neigh.kneighbors(atac_embeddings, neighbors) 
    
    a = 0.2
    b = 0.8
    atac_Kscore = compute_Kscore(top10_neighbors,rna_label_knn)
    atac_Dscore = compute_Dscore(rna_label_knn,atac_embeddings,rna_embedding_knn)
    atac_score = a*atac_Kscore + b*atac_Dscore
    atac_predict = atac_score.argmax(1)   
    
    atac_predict_label = numeric_convert_to_str(rna_index, atac_predict)
    
    print('[Label transfer] finished')
    db_name = os.path.basename(setting.atac_paths[0]).split('.')[0]
    fp_pre = open('./output/' + db_name + '_predictions.txt', 'w')
    for i in range(0,atac_predict.shape[0]):
        fp_pre.write(str(atac_predict_label[i])+'\n')
    fp_pre.close()
    
    #test
    if setting.atac_labels is True:
        atac_labels_str = sc.read((setting.atac_paths[0])).obs['cell_type'].tolist()
    indices = [index for index, value in enumerate(atac_labels_str) if value in rna_labels_str]
    atac_true = [atac_labels_str[index] for index in indices]
    _,atac_true = str_convert_to_numeric(atac_true)
    atac_pre = [atac_predict[index] for index in indices]
    
    
    ari=adjusted_rand_score(atac_true, atac_pre)  
    print('ARI:%f'%(ari))
    recall = recall_score(atac_true, atac_pre,average="weighted")
    print('Recall：%f'%(recall))
    precision = precision_score(atac_true, atac_pre, average="weighted")
    print('Precision:%f'%(precision))
    f1 = f1_score(atac_true, atac_pre, average="weighted")
    print('F1-score：%f'%(f1))

    
if __name__ == "__main__":
    setting = Setting()
    Transfer(setting)