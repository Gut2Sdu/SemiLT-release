import torch
import torch.nn as nn
import numpy as np
from setting import Setting
import torch.nn.functional as F
setting = Setting()

def getIdx(a):
    co = a.unsqueeze(1)-a.unsqueeze(0)
    uniquer = co.unique(dim=0)
    out = []
    for r in uniquer:
        cover = torch.arange(a.size(0)).to(setting.device)
        mask = r==0
        idx = cover[mask]
        out.append(idx)
    return out

def find_id(rna_sort_label, k):
    for i, rna_sort in enumerate(rna_sort_label):
        for element in rna_sort:
            if element == k:
                return i
                
def cell_cluster_dist(embedding,rna_label):
    rna_sort_label = getIdx(rna_label)
    dist_mean = torch.zeros(rna_label.unique().size()[0],rna_label.unique().size()[0])
    dim = 1
    for i in range(len(rna_sort_label)):
        rna_id = embedding[rna_sort_label[i],:]
        for j in range(len(rna_sort_label)):
            if i != j:
                other_id = embedding[rna_sort_label[j],:]
                dist_mean[i,j] = torch.cdist(rna_id,other_id,p = dim).mean()/rna_id.size()[1]
    return dist_mean

def rna_embedding(embedding,rna_label):
    rna_sort_label = getIdx(rna_label)
    dist_mean = torch.zeros(rna_label.unique().size()[0],rna_label.unique().size()[0])
    loss = 0
    for i in range(len(rna_sort_label)):
        rna_id = embedding[rna_sort_label[i],:]
        if rna_id.size()[0]>1:
            loss = loss + rna_id.std(dim=0).mean()
    loss = loss/len(rna_sort_label)
    return loss
    
def compute_rna_centre(rna_embedding, rna_label):
    rna_sort_label = getIdx(rna_label)
    out = []
    for i in range(len(rna_sort_label)):
        rna_id = rna_sort_label[i]
        rna_embedding_id = rna_embedding[rna_id,:]
        rna_mean = torch.mean(rna_embedding_id,dim= 0)
        out.append(rna_mean)
    rna_mean = out
    return rna_mean

def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))
    return sim


class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay
        return regularization_loss

def reduction_loss(embedding, identity_matrix, size ,k, w):
    loss = ( 1 - w ) / embedding.std(dim=0).mean()  
    loss = loss + round(k/10) * (torch.cov(embedding.T).abs() - torch.diag(torch.cov(embedding.T).abs().diag()))[torch.cov(embedding.T).abs() - torch.diag(torch.cov(embedding.T).abs().diag())>0].mean() #torch.cov(embedding.T).abs().mean()
    loss = loss + (2/k) * torch.mean(torch.abs(embedding)) #0.5
    return loss    
    
def rna_reduction(embedding, rna_label, identity_matrix, rare):
    
    rna_sort_label = getIdx(rna_label)
    rna_mean_temp = compute_rna_centre(embedding,rna_label)
    rna_mean = torch.zeros(len(rna_mean_temp),rna_mean_temp[0].shape[0]).to(setting.device)
    for i in range(0,len(rna_mean_temp)):
        rna_mean[i,:] = rna_mean_temp[i]
    celltype_dist_mean = cell_cluster_dist(embedding,rna_label)    
    
    w = round(len(rna_sort_label)/10) * (rare + 0.01)
    
    rna_other_loss = round(len(rna_sort_label)/10) * (torch.cov(embedding.T).abs() - torch.diag(torch.cov(embedding.T).abs().diag()))[torch.cov(embedding.T).abs() - torch.diag(torch.cov(embedding.T).abs().diag())>0].mean() +  (2/len(rna_sort_label)) * torch.mean(torch.abs(embedding))#
    rna_embedding_loss =  - w * celltype_dist_mean.mean() +  ( 1 - w ) / embedding.std(dim=0).mean() + rna_embedding(embedding,rna_label) 
    loss = rna_embedding_loss + rna_other_loss
    return loss

    
def remove_batch(expression_batch1, expression_batch2):
    mean_batch1 = torch.mean(expression_batch1, dim=0)
    var_batch1 = torch.std(expression_batch1, dim=0)
    mean_batch2 = torch.mean(expression_batch2, dim=0)
    var_batch2 = torch.std(expression_batch2, dim=0)
    mean_diff = (mean_batch1 - mean_batch2).abs().mean()
    var_diff = (var_batch1 - var_batch2).abs().mean()
    loss = 0.01 * mean_diff + var_diff # 
    return loss

    
# Embedding layer loss
class EncodingLoss(nn.Module):
    def __init__(self, dim=64,use_gpu = True):
        super(EncodingLoss, self).__init__()
        if use_gpu:
            self.identity_matrix = torch.tensor(np.identity(dim)).float().to(setting.device)
        else:
            self.identity_matrix = torch.tensor(np.identity(dim)).float()
        self.dim = dim
        
    def forward(self, atac_embeddings, rna_embeddings,rna_labels,peak_data):
        
        rna_label=rna_labels[0]
        for i in range(1,len(rna_labels)):
            rna_label = torch.cat((rna_label,rna_labels[i]),dim=0)
            
        # rna
        rna_embedding_cat = rna_embeddings[0]
        
        rna_sort_label = getIdx(rna_label)
        label_num = []
        for label in rna_sort_label:
            label_num.append(len(label))
        rare_label = [i for i, x in enumerate(label_num) if x < rna_embedding_cat.size()[0]*0.03]
        
        rna_reduction_loss = rna_reduction(rna_embeddings[0], rna_label, self.dim, torch.unique(rna_label, return_counts=True)[1][rare_label].sum()/rna_label.size()[0] )
        
        
        # atac
        atac_embedding_cat = atac_embeddings[0]
        atac_reduction_loss = reduction_loss(atac_embeddings[0], self.identity_matrix, self.dim ,rna_label.unique().size()[0], round(len(rna_sort_label)/10) * (torch.unique(rna_label, return_counts=True)[1][rare_label].sum()/rna_label.size()[0] + 0.01))
        
        rna_label=rna_labels[0]
        for i in range(1,len(rna_labels)):
            rna_label = torch.cat((rna_label,rna_labels[i]),dim=0)
                
        rna_mean_temp = compute_rna_centre(rna_embedding_cat,rna_label)
        rna_mean = torch.zeros(len(rna_mean_temp),rna_mean_temp[0].shape[0]).to(setting.device)
        for i in range(0,len(rna_mean_temp)):
            rna_mean[i,:] = rna_mean_temp[i]
        rna_sort_label = getIdx(rna_label)
        
        label_num = []
        for label in rna_sort_label:
            label_num.append(len(label))
        rare_label = [i for i, x in enumerate(label_num) if x < rna_embedding_cat.size()[0]*0.03] #0.03
        
        rna_mean_sim = torch.zeros(len(rna_sort_label))
        for i in range(0,len(rna_sort_label)):
            rna_mean_sim[i] = cosine_sim(rna_embedding_cat[rna_sort_label[i]], rna_mean[i].unsqueeze(0)).min()
               
        #peak
        atac_cos = cosine_sim(peak_data,peak_data) - torch.diag(torch.diag(cosine_sim(peak_data,peak_data)))
        k = torch.unique(rna_label, return_counts=True)[1].min()
        peak_knn_graph = torch.zeros(peak_data.size()[0], peak_data.size()[0]).to(setting.device).double()
        for i in range(0,peak_data.size()[0]):
            peak_knn_graph[i,torch.topk(atac_cos,k)[1][i,:]] = atac_cos[i,:][torch.topk(atac_cos,k)[1][i,:]]
        peak_knn_graph[peak_knn_graph < 0.8] = 0 


        near_loss = 0
        cell_knn = []
        for i in range(0,atac_embedding_cat.size()[0]):
            cell_near = torch.cat((torch.Tensor([i]).to(setting.device),torch.nonzero(peak_knn_graph[i, :] > 0).squeeze(1)),dim=0).long()
            cell_knn.append(torch.nonzero(peak_knn_graph[i, :] > 0).squeeze(1))
            
            sii = 0
            if cell_near.numel()>1:
                sii =  atac_embedding_cat[cell_near,:].std(dim=0).mean()
            near_loss +=  sii   
        near_loss = near_loss/atac_embedding_cat.size()[0]
               
        atac_rna = cosine_sim(atac_embedding_cat,rna_embedding_cat)
        rna_atac = cosine_sim(rna_embedding_cat,atac_embedding_cat)
        mnn_loss = 0
        L_mnn = 0
        L_mnn_near = 0
        mnn_loss_near = 0
        
        count1 = 0
        count2 = 0
        atac_id = []
        atac_pre = []
        k = max(2, torch.unique(rna_label, return_counts=True)[1].min())
        for i in range(0,atac_embedding_cat.size()[0]):
            atac_mnn_rna = torch.any(torch.topk(rna_atac,k)[1][torch.topk(atac_rna,1)[1][i,:],:] == i, dim=1).nonzero(as_tuple=False).T.squeeze()
            rna_id = torch.topk(atac_rna,1)[1][i,:][atac_mnn_rna]
            if rna_id.numel() == 1:
                if atac_rna[i,rna_id] > 0.5:
                    atac_id.append(i)
                    count1 += 1
                    if setting.number_of_class == len(rna_sort_label):
                        rna_type = rna_label[rna_id].int()
                        atac_pre.append(int(rna_type))
                    else:
                        rna_type = find_id(rna_sort_label,rna_id)
                        atac_pre.append(int(rna_type))
        
        for i in range(0,atac_embedding_cat.size()[0]):
            atac_mnn_rna = torch.any(torch.topk(rna_atac,k)[1][torch.topk(atac_rna,1)[1][i,:],:] == i, dim=1).nonzero(as_tuple=False).T.squeeze()
            rna_id = torch.topk(atac_rna,1)[1][i,:][atac_mnn_rna]
            if rna_id.numel() == 1:
                if atac_rna[i,rna_id] > 0.5:
                    
                    cell_near = cell_knn[i].tolist()
                    not_in_atac_id = [x for x in cell_near if x not in atac_id]
                    
                    if setting.number_of_class == len(rna_sort_label):
                        rna_type = rna_label[rna_id].int()
                        if rna_type in rare_label:
                            rare = 0.25
                        else:
                            rare = 0
                        
                        L_mnn += (1 + rare) * (atac_embedding_cat[i,:] - rna_mean[rna_type,:]).abs().mean()
                        if len(not_in_atac_id) > 0:
                            for j in range(0,len(not_in_atac_id)):
                                if cosine_sim(atac_embedding_cat[not_in_atac_id[j],:].unsqueeze(0), rna_mean).argmax() == rna_type:
                                    w = (1 + rare) * 0.8
                                    L_mnn_near += w * (atac_embedding_cat[not_in_atac_id[j],:] - rna_mean[rna_type,:]).abs().mean()
                    else:
                        rna_type = find_id(rna_sort_label,rna_id)
                        if rna_type in rare_label:
                            rare = 0.25
                        else:
                            rare = 0
                        
                        L_mnn += (1 + rare) * (atac_embedding_cat[i,:] - rna_mean[rna_type,:]).abs().mean()
                        if len(not_in_atac_id) > 0:
                            for j in range(0,len(not_in_atac_id)):
                                if cosine_sim(atac_embedding_cat[not_in_atac_id[j],:].unsqueeze(0), rna_mean).argmax() == rna_type:
                                    w = (1 + rare) * 0.8
                                    L_mnn_near += w * (atac_embedding_cat[not_in_atac_id[j],:] - rna_mean[rna_type,:]).abs().mean()
                            
                    mnn_loss += (1 + rare) * (atac_embedding_cat[i,:] - rna_embedding_cat[rna_id,:]).abs().mean() 
                    if len(not_in_atac_id) > 0:
                        for j in range(0,len(not_in_atac_id)):
                            if cosine_sim(atac_embedding_cat[not_in_atac_id[j],:].unsqueeze(0), rna_mean).argmax() == rna_type:
                                count2 += 1
                                w = (1 + rare) * 0.8
                                mnn_loss_near += w * (atac_embedding_cat[not_in_atac_id[j],:] - rna_embedding_cat[rna_id,:]).abs().mean()
                                atac_id.append(not_in_atac_id[j])
                                atac_pre.append(int(rna_type))
        
        # print([count1,count2])                    
        mnn_loss = mnn_loss/atac_embedding_cat.size()[0] + mnn_loss_near/(atac_embedding_cat.size()[0])
        L_mnn = L_mnn/atac_embedding_cat.size()[0] + L_mnn_near/(atac_embedding_cat.size()[0])
        
        
        atac_dist_loss = 0
        if len(set(atac_pre)) > 1:
            atac_embedding_pre = atac_embedding_cat[atac_id,:]
            atac_pre_dist = cell_cluster_dist(atac_embedding_pre,torch.Tensor(atac_pre).int())
            w = round(len(rna_sort_label)/10) * (torch.unique(rna_label, return_counts=True)[1][rare_label].sum()/rna_label.size()[0] + 0.01)
            atac_dist_loss = - w * atac_pre_dist.mean()
        
        batch_loss = remove_batch(atac_embedding_cat, rna_embedding_cat)
        
        loss =  rna_reduction_loss + atac_reduction_loss + atac_dist_loss + near_loss + batch_loss + mnn_loss + 1.5 * L_mnn

        return loss


# Prediction layer loss
class CellLoss(nn.Module):
    def __init__(self):
        super(CellLoss, self).__init__()
    def forward(self, rna_cell_out, rna_cell_label):       
        predict_loss = torch.Tensor([0]).to(setting.device)
        for i in range(rna_cell_out.size()[0]):
            predict_loss += 1/(rna_cell_out[i,rna_cell_label.long()[i]] + 0.1) # - rna_cell_out[i,rna_cell_label.long()[i]] #
            other_id = []
            for j in range(rna_cell_out.shape[1]):
                if j != rna_cell_label.long()[i]:
                    other_id.append(j)
            predict_loss += rna_cell_out[i,other_id][rna_cell_out[i,other_id] > rna_cell_out[i,rna_cell_label.long()[i]]].sum()
        predict_loss = predict_loss/rna_cell_out.size()[0]
        return predict_loss
    