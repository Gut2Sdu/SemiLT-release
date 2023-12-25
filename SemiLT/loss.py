import torch
import torch.nn as nn
import numpy as np
from setting import Setting
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

def cell_cluster_dist(embedding,rna_label):
    rna_sort_label = getIdx(rna_label)
    dist = torch.zeros(rna_label.unique().size()[0],rna_label.unique().size()[0])
    dist_mean = torch.zeros(rna_label.unique().size()[0],rna_label.unique().size()[0])
    dim = 1
    for i in range(len(rna_sort_label)):
        rna_id = embedding[rna_sort_label[i],:]
        for j in range(len(rna_sort_label)):
            if i != j:
                other_id = embedding[rna_sort_label[j],:]
                dist[i,j] = torch.cdist(rna_id,other_id,p = dim).min()/rna_id.size()[1]
                dist_mean[i,j] = torch.cdist(rna_id,other_id,p = dim).mean()/rna_id.size()[1]
    return dist,dist_mean

def compute_rna_centre(rna_embedding, rna_label):
    rna_sort_label = getIdx(rna_label)
    out = []
    for i in range(len(rna_sort_label)):
        rna_id = rna_sort_label[i]
        rna_embedding_id = rna_embedding[rna_id,:]
        # out.append(rna_embedding_id)
        rna_mean = torch.mean(rna_embedding_id,dim= 0)
        out.append(rna_mean)
    rna_mean = out
    return rna_mean

def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))
    return sim

def rna_reduction(embedding, rna_label, identity_matrix):
    
    rna_sort_label = getIdx(rna_label)
    rna_mean_temp = compute_rna_centre(embedding,rna_label)
    rna_mean = torch.zeros(len(rna_mean_temp),rna_mean_temp[0].shape[0]).to(setting.device)
    for i in range(0,len(rna_mean_temp)):
        rna_mean[i,:] = rna_mean_temp[i]
    celltype_dist,celltype_dist_mean = cell_cluster_dist(embedding,rna_label)    
    rna_cluster_loss = 1 / celltype_dist_mean[celltype_dist_mean < celltype_dist_mean.mean()].mean()
    rna_common_loss = torch.Tensor([0]).to(setting.device)
    for i in range(len(rna_sort_label)):
        if rna_sort_label[i].size()[0] >1:
            rna_common_loss +=  - cosine_sim(embedding[rna_sort_label[i],:],embedding[rna_sort_label[i],:]).triu(1).mean()
    rna_common_loss = rna_common_loss/len(rna_sort_label)
    rna_other_loss = torch.cov(embedding.T).abs().mean()
    loss = rna_cluster_loss + rna_other_loss + rna_common_loss
    
    return loss


class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay
        return regularization_loss

def reduction_loss(embedding, identity_matrix, size ,k):
    loss = 1 / embedding.std(dim=0).mean()  
    loss = loss + torch.cov(embedding.T).abs().mean()
    loss = loss + torch.mean(torch.abs(embedding))
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
        rna_reduction_loss = rna_reduction(rna_embeddings[0], rna_label, self.dim)
        
        
        # atac
        atac_embedding_cat = atac_embeddings[0]
        atac_reduction_loss = reduction_loss(atac_embeddings[0], self.identity_matrix, self.dim ,rna_label.unique().size()[0])
                
        # cosine similarity loss
        rna_label=rna_labels[0]
        for i in range(1,len(rna_labels)):
            rna_label = torch.cat((rna_label,rna_labels[i]),dim=0)
                
        #cluster
        rna_mean_temp = compute_rna_centre(rna_embedding_cat,rna_label)
        rna_mean = torch.zeros(len(rna_mean_temp),rna_mean_temp[0].shape[0]).to(setting.device)
        for i in range(0,len(rna_mean_temp)):
            rna_mean[i,:] = rna_mean_temp[i]
        rna_sort_label = getIdx(rna_label)
        
        label_num = []
        for label in rna_sort_label:
            label_num.append(len(label))
        rare_label = [i for i, x in enumerate(label_num) if x < rna_embedding_cat.size()[0]*0.03] #0.03
        
        #cell
        rna_mean_sim = torch.zeros(len(rna_sort_label))
        for i in range(0,len(rna_sort_label)):
            rna_mean_sim[i] = cosine_sim(rna_embedding_cat[rna_sort_label[i]], rna_mean[i].unsqueeze(0)).min()
        
        total_k_sim = torch.topk(
            torch.max(cosine_sim(atac_embedding_cat, rna_mean), dim=1)[0],
            int(atac_embedding_cat.shape[0]))
        cos_k_maxid = torch.max(cosine_sim(atac_embedding_cat, rna_mean), dim=1)[1]

        L1_loss = 0
        count = 1
        for i in range(len(total_k_sim[0])):
            if total_k_sim[0][i] > rna_mean_sim.mean():  
                L1_loss += (atac_embedding_cat[total_k_sim[1][i],:] - rna_mean[int(cos_k_maxid[total_k_sim[1][i]]),:]).abs().mean()
        L1_loss = L1_loss/len(total_k_sim[0]) #
               
        #peak
        atac_cos = cosine_sim(peak_data,peak_data) - torch.diag(torch.diag(cosine_sim(peak_data,peak_data)))
        atac_cos2 = atac_cos
        atac_low_cos = cosine_sim(atac_embedding_cat,atac_embedding_cat) - torch.diag(torch.diag(cosine_sim(atac_embedding_cat,atac_embedding_cat)))
        k = torch.unique(rna_label, return_counts=True)[1].min() + 1
        peak_knn_graph = torch.zeros(peak_data.size()[0], peak_data.size()[0]).to(setting.device).double()
        peak_aver = 0
        for i in range(0,peak_data.size()[0]):
            peak_knn_graph[i,torch.topk(atac_cos,k)[1][i,:]] = atac_cos[i,:][torch.topk(atac_cos,k)[1][i,:]]
            peak_aver += atac_cos[i,:][torch.topk(atac_cos,k)[1][i,:]].mean()
        peak_aver = peak_aver/peak_data.size()[0]
        peak_knn_graph[(peak_knn_graph < (peak_knn_graph.sum(dim =1)/k).view(1, peak_knn_graph.size()[0]).repeat(peak_knn_graph.size()[0], 1).T) | (peak_knn_graph < torch.quantile(peak_knn_graph[peak_knn_graph > 0], 0.1))] = 0 #

        near_loss = 0
        for i in range(0,atac_embedding_cat.size()[0]):
            cell_near = torch.cat((torch.Tensor([i]).to(setting.device),torch.nonzero(peak_knn_graph[i, :] > 0).squeeze(1)),dim=0).long()
            cell_no_near = torch.nonzero(peak_knn_graph[i, :] == 0).squeeze(1).long() 
            cell_near_score = torch.cat((torch.Tensor([1]).to(setting.device),peak_knn_graph[i,:][peak_knn_graph[i, :] > 0]),dim=0)
            cell_no_near_score = atac_cos[i,:][peak_knn_graph[i, :] == 0]
            sii =  (cell_near_score * atac_low_cos[i,cell_near]).mean() 
            sij = ((1 - cell_no_near_score)/2 * atac_low_cos[i,cell_no_near]).mean()
            near_loss +=  - sii + 0.1*sij  
        near_loss = near_loss/atac_embedding_cat.size()[0]
               
        atac_rna = cosine_sim(atac_embedding_cat,rna_embedding_cat)
        rna_atac = cosine_sim(rna_embedding_cat,atac_embedding_cat)
        mnn_loss = 0
        k = torch.unique(rna_label, return_counts=True)[1].min() + 1  
        for i in range(0,atac_embedding_cat.size()[0]):
            atac_mnn_rna = torch.any(torch.topk(rna_atac,k)[1][torch.topk(atac_rna,1)[1][i,:],:] == i, dim=1).nonzero(as_tuple=False).T.squeeze()
            rna_id = torch.topk(atac_rna,1)[1][i,:][atac_mnn_rna]
            if rna_id.numel() > 1:
                mnn_loss += torch.cdist(atac_embedding_cat[i,:].unsqueeze(0),rna_embedding_cat[rna_id,:],p=2).min()
            elif rna_id.numel() == 1:
                mnn_loss += (atac_embedding_cat[i,:] - rna_embedding_cat[rna_id,:]).abs().mean() 
        mnn_loss = mnn_loss/atac_embedding_cat.size()[0]  
        celltype_dist,celltype_dist_mean = cell_cluster_dist(rna_embedding_cat,rna_label)
        
        L_RDweight = 0.25 + torch.unique(rna_label, return_counts=True)[1][rare_label].sum()/rna_label.size()[0] 
        L1_weight = 1 - len(rare_label)/len(rna_sort_label) 
        
        loss = L_RDweight*(rna_reduction_loss + atac_reduction_loss + near_loss) + mnn_loss + L1_weight*L1_loss  
        return loss


# Prediction layer loss
class CellLoss(nn.Module):
    def __init__(self):
        super(CellLoss, self).__init__()
    def forward(self, rna_cell_out, rna_cell_label):       
        predict_loss = torch.Tensor([0]).to(setting.device)
        for i in range(rna_cell_out.size()[0]):
            predict_loss += 1/(rna_cell_out[i,rna_cell_label.long()[i]] + 0.1) # #
            other_id = []
            for j in range(rna_cell_out.shape[1]):
                if j != rna_cell_label.long()[i]:
                    other_id.append(j)
            predict_loss += rna_cell_out[i,other_id][rna_cell_out[i,other_id] > rna_cell_out[i,rna_cell_label.long()[i]]].sum()             
        predict_loss = predict_loss/rna_cell_out.size()[0]
        return predict_loss
    
