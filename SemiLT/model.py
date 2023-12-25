import torch
import torch.nn as nn
from setting import Setting
setting = Setting()



# Embedding layer
class Embedding_layer(nn.Module):
    def __init__(self, input_size):
        super(Embedding_layer, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, setting.embedding_size),
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        return embedding
        


# Prediction layer
class Prediction_layer(nn.Module):
    def __init__(self, num_of_class):
        super(Prediction_layer, self).__init__()
        self.cell = nn.Sequential(
            nn.Linear(setting.embedding_size, num_of_class)
        )

    def forward(self, embedding):
        cell_prediction = self.cell(embedding)
        cell_prediction = torch.softmax(cell_prediction,dim=1)
        return cell_prediction
