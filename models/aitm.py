import torch
import numpy as np
from .layers import EmbeddingLayer, MultiLayerPerceptron


class AITMModel(torch.nn.Module):
    """
    A pytorch implementation of Adaptive Information Transfer Multi-task Model.

    Reference:
        Xi, Dongbo, et al. Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising. KDD 2021.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.hidden_dim = bottom_mlp_dims[-1]

        self.g = torch.nn.ModuleList([torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1]) for i in range(task_num - 1)])
        self.h1 = torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.h2 = torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.h3 = torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1])

        self.bottom = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(task_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        fea = [self.bottom[i](emb) for i in range(self.task_num)]

        for i in range(1, self.task_num):
            p = self.g[i - 1](fea[i - 1]).unsqueeze(1)
            q = fea[i].unsqueeze(1)
            x = torch.cat([p, q], dim = 1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            fea[i] = torch.sum(torch.nn.functional.softmax(torch.sum(K * Q, 2, True) / np.sqrt(self.hidden_dim), dim=1) * V, 1)

        results = [torch.sigmoid(self.tower[i](fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results