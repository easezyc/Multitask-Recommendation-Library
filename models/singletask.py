import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron


class SingleTaskModel(torch.nn.Module):
    """
    A pytorch implementation of Single Task Model.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super().__init__()
        self.embedding = torch.nn.ModuleList([EmbeddingLayer(categorical_field_dims, embed_dim) for i in range(task_num)])
        self.numerical_layer = torch.nn.ModuleList([torch.nn.Linear(numerical_num, embed_dim) for i in range(task_num)])
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num

        self.bottom = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(task_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        results = list()
        for i in range(self.task_num):
            categorical_emb = self.embedding[i](categorical_x)
            numerical_emb = self.numerical_layer[i](numerical_x).unsqueeze(1)
            emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
            fea = self.bottom[i](emb)
            results.append(torch.sigmoid(self.tower[i](fea).squeeze(1)))
        return results