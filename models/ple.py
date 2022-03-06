import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron


class PLEModel(torch.nn.Module):
    """
    A pytorch implementation of PLE Model.

    Reference:
        Tang, Hongyan, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. RecSys 2020.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, shared_expert_num, specific_expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)

        self.expert = list()
        for i in range(self.layers_num):
            if i == 0:
                self.expert.append(torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, [bottom_mlp_dims[i]], dropout, output_layer=False) for j in range(self.specific_expert_num * self.task_num + self.shared_expert_num)]))
            else:
                self.expert.append(torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[i - 1], [bottom_mlp_dims[i]], dropout, output_layer=False) for j in range(self.specific_expert_num * self.task_num + self.shared_expert_num)]))
        self.expert = torch.nn.ModuleList(self.expert)

        self.gate = list()
        for i in range(self.layers_num):
            if i == 0:
                input_dim = self.embed_output_dim
            else:
                input_dim = bottom_mlp_dims[i - 1]
            gate_list = [torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + specific_expert_num), torch.nn.Softmax(dim=1)) for j in range(self.task_num)]
            gate_list.append(torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + task_num * specific_expert_num), torch.nn.Softmax(dim=1)))
            self.gate.append(torch.nn.ModuleList(gate_list))
        self.gate = torch.nn.ModuleList(self.gate)

        self.task_expert_index = list()
        for i in range(task_num):
            index_list = list()
            index_list.extend(range(i * self.specific_expert_num, (1 + i) * self.specific_expert_num))
            index_list.extend(range(task_num * self.specific_expert_num, task_num * self.specific_expert_num + self.shared_expert_num))
            self.task_expert_index.append(index_list)
        self.task_expert_index.append(range(task_num * self.specific_expert_num + self.shared_expert_num))

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

        task_fea = [emb for i in range(self.task_num + 1)]
        for i in range(self.layers_num):
            for j in range(self.task_num + 1):
                fea = torch.cat([self.expert[i][index](task_fea[j]).unsqueeze(1) for index in self.task_expert_index[j]], dim = 1)
                gate_value = self.gate[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, fea).squeeze(1)
        
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results