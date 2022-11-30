import torch
from layers import EmbeddingLayer, MultiLayerPerceptron


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

        self.task_experts=[[0] * self.task_num for _ in range(self.layers_num)]
        self.task_gates=[[0] * self.task_num for _ in range(self.layers_num)]
        self.share_experts=[0] * self.layers_num
        self.share_gates=[0] * self.layers_num
        for i in range(self.layers_num):
            input_dim = self.embed_output_dim if 0 == i else bottom_mlp_dims[i - 1]
            self.share_experts[i] = torch.nn.ModuleList([MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False) for k in range(self.shared_expert_num)])
            self.share_gates[i]=torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + task_num * specific_expert_num), torch.nn.Softmax(dim=1))
            for j in range(task_num):
                self.task_experts[i][j]=torch.nn.ModuleList([MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False) for k in range(self.specific_expert_num)])
                self.task_gates[i][j]=torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + specific_expert_num), torch.nn.Softmax(dim=1))
            self.task_experts[i]=torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = torch.nn.ModuleList(self.task_gates[i])

        self.task_experts = torch.nn.ModuleList(self.task_experts)
        self.task_gates = torch.nn.ModuleList(self.task_gates)
        self.share_experts = torch.nn.ModuleList(self.share_experts)
        self.share_gates = torch.nn.ModuleList(self.share_gates)


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

        task_fea = [emb for i in range(self.task_num + 1)] # task1 input ,task2 input,..taskn input, share_expert input
        for i in range(self.layers_num):
            share_output=[expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]]
            task_output_list=[]
            for j in range(self.task_num):
                task_output=[expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_ouput=torch.cat(task_output+share_output,dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)
            if i != self.layers_num-1:#最后一层不需要计算share expert 的输出
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_ouput = torch.cat(task_output_list + share_output, dim=1)
                task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results