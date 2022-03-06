import torch
import numpy as np

class Meta_Linear(torch.nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Meta_Linear, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = torch.nn.functional.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Meta_Linear, self).forward(x)
        return out

class Meta_Embedding(torch.nn.Embedding): #used in MAML to forward input with fast weight
    def __init__(self, num_embedding, embedding_dim):
        super(Meta_Embedding, self).__init__(num_embedding, embedding_dim)
        self.weight.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            out = torch.nn.functional.embedding(
            x, self.weight.fast, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            out = torch.nn.functional.embedding(
            x, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class HeacModel(torch.nn.Module):
    """
    A pytorch implementation of Hybrid Expert and Critic Model.

    Reference:
        Zhu, Yongchun, et al. Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising. KDD 2021.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, critic_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_embedding = Meta_Embedding(task_num, embed_dim)
        self.task_num = task_num
        self.expert_num = expert_num
        self.critic_num = critic_num

        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.critic = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(critic_num)])
        self.expert_gate = torch.nn.Sequential(torch.nn.Linear(embed_dim * 2, expert_num), torch.nn.Softmax(dim=1))
        self.critic_gate = torch.nn.Sequential(torch.nn.Linear(embed_dim * 2, critic_num), torch.nn.Softmax(dim=1))

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1)
        batch_size = emb.size(0)

        gate_input_emb = []
        for i in range(self.task_num):
            idxs = torch.tensor([i for j in range(batch_size)]).view(-1, 1).cuda()
            task_emb = self.task_embedding(idxs).squeeze(1)
            gate_input_emb.append(torch.cat([task_emb, torch.mean(emb, dim=1)], dim=1).view(batch_size, -1))
        
        emb = emb.view(-1, self.embed_output_dim)

        expert_gate_value = [self.expert_gate(gate_input_emb[i]).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        task_fea = [torch.bmm(expert_gate_value[i], fea).squeeze(1) for i in range(self.task_num)]

        critic_gate_value = [self.critic_gate(gate_input_emb[i]) for i in range(self.task_num)]
        results = []
        for i in range(self.task_num):
            output = [torch.sigmoid(self.critic[j](task_fea[i])) for j in range(self.critic_num)]
            output = torch.cat(output, dim=1)
            results.append(torch.mean(critic_gate_value[i] * output, dim=1))

        return results

class MetaHeacModel(torch.nn.Module):
    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, critic_num, dropout):
        super(MetaHeacModel, self).__init__()
        self.model = HeacModel(categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, critic_num, dropout)
        self.local_lr = 0.0002
        self.criterion = torch.nn.BCELoss()

    def forward(self, categorical_x, numerical_x):
        return self.model(categorical_x, numerical_x)

    def local_update(self, support_set_categorical, support_set_numerical, support_set_y):
        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None
        support_set_y_pred = self.model(support_set_categorical, support_set_numerical)
        loss_list = [self.criterion(support_set_y_pred[j], support_set_y[:, j].float()) for j in range(support_set_y.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)

        self.model.zero_grad()
        grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)
        fast_parameters = []
        for k, weight in enumerate(self.model.parameters()):
            if grad[k] is None:
                continue
            # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
            if weight.fast is None:
                weight.fast = weight - self.local_lr * grad[k]  # create weight.fast
            else:
                weight.fast = weight.fast - self.local_lr * grad[k]
            fast_parameters.append(weight.fast)

        return loss

    def global_update(self, list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y):
        batch_sz = len(list_sup_categorical)
        losses_q = []
        for i in range(batch_sz):
            loss_sup = self.local_update(list_sup_categorical[i], list_sup_numerical[i], list_sup_y[i])
            query_set_y_pred = self.model(list_qry_categorical[i], list_qry_numerical[i])

            loss_list = [self.criterion(query_set_y_pred[j], list_qry_y[i][:, j].float()) for j in range(list_qry_y[i].size(1))]
            loss = 0
            for item in loss_list:
                loss += item
            loss /= len(loss_list)

            losses_q.append(loss)
        losses_q = torch.stack(losses_q).mean(0)
        return losses_q