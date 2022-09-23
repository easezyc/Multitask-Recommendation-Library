import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from ple import PLEModel
import torch
np.random.seed(1000)
categorical_field_dims=np.random.randint(0,500,30)
numerical_num=10
embed_dim=20
task_num=3
expert_num=10

batch_size=300
numerical_x=torch.rand((batch_size,numerical_num))
categorical_x=[]
for item in categorical_field_dims:
    item_batch=torch.randint(0,item,(batch_size,1))
    categorical_x.append(item_batch)
categorical_x=torch.cat(categorical_x, dim=1)
print("numerical_x_shape:{},categorical_x_shape:{}".format(numerical_x.shape,categorical_x.shape))

model=PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2),
specific_expert_num=int(expert_num / 2), dropout=0.2)

print(model)
out_put=model(categorical_x,numerical_x)
print([item.shape for item in out_put])