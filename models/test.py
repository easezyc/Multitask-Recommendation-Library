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
# model=PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
# tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2),
# specific_expert_num=int(expert_num / 2), dropout=0.2)
batch_size=30
numerical_x=torch.rand((batch_size,numerical_num))
categorical_x=[]
for item in categorical_field_dims:
    item_batch=torch.randint(0,item,(batch_size,1,1))
    categorical_x.append(item_batch)
    print(item_batch)
# categorical_x=torch
# model()