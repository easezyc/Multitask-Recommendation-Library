import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from ple import PLEModel
categorical_field_dims=np.random.randint(0,500,30)
numerical_num=10
embed_dim=20
task_num=2
expert_num=10
model=PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2),
specific_expert_num=int(expert_num / 2), dropout=0.2)
print(model)