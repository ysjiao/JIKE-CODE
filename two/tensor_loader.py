# -*- coding: UTF-8 -*-
"""
================================
        @Author : ysjiao
    @Data   : 2020/11/26 14:36
=================================
"""
import numpy as np
import torch
from torch import tensor

t1 = tensor(np.array([[1, 2, 3], [4, 5, 6]]))
assert t1.dtype == torch.int32
t2 = tensor([[1, 2, 3], [4, 5, 6]])
assert t2.dtype == torch.int64

device = "gpu" if torch.cuda.is_available() else "cpu"
t3 = tensor(t1, dtype=torch.float64, device="cpu")
tensor(t1, dtype=torch.float64)
print(t1)