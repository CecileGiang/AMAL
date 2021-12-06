# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:16:35 2021

@author: Cécile GIANG
"""

import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
print('Gradcheck for MSE function : ', torch.autograd.gradcheck(mse, (yhat, y)))

# Test du gradient de Linear

X = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
W = torch.randn(5,7, requires_grad=True, dtype=torch.float64)
b = torch.randn(1,7, requires_grad=True, dtype=torch.float64)
B = torch.stack(tensors =[b[0],b[0],b[0],b[0],b[0],b[0],b[0],b[0],b[0],b[0]], axis = 0)

print('Gradcheck for Linear function : ', torch.autograd.gradcheck(linear, (X,W,B)))