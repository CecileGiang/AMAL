# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:16:35 2021

@author: Cécile GIANG
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()

# Initialisation des fonctions lineaire et mse
linear = Linear()
mse = MSE()

for n_iter in range(10000):
    # Création du contexte pour les fonctions linéaire et mse
    linear_ctx = Context()
    mse_ctx = Context()
    
    # Calcul du forward
    yhat = linear.forward(linear_ctx, x, w, b)
    loss = mse.forward(mse_ctx, yhat, y)
    
    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    # Calcul du backward (grad_w, grad_b)
    grad_yhat, grad_y = mse.backward(mse_ctx, 1)
    grad_x, grad_w, grad_b = linear.backward(linear_ctx, grad_yhat)

    # Mise à jour des paramètres du modèle
    w = w - epsilon * grad_w
    b = b - epsilon * grad_b