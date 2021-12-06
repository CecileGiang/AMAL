# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:16:35 2021

@author: Cécile GIANG
"""

# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """ Un objet contexte très simplifié pour simuler PyTorch.
        Un contexte différent doit être utilisé à chaque forward.
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """ Début d'implementation de la fonction MSE.
    """
    @staticmethod
    def forward(ctx, yhat, y):
        # Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        # Renvoyer la valeur de la fonction
        return torch.mean( torch.pow( yhat-y, 2) )

    @staticmethod
    def backward(ctx, grad_output):
        # Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        
        # Renvoyer les deux dérivées partielles (par rapport à yhat et à y)
        grad_yhat = 2 * ( yhat - y ) * grad_output / yhat.nelement()
        grad_y = - 2 * ( yhat - y ) * grad_output / y.nelement()
        
        return grad_yhat, grad_y
        
class Linear(Function):
    """ Début d'implementation de la fonction linéaire.
    """
    @staticmethod
    def forward(ctx, x, w, b):
        # Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(x, w, b)

        # Renvoyer la valeur de la fonction
        return x @ w + b

    @staticmethod
    def backward(ctx, grad_output):
        # Calcul du gradient du module par rapport a chaque groupe d'entrées
        x, w, b = ctx.saved_tensors
        
        # Renvoyer les trois dérivées partielles (par rapport à x, w et b)
        # Précision énoncé: le nombre de sorties doit être égal au nombre de inputs de forward
        grad_x = grad_output @ w.t()        
        grad_w = x.t() @ grad_output
        grad_b = grad_output
        
        return grad_x, grad_w, grad_b
        


mse = MSE.apply
linear = Linear.apply