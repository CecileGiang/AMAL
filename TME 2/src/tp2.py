#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 08:50:28 2021

@author: Cécile GIANG
"""

# NOTE: Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import datamaestro
from tqdm import tqdm
import random as rd

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float, requires_grad=True)
datay = torch.tensor(datay,dtype=torch.float, requires_grad=True).reshape(-1,1)

# Découpage des données en ensembles train (80%) et test (20%)

xtrain, xtest, ytrain, ytest = train_test_split(datax, datay, test_size = 0.2)


###################################################################################################
# ---------------------------- DIFFERENCIATION AUTOMATIQUE: AUTOGRAD ---------------------------- #
###################################################################################################


# Définition du module Linear

class Linear(torch.nn.Module):
    def __init__(self, x_size, y_size):
        """ @param x_size: int, size of each data sample
            @param y_size: int, size of each label sample
        """
        super(Linear, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(x_size, y_size)) # par défaut, requires_grad = True
        self.b = torch.nn.Parameter(torch.randn(y_size))

    def forward(self, x):
        """ @param x: torch.tensor, samples
        """
        return x @ self.w + self.b


# Descente de gradient batch

N_ITERS = 100000

def linear_regression(xtrain, xtest, ytrain, ytest, strat = 'batch', size = 10, epsilon = 1e-6):
    """ @param xtrain: torch.tensor, training samples
        @param xtest: torch.tensor, test samples
        @param ytrain: torch.tensor, training labels
        @param ytest: torch.tensor, test labels
        @param strat: str, 'batch', 'stochastic' or 'mini-batch'
        @param size: int, number of samples to consider for each epoch (if strat == 'mini-batch')
    """
    # Pour l'affichage sous tensorboard
    writer = SummaryWriter()
    
    # Initialisation des fonctions linéaire et mse
    linear = Linear(xtrain.shape[1], ytrain.shape[1])
    mse = torch.nn.MSELoss()

    # Initialisation de la loss
    train_loss = mse(linear.forward(xtrain), ytrain)

    for i_iter in range(N_ITERS):

        # --- Choix de la stratégie

        if strat == 'stochastic':
            index = rd.randint(0, xtrain.shape[0] - 1)
            datax = xtrain[index]
            datay = ytrain[index]
        elif strat == 'mini-batch':
            index = rd.sample(range(xtrain.shape[0]), size)
            datax = xtrain[index]
            datay = ytrain[index]
        else:
            datax = xtrain
            datay = ytrain

        # --- Phase forward

        yhat = linear.forward(datax)
        train_loss = mse(yhat, datay)

        writer.add_scalar('Loss/train/{}'.format(strat), train_loss, i_iter)
        print('Iter {} | Training loss: {}' . format(i_iter, train_loss))

        # `loss` doit correspondre au coût MSE calculé à cette itération
            # on peut visualiser avec
            # tensorboard --logdir runs/


        # --- Phase backward

        train_loss.backward(retain_graph=True)

        # --- Mise à jour des paramètres, remise à zéro des gradients

        with torch.no_grad():
            linear.w -= epsilon * linear.w.grad
            linear.b -= epsilon * linear.b.grad
            linear.w.grad.zero_()
            linear.b.grad.zero_()

        # --- Phase forward

        yhat = linear.forward(xtest)
        test_loss = mse(yhat, ytest)

        writer.add_scalar('Loss/test/{}'.format(strat), test_loss, i_iter)
        #print('Epoch {} | Testing loss: {}' . format(epoch, test_loss))

## Tests
# linear_regression(xtrain, xtest, ytrain, ytest, 'batch')
# linear_regression(xtrain, xtest, ytrain, ytest, 'stochastic')
# linear_regression(xtrain, xtest, ytrain, ytest, 'mini-batch')


###################################################################################################
# ----------------------------------------- OPTIMISEUR ------------------------------------------ #
###################################################################################################


def linear_regression_optim(xtrain, xtest, ytrain, ytest, strat = 'batch', size = 10, epsilon = 1e-6):
    """ @param xtrain: torch.tensor, training samples
        @param xtest: torch.tensor, test samples
        @param ytrain: torch.tensor, training labels
        @param ytest: torch.tensor, test labels
        @param strat: str, 'batch', 'stochastic' or 'mini-batch'
        @param size: int, number of samples to consider for each epoch (if strat == 'mini-batch')
    """
    # Pour l'affichage sous tensorboard
    writer = SummaryWriter()
    
    # Initialisation des fonctions linéaire et mse
    linear = Linear(xtrain.shape[1], ytrain.shape[1])
    mse = torch.nn.MSELoss()

    # Initialisation de la loss
    train_loss = mse(linear.forward(xtrain), ytrain)
    optim = torch.optim.SGD(params=[linear.w,linear.b],lr=epsilon) ## on optimise selon w et b, lr : pas de gradient
    optim.zero_grad()

    for i_iter in range(N_ITERS):

        # --- Choix de la stratégie

        if strat == 'stochastic':
            index = rd.randint(0, xtrain.shape[0] - 1)
            datax = xtrain[index]
            datay = ytrain[index]
        elif strat == 'mini-batch':
            index = rd.sample(range(xtrain.shape[0]), size)
            datax = xtrain[index]
            datay = ytrain[index]
        else:
            datax = xtrain
            datay = ytrain

        # --- Phase forward

        yhat = linear.forward(datax)
        train_loss = mse(yhat, datay)

        writer.add_scalar('Loss/train/{}'.format(strat), train_loss, i_iter)
        print('Iter {} | Training loss: {}' . format(i_iter, train_loss))

        # `loss` doit correspondre au coût MSE calculé à cette itération
            # on peut visualiser avec
            # tensorboard --logdir runs/


        # --- Phase backward

        train_loss.backward(retain_graph=True)

        # --- Mise à jour des paramètres, remise à zéro des gradients

        optim.step() # Mise-à-jour des paramètres w et b
        optim.zero_grad() # Reinitialisation du gradient

        # --- Phase forward

        yhat = linear.forward(xtest)
        test_loss = mse(yhat, ytest)

        writer.add_scalar('Loss/test/{}'.format(strat), test_loss, i_iter)
        #print('Epoch {} | Testing loss: {}' . format(epoch, test_loss))

## Tests
# linear_regression_optim(xtrain, xtest, ytrain, ytest, 'batch')
# linear_regression_optim(xtrain, xtest, ytrain, ytest, 'stochastic')
# linear_regression_optim(xtrain, xtest, ytrain, ytest, 'mini-batch')



###################################################################################################
# ------------------------------ RESEAU A 2 COUCHES NON SEQUENTIEL ------------------------------ #
###################################################################################################


N_ITERS = 30000

def nn_module(xtrain, xtest, ytrain, ytest, strat = 'batch', size = 10, epsilon = 1e-4):
    """ @param xtrain: torch.tensor, training samples
        @param xtest: torch.tensor, test samples
        @param ytrain: torch.tensor, training labels
        @param ytest: torch.tensor, test labels
        @param strat: str, 'batch', 'stochastic' or 'mini-batch'
        @param size: int, number of samples to consider for each epoch (if strat == 'mini-batch')
    """
    # Pour l'affichage sous tensorboard
    writer = SummaryWriter()
    
    # Initialisation des paramètres w et b pour les deux couches linéaires
    w1 = torch.nn.Parameter(torch.randn(xtrain.shape[1],ytrain.shape[1]), requires_grad=True)
    b1 = torch.nn.Parameter(torch.randn(ytrain.shape[1]),requires_grad=True)
    w2 = torch.nn.Parameter(torch.randn(ytrain.shape[1],ytrain.shape[1]), requires_grad=True)
    b2 = torch.nn.Parameter(torch.randn(ytrain.shape[1]),requires_grad=True)
    
    # Initialisation de l'optimiseur
    optim = torch.optim.SGD(params=[w1, b1, w2, b2],lr=epsilon) ## on optimise selon w et b, lr : pas de gradient
    optim.zero_grad()

    # Initialisation de l'activation et de la loss
    tanh = torch.nn.Tanh()
    mse = torch.nn.MSELoss()

    # Descente de gradient
    for i_iter in range(N_ITERS):

	# --- Choix de la stratégie

        if strat == 'stochastic':
            index = rd.randint(0, xtrain.shape[0] - 1)
            datax = xtrain[index]
            datay = ytrain[index]
        elif strat == 'mini-batch':
            index = rd.sample(range(xtrain.shape[0]), size)
            datax = xtrain[index]
            datay = ytrain[index]
        else:
            datax = xtrain
            datay = ytrain
        
	# --- Phase forward

        tmp = tanh( torch.nn.functional.linear(datax, weight = w1.t(), bias = b1) )
        train_loss = mse( torch.nn.functional.linear(tmp, weight = w2.t(), bias = b2), datay)
        
	# --- Phase backward

        train_loss.backward(retain_graph=True)
        writer.add_scalar('Loss/train/{}'.format(strat), train_loss, i_iter)
        print('Iter {} | Training loss: {}' . format(i_iter, train_loss))

	# --- Mise à jour des paramètres, remise à zéro des gradients

        optim.step()
        optim.zero_grad()

	# --- Phase forward pour les données test

        with torch.no_grad():

            tmp = tanh( torch.nn.functional.linear(xtest, weight = w1.t(), bias = b1) )
            test_loss = mse( torch.nn.functional.linear(tmp, weight = w2.t(), bias = b2), ytest)
            writer.add_scalar('Loss/test/{}'.format(strat), test_loss, i_iter)

## Tests
# nn_module(xtrain, xtest, ytrain, ytest, 'batch')
# nn_module(xtrain, xtest, ytrain, ytest, 'stochastic')
# nn_module(xtrain, xtest, ytrain, ytest, 'mini-batch')


###################################################################################################
# -------------------------------- RESEAU A 2 COUCHES SEQUENTIEL -------------------------------- #
###################################################################################################


N_ITERS = 30000

def nn_sequentiel(xtrain, xtest, ytrain, ytest, strat = 'batch', size = 10, epsilon = 1e-4):
    """ @param xtrain: torch.tensor, training samples
        @param xtest: torch.tensor, test samples
        @param ytrain: torch.tensor, training labels
        @param ytest: torch.tensor, test labels
        @param strat: str, 'batch', 'stochastic' or 'mini-batch'
        @param size: int, number of samples to consider for each epoch (if strat == 'mini-batch')
    """
    # Pour l'affichage sous tensorboard
    writer = SummaryWriter()
    
    # Initialisation du module Sequential
    model = torch.nn.Sequential(
                torch.nn.Linear(xtrain.shape[1],ytrain.shape[1]),
                torch.nn.Tanh(),
                torch.nn.Linear(ytrain.shape[1],ytrain.shape[1])
            )
    
    # Initialisation de l'optimiseur
    optim = torch.optim.SGD(params=model.parameters(),lr=epsilon) ## on optimise selon w et b, lr : pas de gradient
    optim.zero_grad()

    # Initialisation de l'activation et de la loss
    mse = torch.nn.MSELoss()

    # Descente de gradient
    for i_iter in range(N_ITERS):

	# --- Choix de la stratégie

        if strat == 'stochastic':
            index = rd.randint(0, xtrain.shape[0] - 1)
            datax = xtrain[index]
            datay = ytrain[index]
        elif strat == 'mini-batch':
            index = rd.sample(range(xtrain.shape[0]), size)
            datax = xtrain[index]
            datay = ytrain[index]
        else:
            datax = xtrain
            datay = ytrain
        
	# --- Phase forward

        yhat = model(datax)
        train_loss = mse(yhat, datay)
        
	# --- Phase backward

        train_loss.backward(retain_graph=True)
        writer.add_scalar('Loss/train/{}'.format(strat), train_loss, i_iter)
        print('Iter {} | Training loss: {}' . format(i_iter, train_loss))

	# --- Mise à jour des paramètres, remise à zéro des gradients

        optim.step()
        optim.zero_grad()

	# --- Phase forward pour les données test

        with torch.no_grad():

            yhat = model(xtest)
            test_loss = mse(yhat, ytest)
            writer.add_scalar('Loss/test/{}'.format(strat), test_loss, i_iter)

## Tests
# nn_sequentiel(xtrain, xtest, ytrain, ytest, 'batch')
# nn_sequentiel(xtrain, xtest, ytrain, ytest, 'stochastique')
# nn_sequentiel(xtrain, xtest, ytrain, ytest, 'mini-batch')