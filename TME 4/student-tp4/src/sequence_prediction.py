#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  12 16:07:28 2021

@author: Cécile GIANG
"""

from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from utils import *


###################################################################################################
# ------------------------------------ CHARGEMENT DES DONNEES ----------------------------------- #
###################################################################################################

# Chargement des données du métro de Hangzhou.
# Les données sont de taille D × T × S × 2 avec D le nombre de jour, T = 73 les tranches successives 
# de quart d’heure entre 5h30 et 23h30, S = 80 le nombre de stations, et les flux entrant et sortant 
# pour la dernière dimension

# Nombre de stations utilisées
CLASSES = 10

# Longueur des séquences
LENGTH = 20

# Dimension de l'entrée (1 (in) ou 2 (in/out))
INPUT_DIM = 2

# Taille du batch
BATCH_SIZE = 16

train, test = torch.load('../../data/hzdataset.pch')
train_dataset = ForecastMetroDataset(train[:, :, :CLASSES, :INPUT_DIM], length = LENGTH)
test_dataset = ForecastMetroDataset(test[:, :, :CLASSES, :INPUT_DIM], length = LENGTH, stations_max = train_dataset.stations_max)
data_train = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0)
data_test = DataLoader(test_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0)


###################################################################################################
# --------------------------------------- CHECKPOINTING ----------------------------------------- #
###################################################################################################


class State:
    """ Classe de sauvegarde sur l'apprentissage d'un modèle.
    """
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0,0


###################################################################################################
# ------------------------------------ PREDICTION DE SEQUENCE ----------------------------------- #
###################################################################################################

""" 
Notre objectif est de faire de la prédiction de séries temporelles : à partir d’une séquence de flux 
de longueur t pour l’ensemble des stations du jeu de données, on veut prédire le flux à t + 1, t + 2, ...
Nous entraînerons un RNN commun à toutes les stations qui prend une série dans R^{n×2} et prédit une série 
dans R^{n×2}.
"""

BATCH_SIZE = 16
N_EPOCHS = 50


def sequence_predictor(input_dim, latent_dim, horizon=10, n_epochs = N_EPOCHS, epsilon = 1e-1):
    """ Réseau de neurones récurrent pour la classification de séquences sur les
        données du métro de Hangzhou. Pour le décodage, comme l’objectif est de faire de la 
        classification multi-classe, on utilise une couche linéaire, suivie d’un softmax couplé 
        à un coût de cross entropie.
        @param input_dim: int, dimension de l'entrée et donc de la sortie également
        @param latent_dim: int, dimension de l'état caché
        @param length: int, longueur de chaque séquence temporelle
        @param length: int, taille de chaque batch
    """
    # Découpage de nos données en batchs de séquences de longueur length
    train_dataset = ForecastMetroDataset(train, length = horizon)
    test_dataset = ForecastMetroDataset(test, length = horizon, stations_max = train_dataset.stations_max)
    data_train = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0)
    data_test = DataLoader(test_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0)

    # Pour l'affichage aec tensorboard
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Chemin vers le modèle. Reprend l'apprentissage si un modèle est déjà sauvegardé.
    savepath = Path('classifier_lat{}_len{}.pch'.format(latent_dim, horizon))
    """
    if savepath.is_file():
        with savepath.open('rb') as file:
            state = torch.load(file)
    
    else:"""
    # Création du modèle et de l'optimiseur, chargemet sur device
    model = RNN(input_dim, latent_dim, input_dim, horizon)
    model = model.to(device)
    optim = torch.optim.SGD(params = model.parameters(), lr = epsilon) # lr : pas de gradient
    state = State(model, optim)

    # Initialisation de la mse loss
    mse = nn.MSELoss()
    
    # Initialisation de l'activation ReLU pour le décodage
    relu = nn.ReLU()
    
    # --- Phase d'apprentissage
    for epoch in range(state.epoch, N_EPOCHS):
        
        # Initialisation des loss en entraînement
        loss_list = []
        
        for x, _ in data_train:
            print(x[0].shape)
            # --- Initialisation de la liste des prédictions
            predictions = []
            
            # --- Remise à zéro des gradients des paramètres à optimiser
            state.optim.zero_grad()
            
            # --- Chargement du batch et des étiquettes correspondantes sur device
            x = x.to(device)
            
            # Initialisation des états cachés de taille (batch, latent)
            h = torch.zeros(BATCH_SIZE, latent_dim, requires_grad = True).to(device)
            
            # --- Phase forward et décodage sur l'état caché final (1ère prédiction)
            h = state.model.forward(x[:-horizon], h)[-1]
            predictions.append(relu(state.model.decode(h)))
            
            # --- Les prochaines prédictions se font sur la base de la 1ère
            for t in range(1, horizon):
                h = state.model.forward(predictions[-1], h)[-1]
                predictions.append(state.model.decode(h))
            
            # --- Phase backward
            train_loss = mse(predictions, x[-horizon:])
            train_loss.backward()
            
            # --- Sauvegarde de la loss actuelle
            loss_list.append(train_loss.item())
            
            # --- Mise à jour des paramètres
            state.optim.step()
            state.iteration += 1
            
            with savepath.open('wb') as file:
                state.epoch = epoch + 1
                torch.save(state, file)
        
        # --- Calcul de la loss en phase d'apprentissage
        train_loss = np.mean(loss_list)
        
        # --- Phase de test
            
        with torch.no_grad():
            
            loss_list = []
            
            for xtest, ytest in data_test:
                predictions = []
                xtest = xtest.to(device)
                ytest = ytest.to(device)
                h = torch.zeros(BATCH_SIZE, latent_dim, requires_grad = True).to(device)
                h = state.model.forward(xtest[:-horizon], h)[-1]
                predictions.append(state.model.decode(h))
                
                for t in range(1, horizon):
                    h = state.model.forward(predictions[-1], h)[-1]
                    predictions.append(state.model.decode(h))
                
                test_loss = mse(predictions, xtest[-horizon:])
                loss_list.append(mse(predictions, xtest[-horizon:]))
            
            test_loss = np.mean(loss_list)
            
        # --- Affichage tensorboard
            
        writer.add_scalar('Loss/train/{}/{}'.format(latent_dim, length), train_loss, epoch)
        print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
        writer.add_scalar('Loss/test/{}/{}'.format(latent_dim, length), test_loss, epoch)

sequence_predictor(2, 10, horizon=3, n_epochs = N_EPOCHS, epsilon = 1e-1)