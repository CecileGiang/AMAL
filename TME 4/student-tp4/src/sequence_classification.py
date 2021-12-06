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
train_dataset = SampleMetroDataset(train[:, :, :CLASSES, :INPUT_DIM], length = LENGTH)
test_dataset = SampleMetroDataset(test[:, :, :CLASSES, :INPUT_DIM], length = LENGTH, stations_max = train_dataset.stations_max)
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
# ---------------------------------- CLASSIFICATION DE SEQUENCE --------------------------------- #
###################################################################################################

""" 
Notre objectif est de contruire un modèle qui à partir d'une séquence d'une certaine longuer infère
la station à laquelle appartient la séquence.
"""

BATCH_SIZE = 16
N_EPOCHS = 100


def sequence_classifier(input_dim, latent_dim, output_dim, length=10, n_epochs = N_EPOCHS, epsilon = 1e-1):
    """ Réseau de neurones récurrent pour la classification de séquences sur les
        données du métro de Hangzhou. Pour le décodage, comme l’objectif est de faire de la 
        classification multi-classe, on utilise une couche linéaire, suivie d’un softmax couplé 
        à un coût de cross entropie.
        @param input_dim: int, dimension de l'entrée
        @param latent_dim: int, dimension de l'état caché
        @param output_dim: int, dimension de la sortie
        @param length: int, longueur de chaque séquence temporelle
        @param length: int, taille de chaque batch
    """
    # Découpage de nos données en batchs de séquences de longueur length
    train_dataset = SampleMetroDataset(train, length = length)
    test_dataset = SampleMetroDataset(test, length = length, stations_max = train_dataset.stations_max)
    data_train = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0)
    data_test = DataLoader(test_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0)

    # Pour l'affichage aec tensorboard
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Chemin vers le modèle. Reprend l'apprentissage si un modèle est déjà sauvegardé.
    savepath = Path('classifier_lat{}_len{}.pch'.format(latent_dim, length))
    """
    if savepath.is_file():
        with savepath.open('rb') as file:
            state = torch.load(file)
    
    else:
    """
    # Création du modèle et de l'optimiseur, chargemet sur device
    model = RNN(input_dim, latent_dim, output_dim, length)
    model = model.to(device)
    optim = torch.optim.SGD(params = model.parameters(), lr = epsilon) # lr : pas de gradient
    state = State(model, optim)

    # Initialisation de la loss cross entropique
    cross_entropy = nn.CrossEntropyLoss()
    
    # --- Phase d'apprentissage
    for epoch in range(state.epoch, N_EPOCHS):
        
        # Initialisation des loss en entraînement
        loss_list = []
        
        for x, y in data_train:
            print(x.shape)
            # --- Remise à zéro des gradients des paramètres à optimiser
            state.optim.zero_grad()
            
            # --- Chargement du batch et des étiquettes correspondantes sur device
            x = x.to(device)
            y = y.to(device)
            
            # --- Initialisation des états cachés de taille (batch, latent)
            h = torch.zeros(BATCH_SIZE, latent_dim, requires_grad = True).to(device)
            
            # --- Phase forward
            h = state.model.forward(x, h)[-1]
            
            # --- Décodage des états cachés finaux pour trouver le y d'intérêt
            yhat = state.model.decode(h)
            
            # --- Phase backward
            train_loss = cross_entropy(yhat, y)
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
        softmax = torch.nn.Softmax()
        
        with torch.no_grad():
            
            loss_list = []
            correct_preds = 0
            total_preds = 0
            
            for xtest, ytest in data_test:
                
                xtest = xtest.to(device)
                ytest = ytest.to(device)
                h = torch.zeros(BATCH_SIZE, latent_dim, requires_grad = True).to(device)
                htest = state.model.forward(xtest, h)[-1]
                yhat_test = state.model.decode(htest)
                loss_list.append(cross_entropy(yhat_test, ytest))
                
                yhat_test = torch.argmax(softmax(yhat_test), dim = 1)
                correct_preds += torch.sum((yhat_test == ytest).int()).item()
                total_preds += len(yhat_test.view(-1,1))
            
            test_loss = np.mean(loss_list)
            
            # --- Calcul de l'accuracy en test
            print("{} correct preds out of {}".format(correct_preds, total_preds))
            accuracy = correct_preds / total_preds
            
        # --- Affichage tensorboard
            
        writer.add_scalar('Loss/train/{}/{}'.format(latent_dim, length), train_loss, epoch)
        print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
        writer.add_scalar('Loss/test/{}/{}'.format(latent_dim, length), test_loss, epoch)
        writer.add_scalar('Accuracy/test/{}/{}'.format(latent_dim, length), accuracy, epoch)
        print('Epoch {} | Test accuracy: {}' . format(epoch, accuracy))


## Tests sur différentes valeurs de length
#sequence_classifier(input_dim = 2, latent_dim = 10, output_dim = 80, length = 5, n_epochs = N_EPOCHS, epsilon = 1e-1)
#sequence_classifier(input_dim = 2, latent_dim = 10, output_dim = 80, length = 20, n_epochs = N_EPOCHS, epsilon = 1e-1)
#sequence_classifier(input_dim = 2, latent_dim = 10, output_dim = 80, length = 50, n_epochs = N_EPOCHS, epsilon = 1e-1)

#sequence_classifier(input_dim = 2, latent_dim = 20, output_dim = 80, length = 20, n_epochs = N_EPOCHS, epsilon = 1e-1)