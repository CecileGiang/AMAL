#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  12 16:07:28 2021

@author: Cécile GIANG
"""

from pathlib import Path
import os
import torch
import torchvision
from torchvision.utils import make_grid
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime


###################################################################################################
# ----------------------------------------- UTILITAIRES ----------------------------------------- #
###################################################################################################


# Chargement des données
train = datasets.MNIST(root = './data', train = True, download = True, transform = None)
test = datasets.MNIST(root = './data', train = False, download = True, transform = None)
train_images, train_labels = train.data, train.targets
test_images, test_labels = test.data, test.targets

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# ------- Visualisation

# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.

# Permet de fabriquer une grille d'images
images = make_grid(images)

# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


###################################################################################################
# ----------------------------------- DATASET, DATASETLOADER ------------------------------------ #
###################################################################################################


BATCH_SIZE = 5000

class MNISTDataset(Dataset):
    """ Classe pour la représentation des données MNIST.
    """
    def __init__(self, data, labels):
        """ Constructeur du dataset pour le jeu de données MNIST. 
    	    Les images sont transformées en vecteurs normalisés entre 0 et 1.
            @param data: torch.Tensor, exemples
            @param labels: torch.Tensor, étiquettes
        """
        self.data = torch.div(data, 255.)
        self.labels = labels

    def __getitem__(self, index):
        """ Retourne un couple (exemple, label) correspondant à l'index.
            @param index: int, indice de l'échantillon à renvoyer
        """
        return self.data[index], self.labels[index]

    def __len__(self):
        """ Renvoie la taille du jeu de données. 
        """
        return len(self.data)


# Initialisation de Dataloader qui est un itérateur sur nos données, et qui permet de spécifier 
# la taille des batchs, s'il faut mélanger ou pas les données et de charger les données en parallèle 
# (multiprocessing)

def seed_worker(worker_id):
    """ Pour assurer la reroducibilité des expérimentations.
        Génération du noyau.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

data_train = DataLoader(MNISTDataset(train_images, train_labels), shuffle = True, batch_size = BATCH_SIZE, num_workers = 0, worker_init_fn = seed_worker, generator = g)
data_test = DataLoader(MNISTDataset(test_images, test_labels), shuffle = True, batch_size = BATCH_SIZE, num_workers = 0, worker_init_fn = seed_worker, generator = g)

## Tests:
# batch_id: identifiant du batch dans le dataloader
# batch: couple (données, labels) pour le batch

#for batch_id, batch in enumerate(data_train):
#    print(batch_id, batch)


###################################################################################################
# ------------------------------- IMPLÉMENTATION D'UN AUTOENCODEUR ------------------------------ #
###################################################################################################


class AutoEncoder(torch.nn.Module):
    """ Classe pour l'auto-encodeur.
    """
    def __init__(self, input_dim, latent_dim):
        """ @param input_dim: int, taille de chaque échantillon d'entrée
            @param latent_dim: int, taille de chaque échantillon dans l'espace latent
        """
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Initialisation des modules linéaire et ReLU pour l'encodeur
        self.linear_1 = torch.nn.Linear(input_dim, latent_dim, bias = True)
        self.relu = torch.nn.ReLU()

        # Initialisation des modules linéaire et sigmoïde pour le décodeur
        self.linear_2 = torch.nn.Linear(latent_dim, input_dim, bias = True)
        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, x):
        """ @param x: torch.Tensor, données à encoder
        """
        return self.relu( self.linear_1(x) )

    def decode(self, x):
        """ @param x: torch.Tensor, données à décoder
        """
        self.linear_2.weight = torch.nn.Parameter(self.linear_1.weight.t())
        return self.sigmoid( self.linear_2(x) )

    def forward(self, x):
        """ @param x: torch.Tensor, données
        """
        return self.decode( self.encode(x) )

## Tests
# x, y = next(iter(data_train)) # On tire un batch du dataloader
# img = x[0]
# plt.imshow(img) # Visualisation de la 1ère image du batch initial
# ae = AutoEncoder(x.shape[1], 10).forward(x)
# img_lat = ae[0]
# plt.imshow(img_lat.detach().numpy()) # Visualisation de la 1ère image reconstruite dans le batch


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
# ----------------------------------- CAMPAGNE D'EXPERIENCES ------------------------------------ #
###################################################################################################


N_EPOCHS = 100

def autoencoding_neuralnet(latent_dim, n_epochs = N_EPOCHS, epsilon = 1e-1):
    """ Réseau de neurones pour minimiser le coût cross entropique entre les images
        initiales et les images reconstruites sur les données MNIST.
    """
    # Pour l'affichage avec tensorboard
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Sélectionner le GPU s'il est disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Chemin vers le modèle. Reprend l'apprentissage si un modèle est déjà sauvegardé.
    savepath = Path('model_{}.pch'.format(latent_dim))
    
    if savepath.is_file():
        with savepath.open('rb') as file:
            state = torch.load(file)
    
    else:
        # Création du modèle et de l'optimiseur, chargement sur device
        model = AutoEncoder(train_images.shape[1], latent_dim)
        model = model.to(device)
        optim = torch.optim.SGD(params = model.parameters(), lr = epsilon) # lr : pas de gradient
        state = State(model, optim)
    
    # Initialisation de la loss
    bce = torch.nn.BCELoss()
    
    # --- Phase d'apprentissage
    for epoch in range(state.epoch, N_EPOCHS):
        
        for x, y in data_train:
            
            # --- Remise à zéro des gradients des paramètres à optimiser
            state.optim.zero_grad()
            
            # --- Chargement du batch sur device
            x = x.to(device)
            
            # --- Phase forward
            xhat = state.model.forward(x)
            
            # --- Phase backward
            train_loss = bce(xhat, x)
            train_loss.backward()
            
            # --- Mise à jour des paramètres
            state.optim.step()
            state.iteration += 1
            
            with savepath.open('wb') as file:
                state.epoch = epoch + 1
                torch.save(state, file)
        
        # --- Phase de test
            
        with torch.no_grad():
            
            loss_list = []
            
            for xtest, ytest in data_test:
                xtest = xtest.to(device)
                xhat_test = state.model.forward(xtest)
                loss_list.append(bce(xhat_test, xtest))
            
            test_loss = np.mean(loss_list)
            
        # --- Affichage tensorboard
            
        writer.add_scalar('Loss/train/{}'.format(latent_dim), train_loss, epoch)
        print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
        writer.add_scalar('Loss/test/{}'.format(latent_dim), test_loss, epoch)
        

## Tests

import time

# --- Pour entrainer un auto-encodeur avec une projection en dimension latent_dim

"""
latent_dim = 32

start = time.time()
torch.manual_seed(42)
autoencoding_neuralnet(latent_dim)
print('Temps d\'exécution: {} ms'.format(time.time() - start))

# --- Pour visualiser une image de data_train et une image reconstruite

path = 'model_{}.pch'.format(latent_dim)

data, label = next(iter(data_train))
state = torch.load(path)
model = state.model

# Affichage de l'image initiale
img = data[0]
plt.imshow(img, cmap = 'gray')
plt.show()

# Affichage de l'image reconstruite
new_img = model.forward(img)
plt.imshow(new_img.detach().numpy(), cmap = 'gray')
plt.show()
"""

## Observation: plus la dimension latente est faible, plus la loss est élevée

## Temps d'exécution sur 100 époques
# model_32.pch: 255.52201676368713 ms
# model_64.pch: 302.09655356407166 ms
# model_128.pch: 427.8325228691101 ms
# model_256.pch: 688.2594003677368 ms


###################################################################################################
# --------------------------------------- HIGHWAY NETWORK --------------------------------------- #
###################################################################################################


# --- Liens pour le highway network
# https://github.com/harshanavkis/PyTorch-implementation-of-Highway-Networks/blob/master/HighwayNet.py
# https://github.com/kefirski/pytorch_Highway/blob/master/highway/highway.py


N_EPOCHS = 50

class Highway(torch.nn.Module):
    """ Classe pour le highway network.
    """
    def __init__(self, input_dim, num_layers = 10):
        """ On choisit H une transformée affine suivie d'une activation non-linéaire
            (ici ReLU), et T une transformée affine suivie d'une autre activation 
            non-linéaire (ici sigmoïde).
            @param input_dim: int, taille de chaque échantillon d'entrée
            @param num_layers: int, nombre de couches dans le réseau de neurones
        """
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # Initialisation des modules H
        self.H = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim) for _ in range(self.num_layers)])
        
        # Initialisation des modules T
        self.T = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim) for _ in range(self.num_layers)])
        
    def forward(self, x):
        """ @param x: torch.Tensor, données
        """
        # Copie du tenseur à forwarder
        x_ = torch.clone(x)
        
        # Propagation sur toutes les couches
        for layer in range(self.num_layers):
            
            # --- Calcul intermédiaire des H et T
            h_out = torch.nn.functional.relu( self.H[layer](x_) )
            t_out = torch.sigmoid( self.T[layer](x_) )
            
            # --- Mise à jour de x_
            x_ = h_out * t_out + (1 - t_out) * x
        
        return x_


def highway_neuralnet(num_layers = 10, n_epochs = N_EPOCHS, epsilon = 1e-1):
    """ Highway network sur les données MNIST.
    """
    # Sélectionner le GPU s'il est disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Chemin vers le modèle. Reprend l'apprentissage si un modèle est déjà sauvegardé.
    savepath = Path('model_highway_{}.pch'.format(num_layers))
    
    if savepath.is_file():
        with savepath.open('rb') as file:
            state = torch.load(file)
    
    else:
        # Création du modèle et de l'optimiseur, chargement sur device
        model = Highway(train_images.shape[1], num_layers)
        model = model.to(device)
        optim = torch.optim.SGD(params = model.parameters(), lr = epsilon) # lr : pas de gradient
        state = State(model, optim)
    
    # Initialisation de la loss
    bce = torch.nn.BCELoss()
    
    # --- Phase d'apprentissage
    for epoch in range(state.epoch, N_EPOCHS):
        
        for x, y in data_train:
            
            # --- Remise à zéro des gradients des paramètres à optimiser
            state.optim.zero_grad()
            
            # --- Chargement du batch sur device
            x = x.to(device)
            
            # --- Phase forward
            xhat = state.model.forward(x)
            
            # --- Phase backward
            train_loss = bce(xhat, x)
            train_loss.backward()
            
            # --- Mise à jour des paramètres
            state.optim.step()
            state.iteration += 1
            
            with savepath.open('wb') as file:
                state.epoch = epoch + 1
                torch.save(state, file)
        
        # --- Phase de test
            
        with torch.no_grad():
            
            loss_list = []
            
            for xtest, ytest in data_test:
                xtest = xtest.to(device)
                xhat_test = state.model.forward(xtest)
                loss_list.append(bce(xhat_test, xtest))
            
            test_loss = np.mean(loss_list)
            
        # --- Affichage tensorboard
            
        writer.add_scalar('Loss/train/', train_loss, epoch)
        print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
        writer.add_scalar('Loss/test/', test_loss, epoch)


## Tests

# --- Pour entrainer un highway network avec un réseau à num_layers couches

"""
num_layers = 10

start = time.time()
torch.manual_seed(42)
highway_neuralnet(num_layers)
print('Temps d\'exécution: {} ms'.format(time.time() - start))

# --- Pour visualiser une image de data_train et une image reconstruite

path = 'model_highway_{}.pch'.format(num_layers)

data, label = next(iter(data_train))
state = torch.load(path)
model = state.model

# Affichage de l'image initiale
img = data[0]
plt.imshow(img, cmap = 'gray')
plt.show()

# Affichage de l'image reconstruite
new_img = model.forward(img)
plt.imshow(new_img.detach().numpy(), cmap = 'gray')
plt.show()
"""

## Observation: plus le nombre de couches est élevée, plus la loss est élevée

## Temps d'exécution sur 50 époques
# model_highway_10.pch: 1236.480786561966 ms
# model_highway_5.pch: 653.408093214035 ms