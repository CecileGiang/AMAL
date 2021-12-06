from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from utils import RNN, ForecastMetroDataset


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
BATCH_SIZE = 32

train, test = torch.load('../data/hzdataset.pch')
train_dataset = ForecastMetroDataset(train, length = 20)
test_dataset = ForecastMetroDataset(test, length =10)
data_train = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0,  drop_last=True)
data_test = DataLoader(test_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 0,  drop_last=True)


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

BATCH_SIZE = 32
N_EPOCHS = 401



def prediction(input_dim, latent_dim, l=20, horizon=10, n_epochs = N_EPOCHS):
    # Découpage de nos données en batchs de séquences de longueur length
    train_dataset = ForecastMetroDataset(train, length = l)
    test_dataset = ForecastMetroDataset(test, length = l)
    data_train = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE, drop_last=True)
    data_test = DataLoader(test_dataset, shuffle = True, batch_size = BATCH_SIZE, drop_last=True)

    # Pour l'affichage aec tensorboard
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    # Création du modèle et de l'optimiseur, chargemet sur device
    model = RNN(input_dim, latent_dim, input_dim, horizon)
    optim = torch.optim.Adam(params = model.parameters())

    # Initialisation de la mse loss
    mse = nn.MSELoss()

    # --- Phase d'apprentissage
    for epoch in range(N_EPOCHS):

        loss_list = []

        for x,y in data_train:
            x = x[:,:,0,:]
            optim.zero_grad()
            h = torch.zeros(BATCH_SIZE, latent_dim, requires_grad = True)
            h_out = model.forward(x, h)
            yhat = model.decode_ReLU(h_out[-1])
            train_loss = mse(yhat, y[:,-1,0,:])
            loss_list.append(train_loss.item())
            train_loss.backward()
            optim.step()

        train_loss = np.mean(loss_list)

        with torch.no_grad():

            loss_list_test = []
            for xtest, ytest in data_test:
                xtest = xtest[:,:,0,:]
                h_test = torch.zeros(BATCH_SIZE, latent_dim)
                htest = model.forward(xtest, h_test)
                yhat_test = model.decode_ReLU(htest[-1])
                test_loss = mse(yhat_test, ytest[:,-1,0,:])
                loss_list_test.append(test_loss.item())

            test_loss = np.mean(loss_list_test)


        writer.add_scalar('Loss/train/{}/{}_Forecast'.format(latent_dim, l), train_loss, epoch)

        writer.add_scalar('Loss/test/{}/{}_Forecast'.format(latent_dim, l), test_loss, epoch)

        if(epoch%50==0) :
            print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
            print('Epoch {} | Test loss: {}' . format(epoch, train_loss))

#sequence_predictor(2, 20, l=10, horizon=1, n_epochs = N_EPOCHS)