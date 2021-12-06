from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from utils import RNN, SampleMetroDataset

# On utilise le GPU s'il est disponible, le CPU sinon
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

#Nombre d'epoch
N_EPOCHS = 50

train, test = torch.load('../data/hzdataset.pch')


###################################################################################################
# --------------------------------------- CHECKPOINTING ----------------------------------------- #
###################################################################################################


# =============================================================================
# class State:
#     """ Classe de sauvegarde sur l'apprentissage d'un modèle.
#     """
#     def __init__(self, model, optim):
#         self.model = model
#         self.optim = optim
#         self.epoch, self.iteration = 0,0
# =============================================================================

###################################################################################################
# ---------------------------------- CLASSIFICATION DE SEQUENCE --------------------------------- #
###################################################################################################

"""
Notre objectif est de contruire un modèle qui à partir d'une séquence d'une certaine longuer infère
la station à laquelle appartient la séquence.
"""



def sequence_classifier(input_dim, latent_dim, output_dim, length=10, n_epochs = N_EPOCHS):
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
    data_train = DataLoader(train_dataset, shuffle = True, batch_size = BATCH_SIZE,drop_last=True)
    data_test = DataLoader(test_dataset, shuffle = True, batch_size = BATCH_SIZE,drop_last=True)

    # Pour l'affichage aec tensorboard
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    #Chemin vers le modèle. Reprend l'apprentissage si un modèle est déjà sauvegardé.
# =============================================================================
#     savepath = Path('classifier_lat{}_len{}.pch'.format(latent_dim, length))
#
#     if savepath.is_file():
#         with savepath.open('rb') as file:
#             state = torch.load(file)
#
#     else:
# =============================================================================
    # Création du modèle et de l'optimiseur, chargemet sur device

    # Création du modèle et de l'optimiseur
    model = RNN(input_dim, latent_dim, output_dim, length)
    #model = model.to(device)
    optim_adam = torch.optim.Adam(params = model.parameters()) # lr : pas de gradient
    #state = State(model, optim_adam)
    # Initialisation de la loss cross entropique
    cross_entropy = nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):

        # Initialisation des loss en entraînement
        loss_list = []
        acc_list = []

        for x, y in data_train:
            # --- Remise à zéro des gradients des paramètres à optimiser
            optim_adam.zero_grad()

            # --- Chargement du batch et des étiquettes correspondantes sur device
            #x = x.to(device)
            #y = y.to(device)

            # Initialisation des états cachés de taille (batch, latent)
            h = torch.zeros(BATCH_SIZE, latent_dim, requires_grad = True)

            # --- Phase forward
            h = model.forward(x, h)

             # --- Décodage des états cachés finaux pour trouver le y d'intérêt
            yhat = model.decode_linear(h[-1])

            # --- Phase backward
            train_loss = cross_entropy(yhat, y)
            loss_list.append(train_loss.item())
            train_loss.backward()

            # --- Mise à jour des paramètres
            optim_adam.step()
# =============================================================================
#             state.optim.step()
#             state.iteration += 1
# =============================================================================

# =============================================================================
#                 with savepath.open('wb') as file:
#                     state.epoch = epoch + 1
#                     torch.save(state, file)
# =============================================================================

            # --- Calcul de la loss en phase d'apprentissage
            acc = torch.where(torch.max (yhat, dim=1)[1]==y,1,0).sum()/y.shape[0]
            acc_list.append(acc)


        train_loss = np.mean(loss_list)
        acc_train = np.mean(acc_list)


        # --- Phase de test
        with torch.no_grad():

            loss_list_test = []
            acc_list_test = []

            for xtest, ytest in data_test:

                h_test = torch.zeros(BATCH_SIZE, latent_dim, requires_grad = True)
                htest = model.forward(xtest, h_test)
                yhat_test = model.decode_linear(htest[-1])

                loss_list_test.append(cross_entropy(yhat_test, ytest).item())

                acc_t = torch.where(torch.max (yhat, 1)[1]==y,1,0).sum()/y.shape[0]
                acc_list_test.append(acc_t)

            test_loss = np.mean(loss_list_test)
            acc_test = np.mean(acc_list_test)


        writer.add_scalar('Loss/train/{}/{}_Classif'.format(latent_dim, length), train_loss, epoch)
        writer.add_scalar('Accuracy/train/{}/{}_Classif'.format(latent_dim, length), acc_train, epoch)
        print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
        writer.add_scalar('Loss/test/{}/{}_Classif'.format(latent_dim, length), test_loss, epoch)
        writer.add_scalar('Accuracy/test/{}/{}_Classif'.format(latent_dim, length), acc_test, epoch)
        print('Epoch {} | Test accuracy: {}' . format(epoch, acc_test))


## Tests sur différentes valeurs de length
#sequence_classifier(input_dim = 2, latent_dim = 20, output_dim = 80, length = 5, n_epochs = N_EPOCHS)
#sequence_classifier(input_dim = 2, latent_dim = 10, output_dim = 80, length = 20, n_epochs = N_EPOCHS)
#sequence_classifier(input_dim = 2, latent_dim = 10, output_dim = 80, length = 50, n_epochs = N_EPOCHS)

#sequence_classifier(input_dim = 2, latent_dim = 10, output_dim = 80, length = 20, n_epochs = N_EPOCHS)