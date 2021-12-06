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


# On utilise le GPU s'il est disponible, le CPU sinon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###################################################################################################
# ----------------------------------- IMPLEMENTATION D'UN RNN ----------------------------------- #
###################################################################################################


class RNN(nn.Module):
    """ Classe pour un réseau récurrent (RNN).
    """
    def __init__(self, input_dim, latent_dim, output_dim, length):
        """ @param input_dim: int, dimension de l'entrée
            @param latent_dim: int, dimension de l'état caché
            @param output_dim: int, dimension de la sortie
            @param length: int, longueur de chaque séquence temporelle
        """
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.length = length

        # Initialisation des modules linéaires pour l'entrée (in), les états cachés (lat) et le décodeur (out)
        self.linear_in = nn.Linear(self.input_dim, self.latent_dim)
        self.linear_lat = nn.Linear(self.latent_dim, self.latent_dim)
        self.linear_out = nn.Linear(self.latent_dim, self.output_dim)

        # Initialisation du module TanH pour le calcul de l'état caché
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU()

    def one_step(self, x, h):
        """ Traite un pas de temps: renvoie le prochain état caché.
            @param x: torch.Tensor, batch des séquences à l'instant t de taille (batch, dim)
            @param h: torch.Tensor, batch des états cachés à l'instant t-1 de taille (batch, latent)
        """
        return self.tanh( self.linear_in(x) + self.linear_lat(h) )

    def forward(self, x, h):
        """ Traite tout le batch de séquences passé en paramètre en appelant successivement la
            méthode forward sur tous les éléments des séquences.
            Renvoie la séquence des états cachés calculés de taille (length, batch, latent)
            @param x: torch.Tensor, batch de séquences à l'instant t de taille (length, batch, dim)
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        # Initialisation de la séquence des état cachés
        hidden_states = list()

        # Appel de la méthode one_step sur nos séquences à chaque instant i
        for i in range(self.length):
            x_t = x[:, i]
            h = self.one_step(x_t, h)
            hidden_states.append(h)

        return torch.stack(hidden_states)

    def decode_linear(self, h):
        """ Décode le batch d'états cachés. Renvoie la sortie d'intérêt y de taille (batch, output).
            L'activation non-linéaire s'effectuera dans la boucle d'apprentissage.
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        return self.linear_out(h)

    def decode_ReLU (self, h) :
        return self.relu(self.linear_out(h))

    def parameters(self):
        return list(self.linear_in.parameters()) + list(self.linear_lat.parameters()) + list(self.linear_out.parameters())


###################################################################################################
# --------------------------------------- CLASSES DATASET --------------------------------------- #
###################################################################################################


class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.stations_max = stations_max
        self.data, self.length= data, length
        if stations_max is None:
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.stations_max = stations_max
        self.data, self.length= data,length
        if stations_max is None:
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]