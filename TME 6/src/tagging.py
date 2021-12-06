#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  27 10:20:14 2021

@author: Cécile GIANG
"""

###################################################################################################
# ---------------------------- IMPORTATION DES MODULES ET LIBRAIRIES ---------------------------- #
###################################################################################################


import itertools
import logging
from tqdm import tqdm

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
from typing import List
import time


# On utilise le GPU s'il est disponible, le CPU sinon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###################################################################################################
# ----------------------------------------- UTILITAIRES ----------------------------------------- #
###################################################################################################


# Format de sortie décrit dans
# https://pypi.org/project/conllu/


class Vocabulary:
    """ Permet de gérer un vocabulaire.
        En test, il est possible qu'un mot ne soit pas dans le vocabulaire : dans ce cas le token 
        "__OOV__" est utilisé. Attention: il faut tenir compte de cela lors de l'apprentissage !
        
        Utilisation:
        * en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté automatiquement 
          s'il n'est pas connu.
        * en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV).
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ Constructeur de la classe Vocabulary.
            @param oov: bool, autorise ou non les mots OOV
        """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        """ Retourne l'indice du mot word dans le vocabulaire.
            @param word: str, mot dont on veut connaître l'indice
        """
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        """ Retourne l'indice du mot word dans le vocabulaire s'il existe, l'ajoute dans le vocabulaire
            sinon (adding = True).
            @param wprd: str, mot dont on veut connaître l'indice ou que l'on souhaite rajouter
        """
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        """ Retourne la taille du vocabulaire.
        """
        return len(self.id2word)

    def getword(self,idx: int):
        """ Retourne le mot associé à l'indice idx dans le vocabulaire.
            @param idx: int, indice d'un mot dans le vocabulaire
        """
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        """ Retourne le mot associé aux indices de idx dans le vocabulaire.
            @param idx: list(int), liste d'indices de mots dans le vocabulaire
        """
        return [self.getword(i) for i in idx]


class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        """ Dataset pour les données d'étiquetage. Chaque item est un couple (token, tag)
            * data : dataset contenant les données d'étiquetage.
            * words : vocabulaire des mots.
            * tags : vocabulaire des étiquettes.
            * adding: vaut True
        """
        self.sentences = []
        
        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
            
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """ Collate using pad_sequence.
    """
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


###################################################################################################
# ----------------------------------- CHARGEMENT DES DONNEES ------------------------------------ #
###################################################################################################


# Taille des batchs d'échantillons
BATCH_SIZE = 128


# Chargement du dataset
logging.basicConfig(level = logging.INFO)
ds = prepare_dataset('org.universaldependencies.french.gsd')
logging.info("Loading datasets...")

# Initialisation du vocabulaire
words = Vocabulary(True)
tags = Vocabulary(False)
logging.info("Vocabulary size: %d", len(words))


# Initialisation des données train, test et validation

train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)

train_loader = DataLoader(train_data, collate_fn = collate_fn, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
dev_loader = DataLoader(dev_data, collate_fn = collate_fn, batch_size = BATCH_SIZE, drop_last = True)
test_loader = DataLoader(test_data, collate_fn = collate_fn, batch_size = BATCH_SIZE, drop_last = True)


###################################################################################################
# -------------------------------------- TAGGING MODULE  ---------------------------------------- #
###################################################################################################


class LSTMTagger(nn.Module):
    """ Classe pour un réseau d'étiquetage.
    """
    def __init__(self, embed_dim, latent_dim, output_dim, vocab_size):
        """ @param embed_dim: int, dimension de l'embedding
            @param latent_dim: int, dimension des états cachés pour le LSTM
            @param output_dim: int, dimension de sortie (taille du vocabulaire des tags)
            @param vocab_size: int, taille du vocabulaire
        """
        super(LSTMTagger, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        
        # Initialisation de l'embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # Initialisation du LSTM qui prend en entrée les embeddingstargets
        self.lstm = nn.LSTM(self.embed_dim, self.latent_dim)
        
        # Initialisation de la couche linéaire de sortie (mappe de l'espace des états cachés à l'espace des tags)
        self.hidden2tag = nn.Linear(self.latent_dim, self.output_dim)
    
    def forward(self, sentence):
        """ @param sentence: torch.Tensor, phrase sous la forme d'indices sur le vocabulaire words
        """
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        
        return tag_scores


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
# ------------------------- BOUCLE D'APPRENTISSAGE: MODELE D'ETIQUETAGE ------------------------- #
###################################################################################################


N_EPOCHS = 50


def train_model(embed_dim = 60, latent_dim = 50, output_dim = tags.__len__(), vocab_size = words.__len__(), epsilon = 1e-3):
    """ Boucle d'apprentissage du modèle d'étiquetage. 
        @param embed_dim: dimension de l'embedding
        @param latent_dim: int, dimension de l'espace latent (états cachés pour le RNN)
        @param output_dim: int, dimension de sortie
        @param epsilon: float, learning rate
    """
    # Pour l'affichage avec tensorboard
    writer = SummaryWriter("runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Chemin vers le modèle. Reprend l'apprentissage si un modèle est déjà sauvegardé.
    savepath = Path('Tagging_emb{}_lat{}.pch' . format(embed_dim, latent_dim))
    
    if savepath.is_file():
        with savepath.open('rb') as file:
            state = torch.load(file)
    
    else:
        # Création du modèle et de l'optimiseur, chargement sur device
        model = LSTMTagger(embed_dim, latent_dim, output_dim, vocab_size)
        model = model.to(device)
        optim = torch.optim.Adam(params = model.parameters(), lr = epsilon) # lr : pas de gradient
        state = State(model, optim)
    
    # Initialisation de la loss
    nll = nn.NLLLoss()
    
    # --- Phase d'apprentissage
    for epoch in range(state.epoch, N_EPOCHS):
        
        for sentences, targets in train_loader:
            
            # --- Remise à zéro des gradients des paramètres à optimiser
            state.optim.zero_grad()
            
            # --- Chargement du batch sur device
            sentences = sentences.to(device)
            targets = targets.to(device)
            
            # --- Entraînement du modèle et calcul des scores moyens
            batch_scores = []      
            
            for x in sentences:
                score = state.model(x)
                batch_scores.append(score)
            
            batch_scores = torch.stack(batch_scores)
            n, m, p = batch_scores.shape
            
            # --- Phase backward
            train_loss = nll( batch_scores.reshape( n * m, p ), targets.reshape( n * m ) )
            train_loss.backward()
            
            # --- Mise à jour des paramètres
            state.optim.step()
            state.iteration += 1
        
        
        # --- Phase de test
        with torch.no_grad():
            for sentences_, targets_ in test_loader:
                
                # --- Chargement du batch sur device
                sentences_ = sentences_.to(device)
                targets_ = targets_.to(device)
                
                # --- Entraînement du modèle et calcul des scores moyens
                batch_scores_ = []            
                
                for x_ in sentences_:
                    score_ = state.model(x_)
                    batch_scores_.append(score_)
                
                batch_scores_ = torch.stack(batch_scores_)
                n_, m_, p_ = batch_scores_.shape
                
                # --- Phase backward
                test_loss = nll( batch_scores_.reshape( n_ * m_, p_ ), targets_.reshape( n_ * m_ ) )
        
        with savepath.open('wb') as file:
            state.epoch = epoch + 1
            torch.save(state, file)
        
        # --- Affichage tensorboard
        writer.add_scalar('Loss/train/{}/{}'.format(embed_dim, latent_dim), train_loss, epoch)
        print('Epoch {} | Training loss: {}' . format(epoch, train_loss))
        writer.add_scalar('Loss/test/{}/{}'.format(embed_dim, latent_dim), test_loss, epoch)

### TESTS

"""
## Entraîner un modèle
train_model(embed_dim = 60, latent_dim = 50, output_dim = tags.__len__(), vocab_size = words.__len__(), epsilon = 1e-3)

## On ouvre un modèle déjà entraîné
state = torch.load('Tagging_emb60_lat50.pch')

## Vérifier une séquence de tags générée
x, y = next(iter(train_loader))
model = state.model

# x_ et y_ la 1ère phrase du batch et les tags correspondants
x_ = x[:,0]
y_ = y[:,0]

# yhat les probabilités associées à chaque tag pour chaque mot
yhat = model(x_)

# preds la séquence de tags prédite sur x_
preds = torch.stack([torch.argmax(e) for e in yhat])

# Affichage des tags prédits (preds) et des tags attendus (y_)
print('Phrase : \n', words.getwords(x_))
print('\n\nTags prédits: \n', tags.getwords(preds))
print('\n\nTags attendus: \n', tags.getwords(y_))
"""