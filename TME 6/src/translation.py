#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  27 10:20:27 2021

@author: Cécile GIANG
"""

###################################################################################################
# ---------------------------- IMPORTATION DES MODULES ET LIBRAIRIES ---------------------------- #
###################################################################################################


import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List

import time
import re
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)

FILE = "../../data/en-fra.txt"

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
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
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage

# =============================================================================
# GRU
# =============================================================================

class GRU(nn.Module):
    """ Classe pour un réseau Gated Recurrent Units (GRU).
    """
    def __init__(self, input_dim, latent_dim, output_dim):
        """ @param input_dim: int, dimension de l'entrée
            @param latent_dim: int, dimension de l'état caché
            @param output_dim: int, dimension de la sortie
        """
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Initialisation des différentes portes et modules linéaires
        self.gate_z = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.gate_r = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.gate_h = nn.Linear(self.input_dim + self.latent_dim, self.latent_dim)
        self.linear_out = nn.Linear(self.latent_dim, self.output_dim)
        
        # Initialisation des modules TanH et Sigmoide
        self.tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    
    def one_step(self, x, h=None):
        """ Traite un pas de temps: renvoie les prochains états externe (ht) et interne (Ct)
            @param x: torch.Tensor, batch des séquences à l'instant t de taille (batch, input)
            @param h: torch.Tensor, batch des états cachés à l'instant t-1 de taille (batch, latent)
        """
        if h==None:
            h = torch.zeros(1, self.latent_dim, dtype = torch.float)
        
        # Concaténation vectorielle de h_{t-1} et x_t
        x_concat = torch.cat((h, x), dim = 1)
        
        # Calcul des différentes portes
        zt = self.sigmoid( self.gate_z( x_concat ) )
        rt = self.sigmoid( self.gate_r( x_concat ) )
        
        # Mise à jour des mémoires interne/externe (ht)
        ht = (1 - zt) * h + zt * self.tanh( self.gate_h( torch.cat((rt * h, x), dim = 1) ) )
        
        return ht
    
    def forward(self, x, h=None):
        """ Traite tout le batch de séquences passé en paramètre en appelant successivement la
            méthode forward sur tous les éléments des séquences. 
            Renvoie la séquence des états cachés calculés de taille (batch, latent)
            @param x: torch.Tensor, batch de séquences à l'instant t de taille (length, batch, dim)
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        if h==None:
            h = torch.zeros(1, self.latent_dim, dtype = torch.float)
        
        # Initialisation de la séquence des état cachés
        hidden_states = list()
        
        # Appel de la méthode one_step sur nos séquences à chaque instant i
        for i in range(len(x)):
            h = self.one_step(x[i], h)
            hidden_states.append(h)
            
        return torch.stack(hidden_states)
    
    def decode(self, h):
        """ Décode le batch d'états cachés. Renvoie la sortie d'intérêt y de taille (batch, output).
            L'activation non-linéaire s'effectuera dans la boucle d'apprentissage.
            @param h: torch.Tensor, batch des états cachés de taille (batch, latent)
        """
        return self.linear_out(h)

# =============================================================================
# Encodeur
# =============================================================================

class Encodeur(nn.Module):
    """ Classe de l'encodeur pour la tâche de traduction.
    """
    
    def __init__(self, embed_dim, latent_dim, output_dim, vocab):
        """ @param embed_dim: int, dimension de l'embedding
            @param latent_dim: int, dimension des états cachés pour le GRU
            @param output_dim: int, dimension de sortie
            @param vocab_size: int, taille du vocabulaire
        """
        super(Encodeur, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.vocab = vocab
        
        # Initialisation de l'encodeur
        self.vocab_size = self.vocab.__len__()
        self.embedding_in = nn.Embedding(self.vocab_size, self.embed_dim)
        self.gru_enc = GRU(embed_dim, latent_dim, output_dim)
        
        
    def forward(self, sentence):
       """ @param sentence: torch.Tensor, phrase sous la forme d'indices sur le vocabulaire words
       """
       embeds = self.embedding_in(sentence)
       gru_out, _ = self.gru_enc(embeds.view(len(sentence), 1, -1))
       
       return gru_out
       
# =============================================================================
# Décodeur
# =============================================================================
        
class Decodeur(nn.Module):
    """ Classe du décodeur pour la tâche de traduction.
    """
    
    def __init__(self, embed_dim, latent_dim, output_dim, vocab):
        """ @param embed_dim: int, dimension de l'embedding
            @param latent_dim: int, dimension des états cachés pour le GRU
            @param output_dim: int, dimension de sortie
            @param vocab_size: int, taille du vocabulaire
        """
        super(Decodeur, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.vocab = vocab
        
        # Initialisation du décodeur
        self.vocab_size = vocab.__len__()
        self.embedding_out = nn.Embedding(self.vocab_size, self.embed_dim)
        self.gru_dec = GRU(embed_dim, latent_dim, output_dim)
        
    def forward(self, x):
       embeds = self.embedding_out(x)
       gru_out, _ = self.gru_dec(embeds.view(len(x), 1, -1))
       output = torch.argmax(nn.softmax(self.gru_dec.decode(gru_out[-1]), dim=1), axis = 1)
       
       return output
       
    def generate(self, hidden, lenseq=None):
        idx_SOS = self.vocab.get('SOS')
        idx_EOS = self.vocab.get('EOS')
        
        gen_sequence = [idx_SOS]
        gen_len = 1
        ht = torch.zeros(1, self.latent_dim, dtype = torch.float)
        
        while gen_sequence[-1] != idx_EOS and len(gen_sequence) < lenseq:
            ht = self.gru_dec.one_step(self.embedding_out(gen_sequence[-1]), ht)
            output = nn.softmax(self.gru_dec.decode(ht), dim=1)
            gen_sequence.append(torch.argmax(output, axis = 1))
            
        gen_sequence = torch.Tensor(gen_sequence)
        
        return self.vocab.getwords(gen_sequence)
        
        
###################################################################################################
# ----------------------------- APPRENTISSAGE DE RESEAUX RECURRENTS ----------------------------- #
###################################################################################################


N_EPOCHS = 50


def train_model(embed_dim=60, latent_dim=50, vocab_in=vocEng, vocab_out=vocFra, epsilon = 1e-3):
    """ Boucle d'apprentissage du modèle d'étiquetage. 
        @param embed_dim: dimension de l'embedding
        @param latent_dim: int, dimension de l'espace latent (états cachés pour le RNN)
        @param output_dim: int, dimension de sortie
        @param epsilon: float, learning rate
    """

    # Création du modèle et de l'optimiseur, chargement sur device
    encodeur = Encodeur(embed_dim, latent_dim, vocab_in.__len__(), vocab_in)
    decodeur = Decodeur(embed_dim, latent_dim, vocab_out.__len__(), vocab_out)
    
    parameters = list(encodeur.parameters()) + list(decodeur.parameters())
    
    optim = torch.optim.Adam(params = parameters, lr = epsilon) # lr : pas de gradient
    
    # Initialisation de la loss
    nll = nn.NLLLoss()
    
    # --- Phase d'apprentissage
    for epoch in range(0, N_EPOCHS):
        
        print(next(iter(train_loader)).shape)
        
        for sentences, targets in train_loader:
            
            # --- Remise à zéro des gradients des paramètres à optimiser
            optim.zero_grad()
            
            # --- Chargement du batch sur device
            sentences = sentences.to(device)
            targets = targets.to(device)
            
            # --- Entraînement du modèle et calcul des scores moyens
            batch_scores = []      
            
            for x in sentences:
                enc = encodeur(x)
                dec = decodeur(enc)

            # --- Phase backward
            train_loss = nll( dec, targets)
            train_loss.backward()
            
            # --- Mise à jour des paramètres
            optim.step()
        
        """
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
        """

## Entraîner un modèle
train_model(embed_dim=60, latent_dim=50, vocab_in=vocEng, vocab_out=vocFra, epsilon = 1e-3)