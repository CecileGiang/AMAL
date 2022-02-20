# -*- coding: utf-8 -*-

"""
Created on Thu Jan 27 10:31:30 2022

@author: Cécile GIANG
"""


###############################################################################
# ------------------------------ IMPORTATIONS ------------------------------- #
###############################################################################

import math
import click
from torch.utils.tensorboard import SummaryWriter
import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import PositionalEncoding


###############################################################################
# ----------------------------- CONFIGURATION ------------------------------- #
###############################################################################


logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###############################################################################
# --------------------------- VARIABLES GLOBALES ---------------------------- #
###############################################################################

MAX_LENGTH = 500
BATCH_SIZE = 16
EMB_SIZE = 100


###############################################################################
# ------------------------- PREPARATION DES DONNEES ------------------------- #
###############################################################################

class FolderText(Dataset):
    """ Dataset basé sur des dossiers (un par classe) et fichiers.
    """

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        """ Constructeur de la classe FolderText.
        """
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]
    
    def get_txt(self,ix):
        s = self.files[ix]
        return s if isinstance(s,str) else s.read_text(), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """ Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage (embedding_size = [50,100,200,300]):
            - dictionnaire word vers ID
            - embeddings (Glove)
            - DataSet (FolderText) train
            - DataSet (FolderText) test
    """
    
    WORDS = re.compile(r"\S+")
    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    
    OOVID = len(words)
    words.append("__OOV__")
    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")
    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")
    
    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)


def load_data(emb_size = 100, batch_size = BATCH_SIZE):
    """ Chargement des données.
    """
    word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
    id2word = dict((v, k) for k, v in word2id.items())
    PAD = word2id["__OOV__"]
    embeddings = torch.Tensor(embeddings)
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
    
    def collate(batch):
        """ Collate function for DataLoader.
        """
        data = [torch.LongTensor(item[0][:MAX_LENGTH]) for item in batch]
        lens = [len(d) for d in data]
        labels = [item[1] for item in batch]
        return emb_layer(torch.nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value = PAD)).to(device), torch.LongTensor(labels).to(device), torch.Tensor(lens).to(device)
    
    # Chargement des données
    train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=batch_size, collate_fn=collate, drop_last = True)
    test_loader = DataLoader(test_data, batch_size=batch_size,collate_fn=collate,shuffle=False, drop_last = True)
    
    return train_loader, test_loader


## Chargement des données
train_loader, test_loader = load_data(emb_size = EMB_SIZE)


###############################################################################
# ------------------------ MODULES DE SELF-ATTENTION ------------------------ #
###############################################################################

class Attention1(nn.Module):
    """ Attention simple.
    """
    def __init__(self, emb_size):
        """ @param emb_size: int, taille de l'embedding
        """
        super(Attention1, self).__init__()
        
        # Taille de l'embedding
        self.emb_size = emb_size
        
        # Couches Queries, Keys et Values
        self.Q = nn.Linear(emb_size, emb_size)
        self.K = nn.Linear(emb_size, emb_size)
        self.V = nn.Linear(emb_size, emb_size)
        
        # Denière couche
        self.final = nn.Linear(emb_size, emb_size)
        
        # Softmax pour transformer les logits en probabilités
        self.softmax = nn.Softmax(dim = 1)
        
        # Couche de normalisation pour stabiliser le réseau
        self.layer_norm = nn.LayerNorm(emb_size)
    
    def forward(self, x):
        x_ = self.layer_norm(x)
        att = ( torch.matmul( self.Q(x_), torch.permute(self.K(x_), (0, 2, 1)) )) / np.sqrt(self.emb_size)
        att = self.softmax(att)
        out = torch.matmul( att, self.V(x_))
        return self.final(out)


class Attention2(nn.Module):
    """ Attention résiduelle.
    """
    def __init__(self, emb_size):
        """ @param emb_size: int, taille de l'embedding
        """
        super(Attention2, self).__init__()
        
        # Taille de l'embedding
        self.emb_size = emb_size
        
        # Couches Queries, Keys et Values
        self.Q = nn.Linear(emb_size, emb_size)
        self.K = nn.Linear(emb_size, emb_size)
        self.V = nn.Linear(emb_size, emb_size)
        
        # Denière couche
        self.final = nn.Linear(emb_size, emb_size)
        
        # Softmax pour transformer les logits en probabilités
        self.softmax = nn.Softmax(dim = 1)
        
        # Couche de normalisation pour stabiliser le réseau
        self.layer_norm = nn.LayerNorm(emb_size)
    
    def forward(self, x):
        x_ = self.layer_norm(x)
        att = ( torch.matmul( self.Q(x_), torch.permute(self.K(x_), (0, 2, 1)) )) / np.sqrt(self.emb_size)
        att = self.softmax(att)
        out = torch.matmul( att, self.V(x_))
        return self.final(out + x_)
    

class SelfAttention(nn.Module):
    """ Classe de Self-Attention.
    """
    def __init__(self, modeltype, emb_size = EMB_SIZE, dimout = 2, L = 3):
        """ @param modeltype: int, type de modèle d'attention
            @param emb_size: int, taille de l'embedding
            @param L: int, nombre de couches d'attention
        """
        super(SelfAttention, self).__init__()
        self.modeltype = modeltype
        self.emb_size = emb_size
        self.dimout = dimout
        self.L = L
        
        # Couches d'attention
        att_layers= []
        
        for i in range(self.L):
            att_layers.append(model_map[self.modeltype](self.emb_size))
            att_layers.append(nn.ReLU())
        
        self.attention = nn.Sequential(*att_layers)
        
        # Couche de classification
        self.linout = nn.Linear(self.emb_size, self.dimout)
    
    def forward(self, x):
        att = self.attention(x)
        att = torch.mean(att, axis=1)
        return self.linout(att)


class SelfAttentionPE(nn.Module):
    """ Classe de Self-Attention avec encoding de la position.
    """
    def __init__(self, modeltype, emb_size = EMB_SIZE, dimout = 2, L = 3):
        """ @param modeltype: int, type de modèle d'attention
            @param emb_size: int, taille de l'embedding
            @param L: int, nombre de couches d'attention
        """
        super(SelfAttentionPE, self).__init__()
        self.modeltype = modeltype
        self.emb_size = emb_size
        self.dimout = dimout
        self.L = L
        
        # Module pour encoder les positions dans une séquence
        self.pe = PositionalEncoding(d_model = self.emb_size, max_len = 500)
        
        # Couches d'attention
        att_layers= []
        
        for i in range(self.L):
            att_layers.append(model_map[self.modeltype](self.emb_size))
            att_layers.append(nn.ReLU())
        
        self.attention = nn.Sequential(*att_layers)
        
        # Couche de classification
        self.linout = nn.Linear(self.emb_size, self.dimout)
    
    def forward(self, x):
        att = self.attention(self.pe(x))
        att = torch.mean(att, axis=1)
        return self.linout(att)
    

class SelfAttentionPECLS(nn.Module):
    """ Classe de Self-Attention avec encoding de la position et token CLS.
    """
    def __init__(self, modeltype, emb_size = EMB_SIZE, dimout = 2, L = 3):
        """ @param modeltype: int, type de modèle d'attention
            @param emb_size: int, taille de l'embedding
            @param L: int, nombre de couches d'attention
        """
        super(SelfAttentionPECLS, self).__init__()
        self.modeltype = modeltype
        self.emb_size = emb_size
        self.dimout = dimout
        self.L = L
        
        # Module pour encoder les positions dans une séquence
        self.pe = PositionalEncoding(d_model = self.emb_size, max_len = 501)
        
        # Token CLS à apprendre
        self.cls = nn.Parameter(torch.ones(self.emb_size)) 
        
        # Couches d'attention
        att_layers= []
        
        for i in range(self.L):
            att_layers.append(model_map[self.modeltype](self.emb_size))
            att_layers.append(nn.ReLU())
        
        self.attention = nn.Sequential(*att_layers)
        
        # Couche de classification
        self.linout = nn.Linear(self.emb_size, self.dimout)
    
    def forward(self, x):
        x_cls = torch.cat((self.cls.repeat(BATCH_SIZE, 1, 1), x), dim=1)
        att = self.attention(self.pe(x_cls))
        
        # On ne classifie que sur le token CLS
        cls_learned = att[:,0,:]
        
        return self.linout(cls_learned)


# Dictionnaire de mapping des modèles d'attention
model_map = {0: Attention1, 1: Attention2}


###############################################################################
# ---------------------------------- STATE ---------------------------------- #
###############################################################################

class State(object) :
    """ Etat du modèle pour sauvegarde.
    """
    def __init__(self, model, optim, linconv=None):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0
        

###############################################################################
# -------------------------- BOUCLE D'APPRENTISSAGE ------------------------- #
###############################################################################
    
def main(train_data, test_data, attention = SelfAttention, modeltype = 0, emb_size = EMB_SIZE, n_epochs = 100, lr = 1e-4):
    """ Boucle d'apprentissage.
    """
    # x, y, l = next(iter(train_data))
    # x = x.to(device)
    # res = model_map[modeltype](emb_size).to(device)
    
    # sa = SelfAttentionPECLS(modeltype).to(device)
    # res_att = sa(x)
    
    # return res_att
    
    # Enregistrement des logs
    writer = SummaryWriter('runs')
    
    # On reprend l'entraînement si le modèle existe déjà: on le crée sinon
    savepath = Path('{}_{}_{}.pch'.format(attention.__name__.lower(), model_map[modeltype].__name__.lower(), emb_size))
    
    if savepath.is_file():
        print("Restarting from previous state.")
        with savepath.open("rb") as fp:
            state = torch.load(fp)
    else:
        model = attention(modeltype, emb_size, dimout = 2, L = 3).to(device)
        optim = torch.optim.Adam(params = model.parameters(), lr = lr)
        state = State(model, optim)
    
    loss = nn.CrossEntropyLoss()
    
    # Phase d'entraînement
    for epoch in range(n_epochs):
        
        loss_train = []
        acc_train = []
        
        for x, y, l in train_data:
            
            # Remise à zéro des gradients
            state.optim.zero_grad()
            
            # On met les données sur device puis l'on exécute le modèle
            y = y.to(device)
            yhat = state.model(x)
            
            ltrain = loss(yhat, y)
            ltrain.backward()
            state.optim.step()
            state.iteration += 1
            
            # Mise à jour de la loss et de l'accuracy
            loss_train.append(ltrain.item())
            acc_train.append( torch.where(yhat.argmax(dim=1) == y, 1, 0).sum().item() / len(y) )
        
        # Phase de test
        loss_test = []
        acc_test = []
        
        with torch.no_grad():
            
            for x_, y_, l_ in test_data:
            
                # On met les données sur device puis l'on exécute le modèle
                y_ = y_.to(device)
                yhat_ = state.model(x_)
                ltest = loss(yhat_, y_)
                
                # Mise à jour de la loss et de l'accuracy
                loss_test.append(ltest.item())
                acc_test.append( torch.where(yhat_.argmax(dim=1) == y_, 1, 0).sum().item() / len(y_) )
        
        # Sauvegarde de la loss
        
        writer.add_scalars(attention.__name__ + '/Loss', {'train': np.mean(loss_train), 'test': np.mean(loss_test)}, epoch)
        writer.add_scalars(attention.__name__ + '/Accuracy',{'train': np.mean(acc_train), 'test': np.mean(acc_test)}, epoch)
        
        # Affichage de la loss
        print('Epoch {} | Train loss: {} | Train accuracy: {} | Test loss: {} | Test accuracy: {}'. format(epoch, np.mean(loss_train), np.mean(acc_train), np.mean(loss_test), np.mean(acc_test)))
        
        # Mise à jour du state
        with savepath.open ("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)