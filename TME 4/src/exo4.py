import string
import unicodedata
import torch
import sys
import datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]

BATCH_SIZE = 32

N_EPOCHS = 100
#Chargement des données

text_file = fread('../data/trump_full_speech.txt')

data_dataset = TrumpDataset(text=text)
train_len = round(data_dataset.__len__()*0.7)
test_len = data_dataset.__len__() - train_len

text_train = ""
text_test = ""

for i, phrase in enumerate (data_dataset.phrases) :
    if(i<train_len) :
        text_train += phrase
    else:
        text_test += phrase

train_dataset = TrumpDataset(text=text_train)
test_dataset = TrumpDataset(text=text_test)
train = DataLoader(train_dataset, shuffle = False, batch_size = BATCH_SIZE,  drop_last=True)
test = DataLoader (test_dataset, shuffle = False, batch_size = BATCH_SIZE, drop_last = True)

def generator(input_dim, latent_dim, output_dim, n_epochs = N_EPOCHS):
    # Pour l'affichage aec tensorboard
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Création du modèle et de l'optimiseur, chargemet sur device
    model = RNN(input_dim, latent_dim, input_dim)
    optim = torch.optim.Adam(params = model.parameters())








