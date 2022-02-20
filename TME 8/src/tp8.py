import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
import sentencepiece as spm

from tp8_preprocess import TextDataset

import numpy as np 
import matplotlib.pyplot as plt

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)

test = loaddata("test")
train0 = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train0) - val_size
train, val = torch.utils.data.random_split(train0, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


#  TODO: 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Algorithme trivial qui renvoie systématiquement la classe majoritaire
# =============================================================================

class ModelMajoritaire(nn.Module):
    def __init__(self, nb_class, classe_major):
        super().__init__()
        self.classe_major = classe_major
        self.nb_class = nb_class
        
    def forward(self, inp):
        res = torch.zeros(inp.shape[0], self.nb_class)
        res[:, self.classe_major] = 1
        return res
    
def runMajoritaire(n_epochs=500) :
    major_class = torch.argmax(torch.unique(train0.labels, return_counts=True)[1]).item()
    model = ModelMajoritaire(nb_class=2, classe_major=major_class)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Nombre itérations/batchs pour une epoch :", len(train_iter),'\n')
    train_loss_batch, train_acc_batch = [], []
    val_loss_batch, val_acc_batch = [], []
    test_loss_batch, test_acc_batch = [], []
    
    for epoch in range(n_epochs):
        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        
        # Train
        loop = tqdm(enumerate(train_iter), total=len(train_iter), leave=False)
        for i, (x, y) in loop :     
            # Forward
            output = model(x)
            loss = criterion(output, y)
            
            # Backward
            train_loss.append(loss.item())
            
            # Accuracy
            acc = torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y).item()/len(y)
            train_acc.append(acc*100)
            
            # Update progress bar
            loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop.set_postfix(loss = loss.item(), acc = acc*100)
            
        # Validation     
        for x_val, y_val in val_iter :
            output = model(x_val)
            loss = criterion(output, y_val)
            val_loss.append(loss.item())
            
            acc = torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y_val).item()/len(y_val)
            val_acc.append(acc*100)
        
        train_loss_batch.append(np.mean(train_loss))
        train_acc_batch.append(np.mean(train_acc))
        val_loss_batch.append(np.mean(val_loss))
        val_acc_batch.append(np.mean(val_acc))
        
        print(f'Epoch {epoch} :\n'
              f'\tTrain loss = {train_loss_batch[-1]} | Train acc = {train_acc_batch[-1]}%\n'
              f'\tVal loss   = {val_loss_batch[-1]} | Val acc = {val_acc_batch[-1]}%\n')
        
    #Test    
    for x_test, y_test in test_iter:
        output = model(x_test)
        loss = criterion(output, y_test)
        test_loss_batch.append(loss.item())
        
        acc = torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y_test).item()/len(y_test)
        test_acc_batch.append(acc*100)
        
    print(f"Test loss  = {np.mean(test_loss_batch)} | Test acc = {np.mean(test_acc_batch)}%\n")
     
    return (model, (train_loss_batch, train_acc_batch), 
                    (val_loss_batch, val_acc_batch), 
                    (test_loss_batch, test_acc_batch))
    

# =============================================================================
# Architecture convolutives
# =============================================================================
         
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, in_channels)
        
        self.conv = nn.Sequential(
            # nn.Conv1d(in_channels, out_channels, kernel_size, stride),
            nn.Conv1d(in_channels, 2*out_channels, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size),
            nn.Conv1d(2*out_channels, out_channels, kernel_size, stride),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), # Global Max Pooling
            torch.nn.Flatten() # = squeeze()
        )
        
        self.linear = nn.Linear(out_channels, 2)

    def forward(self, inp):
        x = self.embedding(inp)     # (Batch, seq_length, in_channels)
        x = x.permute((0,2,1))      # (Batch, in_channels, seq_length)
        x = self.conv(x)            # (Batch, out_channels)
        return self.linear(x)       # (Batch, 2)


def run(in_chan=10, out_chan=10, kernel_size=3, stride=1, n_epochs=500, lr=1e-3) :
    
    model = CNN(in_chan, out_chan, kernel_size, stride).to(device)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Nombre itérations/batchs pour une epoch :", len(train_iter),'\n')
    train_loss_batch, train_acc_batch = [], []
    val_loss_batch, val_acc_batch = [], []
    test_loss_batch, test_acc_batch = [], []
    
    for epoch in range(n_epochs):
        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        
        # Train
        model.train()
        loop = tqdm(enumerate(train_iter), total=len(train_iter), leave=False)
        for i, (x, y) in loop :   
            # Remise à zéro de l'optimiseur
            optim.zero_grad()
            
            # Transfert sur le GPU
            x = x.to(device)
            y = y.to(device)
            
            # Forward
            output = model(x)
            loss = criterion(output, y)
            
            # Backward
            train_loss.append(loss.item())
            loss.backward()
            optim.step()
            
            # Accuracy
            acc = torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y).item()/len(y)
            train_acc.append(acc*100)
    
            # Update progress bar
            loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop.set_postfix(loss = loss.item(), acc = acc*100)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_iter :
                
                # Transfert sur le GPU
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                output = model(x_val)
                loss = criterion(output, y_val)
                val_loss.append(loss.item())
                
                # Accuracy
                acc = torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y_val).item()/len(y_val)
                val_acc.append(acc*100)
            
        train_loss_batch.append(np.mean(train_loss))
        train_acc_batch.append(np.mean(train_acc))
        val_loss_batch.append(np.mean(val_loss))
        val_acc_batch.append(np.mean(val_acc))
        
        print(f'Epoch {epoch} :\n'
              f'\tTrain loss = {train_loss_batch[-1]} | Train acc = {train_acc_batch[-1]}%\n'
              f'\tVal loss   = {val_loss_batch[-1]} | Val acc = {val_acc_batch[-1]}%\n')
    
    # Test
    for x_test, y_test in test_iter:
        
        # Transfert sur le GPU
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        output = model(x_test)
        loss = criterion(output, y_test)
        test_loss_batch.append(loss.item())
        
        # Accuracy
        acc = torch.sum(torch.argmax(torch.softmax(output, 1), 1) == y_test).item()/len(y_test)
        test_acc_batch.append(acc*100)
        
    print(f"Test loss  = {np.mean(test_loss_batch)} | Test acc = {np.mean(test_acc_batch)}%\n")

    return (model, (train_loss_batch, train_acc_batch), 
                    (val_loss_batch, val_acc_batch), 
                    (test_loss_batch, test_acc_batch))

# =============================================================================
# Apprentissage et test
# =============================================================================

# Paramètres
in_chan=10
out_chan=16
kernel_size = 3
stride = 1

EPOCHS = 1
lr = 1e-3

# Runs
# Algorithme majoritaire
# model_0, train_lists_0, val_lists_0, test_lists_0 = runMajoritaire(n_epochs = EPOCHS)

# CNN 
model, train_lists, val_lists, test_lists = run(in_chan, out_chan, kernel_size, stride, EPOCHS, lr)


# =============================================================================
# Autres
# =============================================================================

# # First convolutional layer : dim = (2*out_chan, in_chan, kernel_size)
# conv_weights = model.conv[0].weight

# # Sum on kernel_size dimension : dim = (2*out_chan, in_chan)
# w = torch.sum(conv_weights, 2).detach().numpy()
# print(w.shape)
# print(w)
