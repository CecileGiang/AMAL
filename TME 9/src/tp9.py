import logging
import re
from pathlib import Path
#from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch.utils.tensorboard import SummaryWriter
import datetime

from sklearn.metrics import accuracy_score
from scipy.special import softmax


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
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

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

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

#########################################################################

word2id, embeddings, train, test = get_imdb_data(embedding_size=50)

class Attention(nn.Module):
    def __init__(self, embeddings):
        super(Attention, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.embed_size = embeddings.shape[1]

    def forward(self, x):
        pass

class BasedModel(Attention):
    def __init__(self, embeddings):
        super(BasedModel, self).__init__(embeddings)
        self.model = nn.Sequential( nn.Linear(self.embed_size, 2),
                                                nn.ReLU())

    def forward(self, x):
        embed_x = self.embed(x)
        embed_x = embed_x.mean(dim=1)
        return self.model(embed_x)

class SimpleModel (BasedModel) :
    def __init__(self, embeddings):
        super(SimpleModel, self).__init__(embeddings)
        self.query = nn.Linear(self.embed_size, 1)

    def forward (self, x) :
        embed_x = self.embed(x)
        query_x = self.query(embed_x)
        log_p = 0 + torch.einsum('ijk,ijk->ij', query_x, embed_x).unsqueeze(2)
        p = torch.softmax(log_p, dim=1)
        t_hat = torch.einsum('ijk, ijq-> iq', p, embed_x)
        return self.model(t_hat)

class Model_Q (BasedModel) :
    def __init__(self, embeddings):
        super(Model_Q, self).__init__(embeddings)
        self.query = nn.Linear(self.embed_size, self.embed_size)

    def forward (self, x) :
        embed_x = self.embed(x)
        query_x = self.query(embed_x.mean(dim=1)).unsqueeze(2)
        log_p = 0 + torch.einsum('ijk,ikq->ij',embed_x , query_x).unsqueeze(2)
        p = torch.softmax(log_p, dim=1)
        t_hat = torch.einsum('ijk, ijq-> ik', embed_x, p)
        return self.model(t_hat)


class Model_QV (Model_Q) :
    def __init__(self, embeddings):
        super(Model_QV, self).__init__(embeddings)
        hidden_size = 20
        self.value = nn.Linear(self.embed_size, hidden_size)
        self.model = nn.Sequential( nn.Linear(hidden_size, 2),
                                                nn.ReLU())
    def forward (self, x) :
        embed_x = self.embed(x)
        query_x = self.query(embed_x.mean(dim=1)).unsqueeze(2)
        value_x = self.value(embed_x)
        log_p = 0 + torch.einsum('ijk,ikq->ij',embed_x , query_x).unsqueeze(2)
        p = torch.softmax(log_p, dim=1)
        t_hat = torch.einsum('ijk, ijq-> ik', value_x, p)
        return self.model(t_hat)


BATCH_SIZE = 64

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def collate_fn(bd):
    data = [torch.tensor(b[0]) for b in bd]
    labels =[b[1] for b in bd]
    return nn.utils.rnn.pad_sequence(data, batch_first = True),  torch.tensor(labels)

data_train = DataLoader(train, collate_fn = collate_fn, shuffle = True, batch_size = BATCH_SIZE, drop_last=True)
data_test = DataLoader(test, collate_fn = collate_fn, shuffle = True, batch_size = BATCH_SIZE, drop_last=True)


NB_EPOCHS = 100

def attention (model, model_name, train, test, embeddings, learning_rate=0.001) :
    model = model(embeddings).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    for epoch in range (NB_EPOCHS) :
        loss_train, loss_test  = list(), list()
        accuracy_train, accuracy_test = list(), list()
        for x,y in train :
            x = x.to(device)
            y = y.to(device)
            output = model.forward(x)
            loss_ = loss(output, y)
            loss_train.append(loss_.item())
            yhat = np.argmax(softmax(output.cpu().detach().numpy(), axis=1), axis=1)
            accuracy_train.append(accuracy_score(yhat, y.cpu().detach().numpy()))
            optim.zero_grad()
            loss_.backward()
            optim.step()

        with torch.no_grad():

            for x_test,y_test in test :
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                output_test = model.forward(x_test)
                loss_test.append(loss(output_test, y_test).item())
                yhat_test = np.argmax(softmax(output_test.cpu().detach().numpy(), axis=1), axis=1)
                accuracy_test.append(accuracy_score(yhat_test, y_test.cpu().detach().numpy()) )


        loss_train_batch = np.mean(loss_train)
        loss_test_batch = np.mean(loss_test)

        accuracy_train_batch = np.mean(accuracy_train)
        accuracy_test_batch = np.mean(accuracy_test)

        print("Epoch {} | Train loss = {:.5} | Train accuracy = {:.5} | Test loss = {:.5} | Test accuracy = {:.5}" . format(epoch, loss_train_batch, accuracy_train_batch, loss_test_batch, accuracy_test_batch))

# =============================================================================
#         with open(model_name+'.txt', 'a+') as res_file :
#             res_file.write("Epoch {} | Train loss = {:.5} | Train accuracy = {:.5} | Test loss = {:.5} | Test accuracy = {:.5}" . format(epoch, loss_train_batch, accuracy_train_batch, loss_test_batch, accuracy_test_batch))
#
#         writer.add_scalar('Loss/train_{}'.format(model_name), loss_train_batch, epoch)
#         writer.add_scalar('Accuracy/train_{}'.format(model_name), accuracy_train_batch, epoch)
#         writer.add_scalar('Loss/test_{}'.format(model_name), loss_test_batch, epoch)
#         writer.add_scalar('Accuracy/test_{}'.format(model_name), accuracy_test_batch, epoch)
#
#
# =============================================================================
