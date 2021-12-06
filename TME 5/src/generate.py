#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  25 10:07:28 2021

@author: Cécile GIANG
"""

###################################################################################################
# ---------------------------- IMPORTATION DES MODULES ET LIBRAIRIES ---------------------------- #
###################################################################################################


from textloader import  string2code, code2string, id2lettre

import numpy as np
import math
import torch

# On utilise le GPU s'il est disponible, le CPU sinon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###################################################################################################
# ----------------------------------- FONCTIONS DE GENERATION ----------------------------------- #
###################################################################################################


def generate(model, emb, decoder, latent, log_softmax, eos, start = "", maxlen = 200, C = False):
    """ Fonction de génération (l'embedding et le decodeur doivent être des fonctions du RNN).
        Initialise le réseau avec start (ou à 0 si start est vide), et génère une séquence à partir du 
        RNN et d'une fonction decoder qui renvoie des logits (logarithme de probabilité à une constante 
        près, i.e. ce qui vient avant le softmax) des différentes sorties possibles. 
        La génération s'arrête quand la séquence atteint une longueur de maxlen, ou quand eos est généré.
        @param model: le réseau
        @param emb: la couche d'embedding
        @param decoder: le décodeur
        @param latent: dimension de l'espace des états cachés (RNN)
        @param log_softmax: fonction log softmax pour le calcul des distributions
        @param eos: ID du token end of sequence
        @param start: début de la phrase
        @param maxlen: longueur maximale
        @param C: bool, vaut True dans le cas d'un réseau LSTM
    """
    if start == "" :
        raise ValueError('Starting sequence cannot be empty.')
        
    else :
        
        # Embedding de la séquence de départ start
        x = string2code(start).to(device)
        x_emb = emb(x).to(device)
        
        # Forward sur le modèle RNN et calcul des états externe/interne
        h = torch.zeros(1, latent, dtype = torch.float).to(device)
        
        if C:
            Ct = torch.zeros(1, latent, dtype = torch.float).to(device)
            ht = model.forward(x_emb, h, Ct)[-1]
        else:
            ht = model.forward(x_emb, h)[-1]
        
        # Calcul du premier élément de la séquence
        distribution = log_softmax( decoder(ht) )
        output = torch.argmax(distribution, axis = 1)
        
        # Initialisation de la séquence générée et de sa taille
        gen_sequence = [output]
        gen_len = 1
        
        while gen_sequence[-1] != eos and len(gen_sequence) < maxlen:
            
            if C:
                ht, Ct = model.one_step(emb(gen_sequence[-1]), ht, Ct)
            else:
                ht = model.one_step(emb(gen_sequence[-1]), ht)
                
            distribution = log_softmax( decoder(ht) )
            output = torch.argmax(distribution, axis = 1)
            
            gen_sequence.append(output)
            
    gen_sequence = torch.Tensor(gen_sequence)
    
    return start + ' --> ' + start + code2string(gen_sequence)


def generate_beam(model, emb, decoder, latent, log_softmax, eos, k, start="", maxlen = 200, C = False):
    """ Génère une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles 
        les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés 
        (au sens de la vraisemblance) pour l'itération suivante.
        @param model: le réseau
        @param emb: la couche d'embedding
        @param decoder: le décodeur
        @param latent: dimension de l'espace des états cachés (RNN)
        @param log_softmax: fonction log softmax pour le calcul des distributions
        @param eos: ID du token end of sequence
        @param k: int, le paramètre du beam search
        @param start: début de la phrase
        @param maxlen: longueur maximale
        @param C: bool, vaut True dans le cas d'un réseau LSTM
    """
    if start == "" :
        raise ValueError('Starting sequence cannot be empty.')
        
    else :
        
        # Embedding de la séquence de départ start
        x = string2code(start).to(device)
        x_emb = emb(x).to(device)
        
        # Forward sur le modèle RNN et calcul des états externe/interne
        h = torch.zeros(1, latent, dtype = torch.float).to(device)
        
        if C:
            Ct = torch.zeros(1, latent, dtype = torch.float).to(device)
            ht = model.forward(x_emb, h, Ct)[-1]
        else:
            ht = model.forward(x_emb, h)[-1]
        
        # Calcul de la distribution sur la séquence
        distribution = log_softmax( decoder(ht) ).squeeze()
        
        # Score initial: somme des scores de chaque lettre de la séquence
        score = sum([ distribution[i] for i in x ])
        
        # Calcul des k séquences candidates les plus probables
        all_candidates = [ ( x.tolist() + [code], score + proba ) for code, proba in enumerate(distribution) ]
        all_candidates = sorted(all_candidates, key=lambda x: x[1])[-k:]
        
        # Beam Search
        
        for i in range(1, maxlen):
            # Liste des nouvelles séquences candidates pour la séquence considérée
            new_candidates = []
            
            for x, score in all_candidates:
                
                # Embedding de la séquence candidate
                x_emb = emb(torch.Tensor(x).long()).to(device)
                
                # Forward sur le modèle RNN et calcul des états externe/interne
                h = torch.zeros(1, latent, dtype = torch.float).to(device)
                
                if C:
                    Ct = torch.zeros(1, latent, dtype = torch.float).to(device)
                    ht = model.forward(x_emb, h, Ct)[-1]
                else:
                    ht = model.forward(x_emb, h)[-1]
                
                distribution = log_softmax( decoder(ht) ).squeeze()
                
                # Calcul des nouvelles séquences candidates
                new_candidates += [ ( x + [code], score + proba ) for code, proba in enumerate(distribution) ]
            
            # On garde les k séquences candidates les pmus probables
            all_candidates = sorted(new_candidates, key=lambda x: x[1])[-k:]
        
        # Affichage des k séquences candidates les plus probables
        for candidate in all_candidates:
            text = candidate[0]
            eos = np.where( np.array(text) == 0 )[0][0] if 0 in text else len(text)
            print( start + ' --> ' + code2string( text[:eos] ) + '.' )
        
        return all_candidates


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute
