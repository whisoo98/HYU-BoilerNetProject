#! /usr/bin/python3

import argparse
import os
import pickle
import csv
import math
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


class Metrics:
    """Calculate metrics for a dev-/testset and add them to the logs."""
    def __init__(self, clf, data, steps, interval, prefix=''):
        self.clf = clf
        self.data = data
        self.steps = steps
        self.interval = interval
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.interval == 0:
            y_true, y_pred = self.clf.eval(self.data, self.steps, desc=self.prefix)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        else:
            p, r, f, s = np.nan, np.nan, np.nan, np.nan
        logs_new = {'{}_precision'.format(self.prefix): p,
                    '{}_recall'.format(self.prefix): r,
                    '{}_f1'.format(self.prefix): f,
                    '{}_support'.format(self.prefix): s}
        logs.update(logs_new)

class Saver:
    """Save the model."""
    def __init__(self, model, path, interval):
        self.model = model
        self.path = path
        self.interval = interval

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.interval == 0:
            file_name = os.path.join(self.path, 'model.{:03d}.pth'.format(epoch))
            torch.save(self.model.state_dict(), file_name)

class LeafClassifier(nn.Module):
    """This classifier assigns labels to sequences based on words and HTML tags."""
    def __init__(self, input_size, num_layers, hidden_size, dropout, dense_size):
        """Construct the network."""
        super(LeafClassifier, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dense_size = dense_size
        self.model= self._get_model()

    def forward(self, input):
        return self.model(input)

    def _get_model(self):
        model = nn.Transformer(d_model=self.input_size, nhead=4, num_encoder_layers=self.num_layers,
                               num_decoder_layers=self.num_layers, dim_feedforward=self.hidden_size,
                               dropout=self.dropout)
        model.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.dense_size),
            nn.ReLU()
        )
        model.decoder = nn.Sequential(
            nn.Linear(self.input_size, self.dense_size),
            nn.ReLU()
        )
        model.final_layer = nn.Linear(self.dense_size, 1)
        model.sigmoid = nn.Sigmoid()
        model.argmax = nn.MaxPool1d(2)
        return model

    
    def train1(self, train_loader, optimizer, criterion, device):
        self.train()
        train_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            data = data.to(torch.float64)
            label = label.to(torch.float64)
            print(data.dtype)
            print(label.dtype)
            output = self.model.forward(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss