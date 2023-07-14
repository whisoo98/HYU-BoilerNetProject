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
        result = []
        for x in input:
            result.append(self.model(x))
        return torch.tensor(result)

    def _get_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, self.dense_size),
            nn.ReLU(),
            nn.Linear(self.dense_size, self.dense_size),
            nn.ReLU(),
            nn.Linear(self.dense_size, 1),
            nn.Sigmoid()
        )

        return model

    
    def train1(self, train_loader, optimizer, criterion, device):
        self.train()
        train_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            data = data.to(torch.float32)
            label = label.to(torch.float32)
            output = self.model.forward(data).squeeze()
            print(output.shape)
            print(label.shape)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss
    
    def evaluate(self, train_loader, device):
        self.eval()
        result0 = 0
        total0 = 0
        result1 = 0
        total1 = 0
        for data, label in train_loader:
            data = data.to(device)
            data = data.to(torch.float32)
            label = label.to(torch.float32).squeeze()
            output = self.model.forward(data).squeeze()
            # print(label)
            # print(output)
            # total += len(label)
            #print(label.shape)
            #print(output.shape)
            total1 += sum(label)
            total0 += len(label) - sum(label)
            for i in range(len(label)):
                
                if output[i] >= 0.5 and label[i] == 1:
                    result1 += 1
                elif output[i] <= 0.5 and label[i] == 0:
                    result0 += 1
        return (result0 / total0 * 100, result1 / total1 * 100)