#! /usr/bin/python3
"""
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
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class Metrics:
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
    def __init__(self, model, path, interval):
        self.model = model
        self.path = path
        self.interval = interval

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.interval == 0:
            file_name = os.path.join(self.path, 'model.{:03d}.pth'.format(epoch))
            torch.save(self.model.state_dict(), file_name)

class LeafClassifier(nn.Module):
    
    def __init__(self, hidden_size, num_layers, dropout):
        super(LeafClassifier, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-multilingual-cased')
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = num_layers
        self.config.hidden_dropout_prob = dropout
        self.bert = BertForSequenceClassification(config=self.config)
        self.bert.resize_token_embeddings(len(tokenizer))
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        temp = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return nn.Softmax(dim=1)(temp.logits)

    
    def train1(self, train_loader, optimizer, loss_fn, epochs=20, device='cpu'):
        print("train loader length : ",len(train_loader))
        self.bert.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (label, data) in enumerate(train_loader):
                print("batch_idx : ", batch_idx)
                encoded_input=tokenizer(data[batch_idx], padding=True, truncation=True, return_tensors="pt")
                label = label.to(torch.float32)
                output = self.forward(**encoded_input)
                print("-------shape-------")
                print(output.shape)
                print(label.shape)
                print("-------shape-------")
                loss = loss_fn(output, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
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
            padding_token = 0
            # attention_mask = np.array([[[1 if token != padding_token else 0 for token in data[i][j]] for j in range(len(data[i]))] for i in range(len(data))])
            # attention_mask = torch.tensor(attention_mask)
            attention_mask = torch.where(data != 0, torch.tensor(1), torch.tensor(0))
            output = self.forward(data, attention_mask).squeeze()
            # output = self.model.forward(input=data, attention_mask=attention_mask).squeeze()
            
            # output = self.model.forward(data, attention_mask).squeeze()
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
"""
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
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import time

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

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
    def __init__(self, hidden_size, num_layers, dropout, dense_size):
        super(LeafClassifier, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-multilingual-cased')
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = num_layers
        self.config.hidden_dropout_prob = dropout
        self.bert = BertForSequenceClassification(config=self.config)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        temp = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        temp1 = self.sigmoid(temp.logits)
        print(temp.shape)
        print(temp1.shape)
        return temp1
    
def train1(clf,train_loader, optimizer, loss_fn, epochs=20, device='cpu'):
    clf.train()
    print("train loader length : ",len(train_loader))
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            print("batch idx : ", batch_idx)
            encoded_input=tokenizer(data[batch_idx], padding=True, truncation=True, return_tensors="pt")
            optimizer.zero_grad()
            label = torch.tensor(label, dtype=torch.int32)
            print("[--------------------DATA-----------------------]")
            encoded_input['labels'] = label[batch_idx]
            print(encoded_input.keys())
            print("[--------------------DATA-----------------------]")
            output = clf.forward(**encoded_input)
            print(output.shape)
            print(label.shape)
            loss = loss_fn(output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    return train_loss

def evaluate(clf,train_loader, device):
    clf.eval()
    result0 = 0
    total0 = 0
    result1 = 0
    total1 = 0
    for data, label in enumerate(train_loader):
        data = data.to(device)
        data = data.to(torch.float32)
        label = label.to(torch.float32).squeeze()
        padding_token = 0
        attention_mask = torch.where(data != 0, torch.tensor(1), torch.tensor(0))
        output = clf.forward(data, attention_mask).squeeze()
        total1 += sum(label)
        total0 += len(label) - sum(label)
        for i in range(len(label)):
            if output[i] >= 0.5 and label[i] == 1:
                result1 += 1
            elif output[i] <= 0.5 and label[i] == 0:
                result0 += 1
    return (result0 / total0 * 100, result1 / total1 * 100)
