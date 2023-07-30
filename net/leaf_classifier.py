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
from transformers import BertModel, BertConfig, BertTokenizer

import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

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
    """This classifier assigns labels to sequences based on words and HTML tags."""
    # def __init__(self, input_size, num_layers, hidden_size, dropout, dense_size):
    #     """Construct the network."""
    #     super(LeafClassifier, self).__init__()
    #     self.input_size = input_size
    #     self.num_layers = num_layers
    #     self.hidden_size = hidden_size
    #     self.dropout = dropout
    #     self.dense_size = dense_size
    #     self.model= self._get_model()

    # def forward(self, input):
    #     result = []
    #     for x in input:
    #         result.append(self.model(x))
    #     return torch.tensor(result)
    
    # def forward(self, input, attention_mask):
    #     result = []
    #     for x, mask in zip(input, attention_mask):
    #         result.append(self.model(x, mask))
    #     return torch.cat(result, dim=0)

    # def _get_model(self):
    #     model = nn.Sequential(
    #         nn.Linear(self.input_size, self.dense_size),
    #         nn.ReLU(),
    #         nn.Linear(self.dense_size, self.dense_size),
    #         nn.ReLU(),
    #         nn.Linear(self.dense_size, 1),
    #         nn.Sigmoid()
    #     )
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, dense_size):
        super(LeafClassifier, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = num_layers
        self.bert = BertModel(config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_size, dense_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(dense_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, attention_mask):
        outputs = self.bert(inputs_embeds=hidden_states, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        x = self.relu(self.classifier(pooled_output))
        logits = self.output_layer(x)
        return self.sigmoid(logits)
    
    # def __init__(self,input_size, num_layers, hidden_size, dropout, dense_size):
    #     super(LeafClassifier, self).__init__()
    #     self.input_size = input_size
    #     self.num_layers = num_layers
    #     self.hidden_size = hidden_size
    #     self.dropout = dropout
    #     self.dense_size = dense_size
    #     # self.model = self._get_model(tokenizer)
    #     self.bert_model = self._get_bert_model()
    #     self.classifier = self._get_classifier()

    # def forward(self, input, attention_mask):
    #     result = []
    #     for x, mask in zip(input, attention_mask):
    #         # outputs = self.model(x, mask)
    #         # pooled_output = outputs.pooler_output
    #         self.model(x, mask)
    #         result.append(self.model(x, mask))
    #     return torch.cat(result, dim=0)
    # def forward(self, input_ids, attention_mask):
    #     outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
    #     pooled_output = outputs.pooler_output
    #     result = self.classifier(pooled_output)
    #     return result
    
    # def _get_bert_model(self):
    #     bert_config = BertConfig.from_pretrained('bert-base-multilingual-cased')
    #     bert_model = BertModel(config=bert_config)
    #     bert_model.resize_token_embeddings(len(tokenizer))  # 추가: tokenizer에 맞게 토큰 임베딩 크기 조정
        
    #     return bert_model    

    def _get_classifier(self):
        classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.dense_size),
            nn.ReLU(),
            nn.Linear(self.dense_size, self.dense_size),
            nn.ReLU(),
            nn.Linear(self.dense_size, 1),
            nn.Sigmoid()
        )
        return classifier
    
    # def _get_model(self,tokenizer):
    #     bert_config = BertConfig.from_pretrained('bert-base-multilingual-cased')
    #     bert_model = BertModel(config=bert_config)
    #     bert_model.resize_token_embeddings(len(tokenizer))  # 추가: tokenizer에 맞게 토큰 임베딩 크기 조정
    #     classifier = nn.Sequential(
    #         nn.Dropout(self.dropout),
    #         nn.Linear(self.hidden_size, self.dense_size),
    #         nn.ReLU(),
    #         nn.Linear(self.dense_size, self.dense_size),
    #         nn.ReLU(),
    #         nn.Linear(self.dense_size, 1),
    #         nn.Sigmoid()
    #     )
    #     model = nn.Sequential(
    #         bert_model,
    #         classifier
    #     )
    #     return model
    
    def train1(self, train_loader, optimizer, loss_fn, epochs=20, device='cpu'):
        print(len(train_loader))
        for _ in range(epochs):
            self.train()
            train_loss = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                print(data.size())
                data = data.to(device)
                optimizer.zero_grad()
                
                data = data.to(torch.float32)
                print(data.size())
                
                label = label.to(torch.float32)

                # train_data = torch.tensor(data[batch_idx])
                train_data = torch.tensor(data)
                attention_mask = torch.where(train_data != 0, torch.tensor(1), torch.tensor(0))
                attention_mask = torch.tensor(attention_mask)
                print("[--------------------DATA-----------------------]")
                print("train_data dimension : ",train_data.dim(), "train_data shape : ", train_data.shape, "train_data size : ", train_data.size())
                print("attention_mask dimension : ",attention_mask.dim(), "attention_mask shape : ", attention_mask.shape)
                print("[--------------------DATA-----------------------]")
                # TODO : data 가 2차원이어야함 -> torch.Size()했을 때 2개만 나와야한다.
                # torch.Size([batch_size, sequence_length, embedding_dim])
                # output = self.model.forward(input=data, attention_mask=attention_mask).squeeze()
                output = self.forward(train_data, attention_mask).squeeze()
                print(output.shape)
                print(label.shape)
                loss = loss_fn(output, label)
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