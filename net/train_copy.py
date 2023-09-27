#! /usr/bin/python3
# train_copy.py

import argparse
import os
import pickle
import csv
import math
import numpy as np
import random
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler

from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

from leaf_classifier import LeafClassifier

from pprint import pprint
from time import sleep
import preprocess_copy2 as preprocess_copy
from transformers import BertForSequenceClassification,AdamW,get_linear_schedule_with_warmup

class CustomDataset(Dataset):
    def __init__(self, tokenized_html, labels, masks):
        self.tokenized_html = tokenized_html
        self.labels = labels
        self.masks  = masks
        
        print(len(self.tokenized_html))
        print(len(self.labels))
        
    def __len__(self):
        return len(self.tokenized_html)

    def __getitem__(self, idx):
        return self.tokenized_html[idx], self.labels[idx], self.masks[idx]

def pad_collate(batch):
    (xx, yy, zz) = zip(*batch)
    # (xx, yy, zz) = zip(*batch)
    # 텐서 변환을 하지 않고 pad_sequence에 리스트를 전달합니다.
    # xx = [torch.tensor(np.array(x)) for x in xx]
    #xx = [torch.tensor(np.array(x)) for x in xx]
    yy = [torch.tensor(np.array(y)) for y in yy]
    # zz = [torch.tensor(np.array(z)) for z in zz]
    # pad_sequence는 텐서의 배치를 입력받아 동일한 길이로 패딩합니다.
    #xx_pad = pad_sequence(xx, batch_first=False, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=False, padding_value=0)
    # zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)
    # zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)
    
    return xx, yy_pad, zz
    

def build_dataloader(ids,  labels, masks, batch_size):
    # torch_ids = torch.tensor(ids)
    # torch_masks = torch.tensor(masks)
    # torch_lable = torch.tensor(labels)
    dataloader = CustomDataset(ids, labels, masks)
    dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    return dataloader

def build_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    device = torch.device("cpu")
    # device = torch.device("cuda")
    # print(f"{torch.cuda.get_device_name(0)} available")
    model = model

    return model, device
    
def test(test_dataloader, model, device):
    model.eval()
    total_accuracy = 0
    for batch in test_dataloader:
        batch = tuple(index.to(device) for index in batch)
        ids, masks, labels = batch
        with torch.no_grad():
            outputs = model(ids, token_type_ids=None, attention_mask=masks)
        pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
        true = [label for label in labels.cpu().numpy()]
        accuracy = accuracy_score(true, pred)
        total_accuracy += accuracy
    avg_accuracy = total_accuracy/len(test_dataloader)
    print(f"test AVG accuracy : {avg_accuracy: .2f}")
    return avg_accuracy

def train(train_dataloader, test_dataloader, args):
    model, device = build_model()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*args.epochs)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model.zero_grad()
    
    for epoch in range(0, args.epochs):
        model.train()
        total_loss, total_accuracy = 0, 0
        print("-"*30)
        for step, batch in enumerate(train_dataloader):
            if step % 500 == 0 :
                print(f"Epoch : {epoch+1} in {args.epochs} / Step : {step}")

            # batch = tuple(index.to(device) for index in batch)
            ids, masks, labels = batch
            print(ids)
            temp = torch.tensor(ids[0])

            outputs = model(temp, token_type_ids=None, attention_mask=masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
            true = [label for label in labels.cpu().numpy()]
            accuracy = accuracy_score(true, pred)
            total_accuracy += accuracy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_accuracy/len(train_dataloader)
        print(f" {epoch+1} Epoch Average train loss :  {avg_loss}")
        print(f" {epoch+1} Epoch Average train accuracy :  {avg_accuracy}")

        acc = test(test_dataloader, model, device)
        print("Epoch {0} Accuracy :{1}".format(epoch,acc))
        # os.makedirs("results", exist_ok=True)
        # f = os.path.join("results", f'epoch_{epoch+1}_evalAcc_{acc*100:.0f}.pth')
        # torch.save(model.state_dict(), f)
        # print('Saved checkpoint:', f)
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DATA_DIR', help='Directory of files produced by the preprocessing script')
    ap.add_argument('--dirs', nargs='+', help='A list of directories containing the HTML files')
    
    ap.add_argument('-l', '--num_layers', type=int, default=2, help='The number of RNN layers')
    ap.add_argument('-u', '--hidden_units', type=int, default=96,
                    help='The number of hidden LSTM units')
    ap.add_argument('-d', '--dropout', type=float, default=0.5, help='The dropout percentage')
    ap.add_argument('-e', '--epochs', type=int, default=20, help='The number of epochs')
    ap.add_argument('-b', '--batch_size', type=int, default=32, help='The batch size')
    ap.add_argument('-s', '--split_dir', help='Directory that contains train-/dev-/testset split')
    ap.add_argument('--interval', type=int, default=5,
                    help='Calculate metrics and save the model after this many epochs')
    ap.add_argument('--working_dir', default='train', help='Where to save checkpoints and logs')
    ap.add_argument('--save', default='result', help='Where to save the results')
    args = ap.parse_args()

    train_set, dev_set, test_set = preprocess_copy.preprocess(args)
    train_dataloader = build_dataloader(train_set[0], train_set[1],  train_set[2], args.batch_size)
    test_dataloader = build_dataloader(test_set[0], test_set[1], test_set[2], args.batch_size)
    train(train_dataloader, test_dataloader, args)

#     info_file = os.path.join(args.DATA_DIR, 'info.pkl')
#     with open(info_file, 'rb') as fp:
#         info = pickle.load(fp)
#         train_steps = math.ceil(info['num_train_examples'] / args.batch_size)

#     # train_set_file = os.path.join(args.DATA_DIR, 'train.tfrecords')
#     train_set_file = os.path.join(args.DATA_DIR, 'train.npy')
#     train_dataset = get_dataset(train_set_file, args.batch_size)
    
#     # dev_set_file = os.path.join(args.DATA_DIR, 'dev.tfrecords')
#     dev_set_file = os.path.join(args.DATA_DIR, 'dev.npy')
#     if os.path.isfile(dev_set_file):
#         dev_dataset = get_dataset(dev_set_file, 1, repeat=False)
#     else:
#         dev_dataset = None
    
#     # test_set_file = os.path.join(args.DATA_DIR, 'test.tfrecords')
#     test_set_file = os.path.join(args.DATA_DIR, 'test.npy')
#     if os.path.isfile(test_set_file):
#         test_dataset = get_dataset(test_set_file, 1, repeat=False)
#     else:
#         test_dataset = None

#     class_weights = get_class_weights(train_set_file)
#     print('using class weights {}'.format(class_weights))

#     kwargs = {
#         # 'input_size': info['num_words'] + info['num_tags'],
#         'hidden_size': args.hidden_units,
#         'num_layers': args.num_layers,
#         'dropout': args.dropout,
#         'dense_size': args.dense_size}
#     clf = LeafClassifier(**kwargs)

#     ckpt_dir = os.path.join(args.working_dir, 'ckpt')
#     log_file = os.path.join(args.working_dir, 'train.csv')
#     os.makedirs(ckpt_dir, exist_ok=True)

#     params_file = os.path.join(args.working_dir, 'params.csv')
#     print('writing {}...'.format(params_file))
#     with open(params_file, 'w') as fp:
#         writer = csv.writer(fp)
#         for arg in vars(args):
#             writer.writerow([arg, getattr(args, arg)])

#     # clf.train1(train_dataset, train_steps, args.epochs, log_file, ckpt_dir, class_weights,
#     #           dev_dataset, info.get('num_dev_examples'),
#     #           test_dataset, info.get('num_test_examples'),
#     #           args.interval)
#     optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
#     # clf.train1(train_loader=train_dataset, optimizer=optimizer, loss_fn=nn.BCELoss(), epochs=args.epochs, device='cpu')
    
#     train_loader=train_dataset
#     loss_fn=nn.BCELoss()
#     # epochs=args.epochs
#     epochs=20
#     device='cpu'
    
#     print("train loader length : ",len(train_loader))
#     model = self._get_model(tokenizer)
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch_idx, (data, label) in enumerate(train_loader):
#             print("batch idx : ", batch_idx)
#             encoded_input=tokenizer(data[batch_idx], padding=True, truncation=True, return_tensors="pt")
#             optimizer.zero_grad()
            
#             label = label.to(torch.float32)

#             print("[--------------------DATA-----------------------]")
#             encoded_input['labels'] = label[batch_idx]
#             print(encoded_input.keys())
#             print(len(data))
#             print(len(label))
#             print("[--------------------DATA-----------------------]")
#             output = model.forward(encoded_input)
#             loss = loss_fn(output, label)
#             train_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
#     return train_loss

# def evaluate(self, train_loader, device):
#     self.eval()
#     result0 = 0
#     total0 = 0
#     result1 = 0
#     total1 = 0
#     for data, label in train_loader:
#         data = data.to(device)
#         data = data.to(torch.float32)
#         label = label.to(torch.float32).squeeze()
#         padding_token = 0
#         attention_mask = torch.where(data != 0, torch.tensor(1), torch.tensor(0))
#         output = self.forward(data, attention_mask).squeeze()
#         total1 += sum(label)
#         total0 += len(label) - sum(label)
#         for i in range(len(label)):
            
#             if output[i] >= 0.5 and label[i] == 1:
#                 result1 += 1
#             elif output[i] <= 0.5 and label[i] == 0:
#                 result0 += 1
#     return (result0 / total0 * 100, result1 / total1 * 100)

if __name__ == '__main__':
    main()