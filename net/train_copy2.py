
import argparse
import os
import pickle
import csv
import math
import numpy as np

# import tensorflow as tf
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.utils import class_weight

from leaf_classifier import LeafClassifier
from leaf_classifier import train1

from pprint import pprint
from time import sleep

class CustomDataset(Dataset):
    def __init__(self, npy_file):
        data = np.load(npy_file, allow_pickle=True)
        self.htmls = data[0]['html_list']
        self.labels = data[0]['label_list']
        print(len(self.htmls))
        print(len(self.labels))
        
    def __len__(self):
        return len(self.htmls)

    def __getitem__(self, idx):
        return self.htmls[idx], self.labels[idx]

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    # (xx, yy, zz) = zip(*batch)
    # 텐서 변환을 하지 않고 pad_sequence에 리스트를 전달합니다.
    # xx = [torch.tensor(np.array(x)) for x in xx]
    yy = [torch.tensor(np.array(y)) for y in yy]
    # pad_sequence는 텐서의 배치를 입력받아 동일한 길이로 패딩합니다.
    # xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    # zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)

    return xx, yy_pad

def get_dataset(npy_file, batch_size, repeat=True):
    dataset = CustomDataset(npy_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # DataLoader의 내용
    # 1. feature
    # 2. lable
    # 3. bert_input

    return dataloader



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DATA_DIR', help='Directory of files produced by the preprocessing script')
    ap.add_argument('-l', '--num_layers', type=int, default=2, help='The number of RNN layers')
    ap.add_argument('-u', '--hidden_units', type=int, default=96,
                    help='The number of hidden LSTM units')
    ap.add_argument('-d', '--dropout', type=float, default=0.5, help='The dropout percentage')
    ap.add_argument('-s', '--dense_size', type=int, default=64, help='Size of the dense layer')
    ap.add_argument('-e', '--epochs', type=int, default=20, help='The number of epochs')
    ap.add_argument('-b', '--batch_size', type=int, default=1, help='The batch size')
    ap.add_argument('--interval', type=int, default=5,
                    help='Calculate metrics and save the model after this many epochs')
    ap.add_argument('--working_dir', default='train', help='Where to save checkpoints and logs')
    args = ap.parse_args()

    info_file = os.path.join(args.DATA_DIR, 'info.pkl')
    with open(info_file, 'rb') as fp:
        info = pickle.load(fp)
        train_steps = math.ceil(info['num_train_examples'] / args.batch_size)

    # train_set_file = os.path.join(args.DATA_DIR, 'train.tfrecords')
    train_set_file = os.path.join(args.DATA_DIR, 'train.npy')
    train_dataset = get_dataset(train_set_file, args.batch_size)
    
    # dev_set_file = os.path.join(args.DATA_DIR, 'dev.tfrecords')
    dev_set_file = os.path.join(args.DATA_DIR, 'dev.npy')
    if os.path.isfile(dev_set_file):
        dev_dataset = get_dataset(dev_set_file, 1, repeat=False)
    else:
        dev_dataset = None
    
    # test_set_file = os.path.join(args.DATA_DIR, 'test.tfrecords')
    test_set_file = os.path.join(args.DATA_DIR, 'test.npy')
    if os.path.isfile(test_set_file):
        test_dataset = get_dataset(test_set_file, 1, repeat=False)
    else:
        test_dataset = None

    kwargs = {
        # 'input_size': info['num_words'] + info['num_tags'],
        'hidden_size': args.hidden_units,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'dense_size': args.dense_size}
    clf = LeafClassifier(**kwargs)
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
    train1(clf=clf, train_loader=train_dataset, optimizer=optimizer, loss_fn=nn.BCELoss(), epochs=args.epochs, device='mps')

if __name__ == '__main__':
    main()
