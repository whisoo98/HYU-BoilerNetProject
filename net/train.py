#! /usr/bin/python3


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

from pprint import pprint
from time import sleep

class CustomDataset(Dataset):
    def __init__(self, npy_file):
        data = np.load(npy_file, allow_pickle=True)
        self.features = data[0]['doc_feature_list']
        self.labels = data[0]['doc_label_list']
        self.bert_input = data[0]['bert_input']
        print(len(self.features[0]))
        print(len(self.labels[0]))
        print(len(self.bert_input[0]))
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.bert_input[idx]

def pad_collate(batch):
    (xx, yy, zz) = zip(*batch)
    # 텐서 변환을 하지 않고 pad_sequence에 리스트를 전달합니다.
    xx = [torch.tensor(np.array(x)) for x in xx]
    yy = [torch.tensor(np.array(y)) for y in yy]
    # pad_sequence는 텐서의 배치를 입력받아 동일한 길이로 패딩합니다.
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, zz

def get_dataset(npy_file, batch_size, repeat=True):
    dataset = CustomDataset(npy_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # DataLoader의 내용 출력
    for batch in dataloader:
        # input_ids, attention_mask= batch
        input_ids, attention_mask, labels = batch
        print("Input IDs:", input_ids)
        print("Attention Masks:", attention_mask)
        print("Labels:", labels)
        
    return dataloader

# def get_dataset(dataset_file, batch_size, repeat=True):
#     # def _read_example(example):
#     #     desc = {
#     #         'doc_feature_list': tf.io.VarLenFeature(tf.int64),
#     #         'doc_label_list': tf.io.VarLenFeature(tf.int64)
#     #     }
#     #     _, seq_features = tf.io.parse_single_sequence_example(example, sequence_features=desc)
#     #     return tf.sparse.to_dense(seq_features['doc_feature_list']), \
#     #            tf.sparse.to_dense(seq_features['doc_label_list'])

#     def _read_example(example):
#         desc = {
#             'doc_feature_list': torch._utils._sparse.Int64Tensor,
#             'doc_label_list': torch._utils._sparse.Int64Tensor
#         }
#         sequence_features, _ = torch._utils._reconstruct_from_sparse(torch._utils._parse_sequence_example(example, sequence_features=desc))
#         return sequence_features['doc_feature_list'].to_dense(), sequence_features['doc_label_list'].to_dense()

#     # buffer_size = 10 * batch_size
#     # dataset = tf.data.TFRecordDataset([dataset_file]) \
#     #     .map(_read_example, num_parallel_calls=4) \
#     #     .prefetch(buffer_size) \
#     #     .padded_batch(
#     #         batch_size=batch_size,
#     #         padded_shapes=([None, None], [None, 1]),
#     #         padding_values=(tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64))) \
#     #     .shuffle(buffer_size=buffer_size)
#     # if repeat:
#     #     return dataset.repeat()
#     # return dataset

#     buffer_size = 10 * batch_size
#     dataset = torch.load(dataset_file)
#     print(dataset)
#     dataset = data_utils.TensorDataset([dataset_file])
#     dataset = dataset.map(_read_example, num_workers=4)
#     # 파일에서 데이터셋 로드
#     dataset = torch.load(dataset_file).map(_read_example, num_workers=4)

#     # 데이터 프리페치와 배치 패딩
#     dataset = dataset.prefetch(buffer_size)
#     dataset = data_utils.PaddingDataset(dataset, padding_values=(torch.tensor(0, dtype=torch.int64), torch.tensor(0, dtype=torch.int64)))
#     dataset = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     if repeat:
#         return data_utils.RepeatDataset(dataset)
#     return dataset


def get_class_weights(train_set_file):
    y_train = []
    for _, y in get_dataset(train_set_file, 1, False):
        y_train.extend(y.numpy().flatten())
    return class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DATA_DIR', help='Directory of files produced by the preprocessing script')
    ap.add_argument('-l', '--num_layers', type=int, default=2, help='The number of RNN layers')
    ap.add_argument('-u', '--hidden_units', type=int, default=768,
                    help='The number of hidden LSTM units')
    ap.add_argument('-d', '--dropout', type=float, default=0.5, help='The dropout percentage')
    ap.add_argument('-s', '--dense_size', type=int, default=256, help='Size of the dense layer')
    ap.add_argument('-e', '--epochs', type=int, default=20, help='The number of epochs')
    ap.add_argument('-b', '--batch_size', type=int, default=16, help='The batch size')
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
    print(train_dataset)
    
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

    class_weights = get_class_weights(train_set_file)
    print('using class weights {}'.format(class_weights))

    kwargs = {
        'input_size': info['num_words'] + info['num_tags'],
        'hidden_size': args.hidden_units,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'dense_size': args.dense_size}
    clf = LeafClassifier(**kwargs)

    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)

    params_file = os.path.join(args.working_dir, 'params.csv')
    print('writing {}...'.format(params_file))
    with open(params_file, 'w') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])

    # clf.train1(train_dataset, train_steps, args.epochs, log_file, ckpt_dir, class_weights,
    #           dev_dataset, info.get('num_dev_examples'),
    #           test_dataset, info.get('num_test_examples'),
    #           args.interval)
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
    clf.train1(train_loader=train_dataset, optimizer=optimizer, loss_fn=nn.BCELoss(), epochs=args.epochs, device='cpu')
    acc = clf.evaluate(test_dataset, 'cpu')
    print("accuracy 0 :", acc[0])
    print("accuracy 1 :", acc[1])

if __name__ == '__main__':
    main()
