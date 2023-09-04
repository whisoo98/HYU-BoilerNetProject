import os
import argparse
import pickle
import json
from collections import defaultdict

import nltk
import numpy as np
import torch
import torch.nn as nn
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

from misc import util
import time

counts = 0
    
def process(doc, tags, words):
    """
    Process "doc", updating the tag and word counts.
    Return the document representation, the HTML tags and the words.
    """
    html_list = [html.strip() for html in (str(doc.find_all('html')[0])).split('\n')]
    label_list = []
    for idx, html in enumerate(html_list):
        if('__boilernet_label="0"' in html):
            label_list.append(0)
            html_list[idx] = html.replace(' __boilernet_label="0"','')
        elif('__boilernet_label="1"' in html):
            label_list.append(1)
            html_list[idx] = html.replace(' __boilernet_label="1"','')
        else:
            label_list.append(0)
    
    return html_list, label_list


def parse(filenames):
    """
    Read and parse all HTML files.
    Return the parsed documents and a set of all words and HTML tags.
    """
    html_result = {}
    label_result = {}
    tags = defaultdict(int)
    words = defaultdict(int)
    global counts
        
    for f in tqdm(filenames):
        try:
            with open(f, 'rb') as hfile:
                doc = BeautifulSoup(hfile, features='html5lib')
            basename = os.path.basename(f)
            # Save bert input as same format of results
            html_result[basename], label_result[basename] = process(doc, tags, words)
        except Exception as e:
            tqdm.write('error processing {}'.format(f))
            break
            
    return html_result, label_result

def write_npy(filename, dataset):
    """Write the dataset to a .pth file."""
    # with tf.io.TFRecordWriter(filename) as writer:
    
    # dataset[0] : result[basename]
    # dataset[1] : bert_input[basename]
    
    data = [{'html_list' : dataset[0], 'label_list': dataset[1] }]
    
    data = np.array(data)
    np.save(file = filename, arr = data, allow_pickle=True)


def save(save_path, train_set, dev_set=None, test_set=None):
    """Save the data."""
    os.makedirs(save_path, exist_ok=True)
    
    info = {}
    
    train_file = os.path.join(save_path, 'train.npy')
    print('writing {}...'.format(train_file))
    write_npy(train_file, train_set)
    info['num_train_examples'] = len(train_set)

    if dev_set is not None:
        dev_file = os.path.join(save_path, 'dev.npy')
        print('writing {}...'.format(dev_file))
        write_npy(dev_file, dev_set)
        info['num_dev_examples'] = len(dev_set)

    if test_set is not None:
        test_file = os.path.join(save_path, 'test.npy')
        print('writing {}...'.format(test_file))
        write_npy(test_file, test_set)
        info['num_test_examples'] = len(test_set)


def read_file(f_name):
    with open(f_name, encoding='utf-8') as fp:
        for line in fp:
            yield line.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DIRS', nargs='+', help='A list of directories containing the HTML files')
    ap.add_argument('-s', '--split_dir', help='Directory that contains train-/dev-/testset split')
    ap.add_argument('-w', '--num_words', type=int, help='Only use the top-k words')
    ap.add_argument('-t', '--num_tags', type=int, help='Only use the top-l HTML tags')
    ap.add_argument('--save', default='result', help='Where to save the results')
    args = ap.parse_args()

    # files required for tokenization
    nltk.download('punkt')

    filenames = []
    for d in args.DIRS:
        filenames.extend(util.get_filenames(d))
    html_result, label_result = parse(filenames)
    
    if args.split_dir:
        train_set_file = os.path.join(args.split_dir, 'train_set.txt')
        dev_set_file = os.path.join(args.split_dir, 'dev_set.txt')
        test_set_file = os.path.join(args.split_dir, 'test_set.txt')
        print(train_set_file)
        print(dev_set_file)
        print(test_set_file)
        
        # save result & bert input in "~~"_set list
        # [0] : result
        # [1] : bert_input
        train_set = [[],[]]
        dev_set = [[],[]]
        test_set = [[],[]]
        
        train_set[0] = [html_result[basename] for basename in read_file(train_set_file)]
        dev_set[0] = [html_result[basename] for basename in read_file(dev_set_file)]
        test_set[0] = [html_result[basename] for basename in read_file(test_set_file)]
        
        train_set[1] = [label_result[basename] for basename in read_file(train_set_file)]
        dev_set[1] = [label_result[basename] for basename in read_file(dev_set_file)]
        test_set[1] = [label_result[basename] for basename in read_file(test_set_file)]
    else:
        train_set = list(html_result.values(), label_result.values() )
        dev_set, test_set = None, None

    save(args.save, train_set, dev_set, test_set)
    
    

if __name__ == '__main__':
    main()