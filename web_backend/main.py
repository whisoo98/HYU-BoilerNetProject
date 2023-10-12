from flask import Flask, render_template, url_for, request
#import tensorflow
import os
import argparse
import pickle
import json
from collections import defaultdict
import json

from transformers import BertTokenizer
from leaf_classifier import LeafClassifier
from train import get_dataset, get_class_weights
from misc import util
from preprocess import get_leaves, parse
import tensorflow as tf
import csv
from tensorflow.keras.models import load_model
import nltk
import numpy as np
import tensorflow as tf
import math
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

from misc import util
import requests
import threading
import time
sem = threading.Semaphore()
#=========preprocess line============

def get_leaves(node, tag_list=[], label=0):
    """Return all leaves (NavigableStrings) in a BS4 tree."""
    tag_list_new = tag_list + [node.name]
    if node.has_attr('__boilernet_label'):
        label = int(node['__boilernet_label'])

    result = []
    for c in node.children:
        if isinstance(c, NavigableString):
            # might be just whitespace
            if c.string is not None and c.string.strip():
                result.append((c, tag_list_new, label))
        elif c.name not in util.TAGS_TO_IGNORE:
            result.extend(get_leaves(c, tag_list_new, label))
    return result


def get_leaf_representation(node, tag_list, label, tokenize=None):
    """Return dicts of words and HTML tags that representat a leaf."""
    tags_dict = defaultdict(int)
    for tag in tag_list:
        tags_dict[tag] += 1
    words_dict = defaultdict(int)
    if tokenize is None:
        tokenize = nltk.word_tokenize
        words = tokenize(node.string)
    else:
        words = tokenize.convert_ids_to_tokens(tokenize.encode(node.string)[1:-1])
    for word in words:  # nltk.word_tokenize(node.string):
        words_dict[word.lower()] += 1
    return dict(words_dict), dict(tags_dict), label


def process(doc, tags, words, tokenize=None):
    """
    Process "doc", updating the tag and word counts.
    Return the document representation, the HTML tags and the words.
    """
    result = []
    for leaf, tag_list, is_content in get_leaves(doc.find_all('html')[0]):
        leaf_representation = get_leaf_representation(leaf, tag_list, is_content, tokenize=tokenize)
        result.append(leaf_representation)
        words_dict, tags_dict, _ = leaf_representation
        for word, count in words_dict.items():
            words[word] += count
        for tag, count in tags_dict.items():
            tags[tag] += count
    return result
def process_doc(doc, tags, words):
    """
    Process "doc", updating the tag and word counts.
    Return the document representation, the HTML tags and the words.
    """
    result = []
    for leaf, tag_list, is_content in get_leaves(doc.find_all('html')[0]):
        result.append(leaf)
    return result

def parse(filenames, language=None):
    """
    Read and parse all HTML files.
    Return the parsed documents and a set of all words and HTML tags.
    """
    result = {}
    tags = defaultdict(int)
    words = defaultdict(int)
    tokenize = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # if language != "English":
    #    tokenize = AutoTokenizer.from_pretrained('bert-base-uncased')#KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    for f in tqdm(filenames):
        try:
            with open(f, 'rb') as hfile:
                doc = BeautifulSoup(hfile, features='html5lib')
            basename = os.path.basename(f)
            result[basename] = process(doc, tags, words, tokenize=tokenize)

        except:
            tqdm.write('error processing {}'.format(f))

    return result, tags, words
def parse_doc(filenames):
    """
    Read and parse all HTML files.
    Return the parsed documents and a set of all words and HTML tags.
    """
    result = {}
    tags = defaultdict(int)
    words = defaultdict(int)

    for f in tqdm(filenames):
        try:
            with open(f, 'rb') as hfile:
                doc = BeautifulSoup(hfile, features='html5lib')
            basename = os.path.basename(f)
            result[basename] = process_doc(doc, tags, words)
        except:
            tqdm.write('error processing {}'.format(f))
    return result, tags, words


def get_feature_vector(words_dict, tags_dict, word_map, tag_map):
    """Return a feature vector for an item to be classified."""
    vocab_vec = np.zeros(len(word_map), dtype='int32')
    for word, num in words_dict.items():
        # if the item is not in the map, use 0 (OOV word)
        vocab_vec[word_map.get(word, 0)] = num

    tags_vec = np.zeros(len(tag_map), dtype='int32')
    for tag, num in tags_dict.items():
        # if the tag is not in the map, use 0 (OOV tag)
        tags_vec[tag_map.get(tag, 0)] = num

    return np.concatenate([vocab_vec, tags_vec])


def get_vocabulary(d, num=None):
    """Return an integer map of the top-k vocabulary items and add <UNK>."""
    l = sorted(d.keys(), key=d.get, reverse=True)
    if num is not None:
        l = l[:num]
    int_map = util.get_int_map(l, offset=1)
    int_map['<UNK>'] = 0
    #추가한 부분
    k = 1
    while len(int_map.keys()) <= num:
        int_map['<UNK_'+str(k)+'>'] = 0
        k += 1
    #print(len(int_map.keys()))
    return int_map


def get_doc_inputs(doc, word_map, tag_map):
    """Transform "docs" into the input format accepted by the classifier."""

    def _int64_feature(l):
        """Return an int64_list."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=l))

    doc_features = []
    doc_labels = []
    for words_dict, tags_dict, label in doc:
        feature_vector = get_feature_vector(words_dict, tags_dict, word_map, tag_map)
        doc_features.append(_int64_feature(feature_vector))
        doc_labels.append(_int64_feature([label]))
    doc_feature_list = tf.train.FeatureList(feature=doc_features)
    doc_label_list = tf.train.FeatureList(feature=doc_labels)
    yield doc_feature_list, doc_label_list


def write_tfrecords(filename, dataset, word_map, tag_map):
    """Write the dataset to a .tfrecords file."""
    with tf.io.TFRecordWriter(filename) as writer:
        for doc_feature_list, doc_label_list in get_doc_inputs(dataset, word_map, tag_map):
            f = {'doc_feature_list': doc_feature_list, 'doc_label_list': doc_label_list}
            feature_lists = tf.train.FeatureLists(feature_list=f)
            example = tf.train.SequenceExample(feature_lists=feature_lists)
            writer.write(example.SerializeToString())


def save(save_path, words, tags, train_set):
    info = {}
    tokenize = BertTokenizer.from_pretrained('bert-base-multilingual-cased') #지워도 될 듯?
    word_map = {word: i for i, word in enumerate(words)}
    tag_map = {tag: i for i, tag in enumerate(tags)}

    info['num_words'] = len(word_map)
    info['num_tags'] = len(tag_map)
    print(info)

    predict_file = os.path.join(save_path, 'predict.tfrecords')
    print('writing {}...'.format(predict_file))
    write_tfrecords(predict_file, train_set, word_map, tag_map)
    info['num_train_examples'] = len(train_set)

    info_file = os.path.join(save_path, 'info.pkl')
    with open(info_file, 'wb') as fp:
        pickle.dump(info, fp)

def preprocess(docs):
    try:
        nltk.download('punkt')
        print('nltk downloaded')
        data, tags, words = parse([docs])
        print('parse complete')
        tags = get_vocabulary(tags, 50)
        words = get_vocabulary(words, 1000)
        print('get_vocab complete')
        save('./process1', words, tags, data['temp.html'])
        print('save complete')
    except Exception as e:
        print("preprocess error: ", e)

def predict():
    params_path = os.path.join('./process', 'params.csv')
    params = {}
    with open(params_path, 'r') as f:
        for i in csv.reader(f):
            params[i[0]] = i[1]
    DATA_DIR = './process1'
    # 文章をノードごとに分割
    file_path = './process1'
    doc_data, _, _ = parse_doc(util.get_filenames(file_path))

    info_file = os.path.join(DATA_DIR, 'info.pkl')
    with open(info_file, 'rb') as fp:
        info = pickle.load(fp)
        info['num_train_examples'] = len(doc_data)
        train_steps = math.ceil(info['num_train_examples'] / int(params["batch_size"]))

    predict_set_file = os.path.join('./process1', 'predict.tfrecords')
    predict_dataset = get_dataset(predict_set_file, 1, repeat=False)

    kwargs = {'input_size': info['num_words'] + info['num_tags'],
              'hidden_size': int(params['hidden_units']),
              'num_layers': int(params['num_layers']),
              'dropout': float(params['dropout']),
              'dense_size': int(params['dense_size'])}
    print(kwargs)
    clf = LeafClassifier(**kwargs)

    checkpoint_path = "model.{:03d}.h5".format(49)
    checkpoint_path = os.path.join("./process", checkpoint_path)
    clf.model = load_model(checkpoint_path)

    _, y_pred = clf.eval(predict_dataset, train_steps, desc="")


    filenames = util.get_filenames(file_path)
    raw_filenames = [i.split("/")[-1] for i in filenames]

    # 予測結果を保存
    pred_index = [bool(i) for i in y_pred]
    delete_index = [not bool(i) for i in y_pred]
    counter = 0
    for i in range(len(raw_filenames)):
        numpy_data = np.array(doc_data[raw_filenames[i]])
        contents = numpy_data[pred_index[counter:counter + len(numpy_data)]]
        delete_contents = numpy_data[delete_index[counter:counter + len(numpy_data)]]
        counter += len(numpy_data)
        content_text = ""
        delete_text = ""
        for content in contents:
            content_text += str(content) + "\n"
        for delete_content in delete_contents:
            delete_text += str(delete_content) + "\n"
        dir_path = './process1'
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, raw_filenames[i].replace("/", "_").replace(".", "_") + ".txt"), 'w') as file:
            file.write(content_text)
        with open(os.path.join(dir_path, raw_filenames[i].replace("/", "_").replace(".", "_") + "delete.txt"),
                  'w') as file:
            file.write(delete_text)
        #print(content_text)
        #print(delete_text)
        return content_text

#==========flask line==========
app =  Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/1')
def requestURL(url="https://n.news.naver.com/article/366/0000926317?cds=news_media_pc"):
    temp = requests.get(url)
    #print(temp)
    #print(temp.text)
    textLine = temp.text
    return temp.text

@app.route('/2', methods=['GET'])
def sampleHtml():
    #return "Welcome to Flask"
    return render_template("index.html")


@app.route('/받는url', methods=['POST'])
def getHtml():
    url = request.get_data().decode('utf-8')
    tempHTML = requests.get(url)
    print(tempHTML)
    dir_path = './process1'
    os.makedirs(dir_path, exist_ok=True)  # 디렉토리가 없으면 생성
    path = os.path.join(dir_path, 'temp.html')
    sem.acquire()
    open(path, 'wb').write(tempHTML.content)
    try:
        #print(tempHTML.content)
        preprocess(path)
        res =  predict()
        sem.release()
        return res
    except Exception as e:
        print("eval error:", e)
        sem.release()
        return "ERROR!"

if __name__ == '__main__':
    sem = threading.Semaphore()
    app.run(host='0.0.0.0', port=9999)