# Create a vocabulary wrapper
import nltk
import pickle
from collections import Counter
import json
import argparse
import os
from tqdm import tqdm

annotations = {
    'didemo': ['../../../playground/vsepp-develop-develop/data/data/train_data.json', '../../../playground/vsepp-develop-develop/data/data/val_data.json', '../../../playground/vsepp-develop-develop/data/data/test_data.json']
}


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            if word in glove:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['UNK']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)



def build_vocab(data_path, data_name, jsons, threshold):
    global glove
    """Build a simple vocabulary wrapper."""
    counter = Counter()

    f_glove = open('/data1/bwzhang/glove.840B.300d.txt','r')
    glove = {}
    for line in tqdm(f_glove):
        line = line[0:-1]
        line_split = line.split(' ')
        word = line_split[0]
        glove[word] = 1


    for path in jsons[data_name]:
        full_path = os.path.join(path)
        print full_path
        dic = json.load(open(full_path,'r'))
        for i in range(len(dic)):
            sen = dic[i]['description']
            tokens = nltk.tokenize.word_tokenize(sen.lower())
            counter.update(tokens)


    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
#    vocab = pickle.load(open('vocab/coco_vocab.pkl','r'))
    vocab = Vocabulary()
    vocab.add_word('PAD')
    vocab.add_word('BOS')
    vocab.add_word('EOS')
    vocab.add_word('UNK')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    print (len(vocab))
    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, data_name, jsons=annotations, threshold=1)
    with open('./vocab/%s_precomp_vocab_total.pkl' % data_name, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", './vocab/%s_precomp_vocab_total.pkl' % data_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/')
    parser.add_argument('--data_name', default='didemo',
                        help='coco|anet|didemo')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)
