import pickle as pl
from anet_vocab import Vocabulary
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description="using vocab to get glove")
parser.add_argument('vocab_path',  help='path for vocab')
parser.add_argument('glove_path',  help='path for glove')
parser.add_argument('output_path', help='path for output')
args = parser.parse_args()
print (args)


def main():
    vocab = pl.load(open(args.vocab_path,'r'))
    f_glove = open(args.glove_path,'r')
    glove = {}
    output_glove = []

    for line in tqdm(f_glove):
        line = line[0:-1]
        line_split = line.split(' ')
        word = line_split[0]
        vector = line_split[1:]
        glove[word] = vector

    for ind in tqdm(range(len(vocab))):
        cur_word = vocab.idx2word[ind]
        if cur_word in glove:
            output_glove.append(glove[cur_word])
    np.savez(args.output_path, output_glove)



if __name__ == "__main__":
    main()
