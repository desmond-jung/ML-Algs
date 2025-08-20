import os
from pathlib import Path
import torch
import re
import random
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

# Maximum length of input sequence - 64 is common for NLP tasks
MAX_LEN = 64

# load data into memory
corpus_movie_conv = './datasets/movie_conversations.txt'
corpus_movie_lines = './datasets/movie_lines.txt'
with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()
with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()

# # movie convo - id of speaker, movie ID, list of line IDs or sequence of convo lines
# print("First line of movie_conversations.txt:")
# print(conv[0])

# # movie lines txt format - line ID, user ID or speaker, movie ID, speaker, dialogue (text we want)
# print("\nFirst line of movie_lines.txt:")
# print(lines[0])

# splitting text using the special line characters
lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]

# generate question answer pairs
pairs = []
for con in conv:
    # the list at the end 
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []

        if i == len(ids) - 1:
            break
            
        first = lines_dic[ids[i]].strip()
        second = lines_dic[ids[i+1]].strip()

        qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
        qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
        pairs.append(qa_pairs)

# Word Piece Tokenizer
os.mkdir('./data')
text_data = []
file_count = 0

for sample in tqdm.tqdm([x[0] for x in pairs]):
    text_data.append(sample)

    # save to file at the 10k mark
    if len(text_data) == 10000:
        with open(f'./data/text_file{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count +=1

paths = [str(x) for x in Path('./data').glob('**/*.txt')]

# training tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text = True,
    handle_chinese_chars = False, # includes spaces around chinese chars
    strip_accents = False, # removes accents
    lowercase=True # views lower and capital as equal
)

tokenizer.train(
    files=paths,
    vocab_size=30_000, # total number of tokens
    min_frequency=5, # min freq for a token to be merged
    limit_alphabet=1000, #max number of diff characters
    wordpieces_prefix='##', #predix added to pieces of words
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'] # pad to equal the length of sentences, classificatoin, separation (EOS), masked word, replacement token if not found in vocab
)

os.mkdir('./bert-it-1')
tokenizer.save_model('./bert-it-1','bert-it')
tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)



class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=64):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines
    
    
