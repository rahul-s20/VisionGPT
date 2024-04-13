import os
import pandas as pd
from utils.normalizer import normalize_string
from functools import reduce
from operator import add
import joblib
from tqdm import tqdm



class Tokenizer:
    def __init__(self, dir: str) ->None :
        self.vocabulary = {}
        self.full_list = list()
        files = os.listdir('data')
        self.text_files = [i for i in files if i.endswith('.txt')]
        self.list_datasets = [self.produce_df(dir, x) for x in self.text_files]
        if len(self.list_datasets) > 0:
            self.df = pd.concat(self.list_datasets, ignore_index=True)
        else:
            raise ValueError('Dataframe criteria does not match')

    def produce_df(self, dir:str, inp_file: str):
        return pd.read_csv(dir + '/' + inp_file, sep='\t', names=['inputs', 'targets'])
    
    def split_sentense(self, sen: str):
        final_list = []
        split_sen = [normalize_string(_i) for _i in sen.split()]
        # for _x in split_sen:
        #     if ' ' in _x:
        #         final_list.extend(_x.split())
        #     else:
        #         final_list.extend(_x)    
        return split_sen
    
    def string_to_int(self, lis_words: list) -> dict:
        word_dict = {}
        for idx, _i in enumerate(lis_words):
            word_dict[_i] = idx
        return word_dict    
    
    def int_to_string(self, words_dict: dict) -> dict:
        return {v: k for k, v in words_dict.items()}

    
    def __call__(self):
        all_words = []
        inputs = self.df['inputs'].tolist()
        targets = self.df['targets'].tolist()
        self.full_list.extend(inputs)
        self.full_list.extend(targets)

        normalized_list = list(map(self.split_sentense, tqdm(self.full_list)))
        for _x in tqdm(normalized_list):
            all_words.append(' ')
            all_words.extend(_x)
        unique_words = list(dict.fromkeys(all_words))
        word_int = self.string_to_int(unique_words)
        int_word = self.int_to_string(word_int)
        self.vocabulary['word_int'] = word_int
        self.vocabulary['int_word'] = int_word
        self.vocabulary['no_unique_words'] = len(unique_words)
        joblib.dump(self.vocabulary, 'saved/vocabulary/vison_voc.joblib')
        return True



a = Tokenizer('data')
a()
print('DOne......')