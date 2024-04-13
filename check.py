# from utils.normalizer import normalize_string

# s = 'What are your interests'

# z = [normalize_string(t) for t in s.split()]

# # x =  normalize_string('interested')

# print(z)

import joblib
import pandas as pd
from utils.normalizer import normalize_string

#constants
n_embed = 6

vocabs = joblib.load('saved/vocabulary/vison_voc.joblib')

# print(vocabs['word_int']['favorite'])

# print(vocabs['no_unique_words'])


print(vocabs['int_word'])