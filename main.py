import joblib
import pandas as pd
from utils.normalizer import normalize_string
from test_encoder_kagg import Embedding, PositionalEmbedding
from torch import nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
#constants
n_embed = 5

vocabs = joblib.load('saved/vocabulary/vison_voc.joblib')

df = pd.read_csv('data/chatbot_dataset.txt', sep='\t', names=['inputs', 'targets'])

data = df[:10]

_vals = data.inputs.tolist()
_vals.extend(data.targets.tolist())
# values = data.values.tolist()
def normalize_sentense(sen: str):
    sen = sen.split()
    return ' '. join([normalize_string(i) for i in sen])
nv = list(map(normalize_sentense, _vals))


input_ = list(map(normalize_sentense, data.inputs.tolist()))
output_ = list(map(normalize_sentense, data.targets.tolist()))

def tokenization(sen: str):
    sen = sen.split()
    return [vocabs['word_int'][i] for i in sen]

tokenized_input = list(map(tokenization, input_))
tokenization_op = list(map(tokenization, output_))

input_max_len = max([len(_i) for _i in tokenized_input]) # vocab_size 5
output_max_len = max([len(_j) for _j in tokenization_op]) # vocab_size 14

def padding(data_lst, context_len):
    return [i+ [0] * (context_len - len(i)) for i in data_lst]

tokenized_input = padding(tokenized_input, input_max_len)
tokenization_op = padding(tokenization_op, output_max_len)


x_ = torch.tensor(tokenized_input, dtype=torch.long).to(device)
y_ = torch.tensor(tokenization_op, dtype=torch.long).to(device)

# enc = Encoder(vocab_size=input_max_len, n_embed=n_embed)
vocab_size = vocabs['no_unique_words']
embed_dim = 512 #default
n_heads = 8 #default

ffn_hidden = 500   #2048
num_layers = 6
drop_prob = 0.1
num_heads = input_max_len

embd = Embedding(vocab_size=vocab_size, embed_dim=embed_dim)
embd = embd.to(device)

positional_enc = PositionalEmbedding(max_seq_len=input_max_len, embed_model_dim=embed_dim)
positional_enc = positional_enc.to(device)

inp_emb = embd(x_)
pos_enc = positional_enc(inp_emb)

print(pos_enc.size())


# input_embedding_table = nn.Embedding(vocabs['no_unique_words'], input_max_len)
# # output_embedding_table = nn.Embedding(vocabs['no_unique_words'], output_max_len)
# # x = input_embedding_table(x_)
# # y = output_embedding_table(y_)
# print(device)
# encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, input_embedding_table)
# print('111111111')
# encoder = encoder.to(device)
# print('2222222222222222222222')

# # print(x_)
# inp = encoder(x_)
# print('33333333333')
# print(inp) 10, 5, 512

"""
step 1 : 10 x 5 x 512 --> 10 x 5 x 8 x 64
"""