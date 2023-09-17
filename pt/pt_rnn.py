import torch
from torch import nn
from d2l import torch as d2l
import rnn_base

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1

num_inputs = vocab_size


# num_layers 通过num_layers的值来设定隐藏层数，可以创建深度循环神经网络
# perplexity 1.3, 288416.5 tokens/sec on cuda:0
def get_rnn(num_layers: 0):
    rnn_layer = nn.RNN(len(vocab), num_hiddens, num_layers)
    model = rnn_base.RNNModel(rnn_layer, len(vocab))
    model = model.to(device)
    return model


# perplexity 1.0, 342542.0 tokens/sec on cuda:0
def get_gru(num_layers: 0):
    gru_layer = nn.GRU(num_inputs, num_hiddens, num_layers)
    model = rnn_base.RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    return model


# lstm
# num_layers = 1 perplexity 1.0, 364835.4 tokens/sec on cuda:0
# num_layers = 2 perplexity 1.0, 229511.9 tokens/sec on cuda:0
def get_lstm(num_layers: 0):
    gru_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = rnn_base.RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    return model


cur_model = get_lstm(4)
d2l.train_ch8(cur_model, train_iter, vocab, lr, num_epochs, device)
# todo 分离梯度
# todo embedding
# todo 写诗
# 双向循环神经网络  不适合预测，做翻译、语音、文本分类可以
# 情感分析 https://zh-v2.d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html#sec-sentiment-rnn
