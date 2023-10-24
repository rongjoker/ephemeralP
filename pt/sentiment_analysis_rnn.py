import torch
from torch import nn
import d2l_torch as d2l
import logging
import datetime
import rnn_base

logger = logging.getLogger('sentiment-analysis-rnn')
logger.setLevel(logging.INFO)

logger.info('loading dataset imdb start')
print('loading dataset imdb start: ', datetime.datetime.now())
batch_size = 128
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
logger.info('loading dataset imdb end')
print('loading dataset imdb end:', datetime.datetime.now())


def lstm_train(lr=0.01, epochs=5):
    def init_weights(module):
        if type(module) == nn.Linear:
            nn.init.xavier_uniform_(module.weight)
        if type(module) == nn.LSTM:
            for param in module._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(module._parameters[param])

    embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
    net = rnn_base.BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

    net.apply(init_weights)
    # Loading Pretrained Word Vectors
    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    print('embeds.shape:', embeds.shape)

    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    # lr, num_epochs = 0.01, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, epochs, devices)
    torch.save(net.state_dict(), 'model/sentiment_lstm_local.pth')


# @save
def lstm_infer(sequence):
    embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
    model = rnn_base.BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    model.load_state_dict(torch.load('model/sentiment_lstm_local.pth'))
    model = model.to(d2l.try_gpu())
    model.eval()  # 设置模型为推理模式
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(model(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


# batch_size = 64 读取imdb耗时3分钟半
lstm_train(epochs=5)
print(lstm_infer("I may consider myself lucky to be alive to watch Christopher Nolan Works which get better by "
                 "years.Oppenheimer is - with no doubt-going to be one of the best movies in the history. Amazing "
                 "cinematography, Exceptional acting and terrifying Soundtracks.All the cast are great from cilian "
                 "Murphy who is going for the oscar with this role to Rupert Downey jr and Emily blunt and finally "
                 "rami malik who has small scenes but you will never forget them.I didn't watch it in Imax as i "
                 "couldn't wait and ran to the nearest cinema but now i will sure book an imax ticket.Don't waste any "
                 "time, book your ticket and Go watch it.. NOW."))
# print(lstm_infer('this movie is so bad'))
