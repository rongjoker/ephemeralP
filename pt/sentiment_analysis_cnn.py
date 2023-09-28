import torch
from torch import nn
import d2l_torch as d2l
import logging
import datetime

logger = logging.getLogger('sentiment-analysis-cnn')
logger.setLevel(logging.INFO)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        # todo 对比了最大汇聚和平均汇聚，最大汇聚效果更好写，7个epoch下test acc 0.871
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def cnn_train(lr=0.001, epochs=5):
    def init_weights(m):
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)

    print('loading dataset imdb start: ', datetime.datetime.now())
    batch_size = 64
    train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
    print('loading dataset imdb end:', datetime.datetime.now())
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    devices = d2l.try_all_gpus()
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
    net.apply(init_weights)
    # Loading Pretrained Word Vectors
    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    print('embeds.shape:', embeds.shape)

    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    # lr, num_epochs = 0.01, 5
    # lr, num_epochs = 0.001, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, epochs, devices)
    torch.save(net.state_dict(), 'model/sentiment_cnn.pth')


# @save
def cnn_infer(sequence):
    logger.info('loading dataset imdb start')
    print('loading dataset imdb start: ', datetime.datetime.now())
    batch_size = 128
    train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
    logger.info('loading dataset imdb end')
    print('loading dataset imdb end:', datetime.datetime.now())
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    model = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
    model.load_state_dict(torch.load('model/sentiment_cnn.pth'))
    model = model.to(d2l.try_gpu())
    model.eval()  # 设置模型为推理模式
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(model(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


# batch_size = 64 读取imdb耗时3分钟半
cnn_train(epochs=1)
print(cnn_infer("I may consider myself lucky to be alive to watch Christopher Nolan Works which get better by "
                "years.Oppenheimer is - with no doubt-going to be one of the best movies in the history. Amazing "
                "cinematography, Exceptional acting and terrifying Soundtracks.All the cast are great from cilian "
                "Murphy who is going for the oscar with this role to Rupert Downey jr and Emily blunt and finally "
                "rami malik who has small scenes but you will never forget them.I didn't watch it in Imax as i "
                "couldn't wait and ran to the nearest cinema but now i will sure book an imax ticket.Don't waste any "
                "time, book your ticket and Go watch it.. NOW."))
# print(lstm_infer('this movie is so bad'))
