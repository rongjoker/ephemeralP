# encoding=utf8
import re
import random
import tarfile
import requests
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F
from paddle.nn import LSTM, Embedding, Dropout, Linear
import net_base as nb

import logging

logger = logging.getLogger('lstm-infer')
logger.setLevel(logging.DEBUG)


def load_imdb(is_training):
    data_set = []

    # aclImdb_v1.tar.gz解压后是一个目录
    # 我们可以使用python的rarfile库进行解压
    # 训练数据和测试数据已经经过切分，其中训练数据的地址为：
    # ./aclImdb/train/pos/ 和 ./aclImdb/train/neg/，分别存储着正向情感的数据和负向情感的数据
    # 我们把数据依次读取出来，并放到data_set里
    # data_set中每个元素都是一个二元组，（句子，label），其中label=0表示负向情感，label=1表示正向情感
    logger.info('start reading imdb')
    for label in ["pos", "neg"]:
        with tarfile.open("work/aclImdb_v1.tar.gz") as tarf:
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
                else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)
            tf = tarf.next()
            while tf != None:
                if bool(path_pattern.match(tf.name)):
                    sentence = tarf.extractfile(tf).read().decode()
                    sentence_label = 0 if label == 'neg' else 1
                    data_set.append((sentence, sentence_label))
                tf = tarf.next()
    logger.info('read imdb finished')
    return data_set


train_corpus = load_imdb(True)
test_corpus = load_imdb(False)

for i in range(5):
    print("sentence %d, %s" % (i, train_corpus[i][0]))
    print("sentence %d, label %d" % (i, train_corpus[i][1]))


def data_preprocess(corpus):
    data_set = []
    for sentence, sentence_label in corpus:
        # 这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        # 一般来说这样的做法有助于效果提升
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")

        data_set.append((sentence, sentence_label))

    return data_set


train_corpus = data_preprocess(train_corpus)
test_corpus = data_preprocess(test_corpus)
print(train_corpus[:5])
print(test_corpus[:5])


# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    word2id_dict = dict()
    word2id_freq = dict()

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict


word2id_freq, word2id_dict = build_dict(train_corpus)
vocab_size = len(word2id_freq)
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(10), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))


# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        # 这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        # 如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict \
                        else word2id_dict['[oov]'] for word in sentence]
        data_set.append((sentence, sentence_label))
    return data_set


train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
print("%d tokens in the corpus" % len(train_corpus))
print(train_corpus[:5])
print(test_corpus[:5])


# 编写一个迭代器，每次调用这个迭代器都会返回一个新的batch，用于训练或者预测
def build_batch(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle=True, drop_last=True):
    # 模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num):

        # 每个epoch前都shuffle一下数据，有助于提高模型训练的效果
        # 但是对于预测任务，不要做数据shuffle
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []
    if not drop_last and len(sentence_batch) > 0:
        yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")


# for batch_id, batch in enumerate(build_batch(word2id_dict, train_corpus, batch_size=3, epoch_num=3, max_seq_len=30)):
#     print(batch)
#     break


paddle.seed(0)
random.seed(0)
np.random.seed(0)

# 定义训练参数
epoch_num = 5
batch_size = 128

learning_rate = 0.0001
dropout_rate = 0.2
num_layers = 1
hidden_size = 256
embedding_size = 256
max_seq_len = 128
vocab_size = len(word2id_freq)


def evaluate(model):
    # 开启模型测试模式，在该模式下，网络不会进行梯度更新
    model.eval()

    # 定义以上几个统计指标
    tp, tn, fp, fn = 0, 0, 0, 0

    # 构造测试数据生成器
    test_loader = build_batch(word2id_dict, test_corpus, batch_size, 1, max_seq_len)

    for sentences, labels in test_loader:
        # 将张量转换为Tensor类型
        sentences = paddle.to_tensor(sentences)
        labels = paddle.to_tensor(labels)

        # 获取模型对当前batch的输出结果
        logits = model(sentences)

        # 使用softmax进行归一化
        probs = F.softmax(logits)

        # 把输出结果转换为numpy array数组，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        probs = probs.numpy()
        for i in range(len(probs)):
            # 当样本是的真实标签是正例
            if labels[i][0] == 1:
                # 模型预测是正例
                if probs[i][1] > probs[i][0]:
                    tp += 1
                # 模型预测是负例
                else:
                    fn += 1
            # 当样本的真实标签是负例
            else:
                # 模型预测是正例
                if probs[i][1] > probs[i][0]:
                    fp += 1
                # 模型预测是负例
                else:
                    tn += 1

    # 整体准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 输出最终评估的模型效果
    print("TP: {}\nFP: {}\nTN: {}\nFN: {}\n".format(tp, fp, tn, fn))
    print("Accuracy: %.4f" % accuracy)


# 加载训练好的模型进行预测，重新实例化一个模型，然后将训练好的模型参数加载到新模型里面
saved_state = paddle.load("lab/sentiment_classifier.pdparams")
sentiment_classifier = nb.SentimentClassifier(hidden_size, vocab_size, embedding_size, num_steps=max_seq_len,
                                           num_layers=num_layers, dropout_rate=dropout_rate)
sentiment_classifier.load_dict(saved_state)

# 评估模型
# evaluate(sentiment_classifier)

ss = [12, 14, 4, 3, 11114, 84, 1364, 2, 6990, 9993, 3, 11114, 320, 1364, 2773, 5335, 9993, 106, 3, 160, 90, 91, 3, 693, 20, 43, 3, 1475, 6354, 4270, 184475, 5, 4999, 15, 3, 282, 19, 6907, 17, 102, 336, 1179, 1019, 4, 36136, 679, 2922, 133, 240, 3, 92628, 36136, 7017, 39921, 5027, 5, 4209, 22573, 40, 1527, 184476, 13, 255, 155, 10032, 2864, 1608, 7, 2, 238, 687, 5, 2, 232, 5, 2359, 1445, 4790, 638, 466, 9000, 11, 2258, 15, 638, 8, 2, 8171, 5, 2, 10855, 4, 9, 120, 12, 661, 295, 10, 1621, 5637, 5, 2263, 713, 1629, 12175, 18916, 4, 1445, 184477, 10805, 11364, 6907, 7306, 42232, 458, 25, 178, 180, 111, 666, 137, 5, 12, 313, 2, 895, 4, 2256, 692, 83, 232, 5, 3, 2226, 187, 2516, 4999, 47455, 675, 2, 6935, 5, 2, 18558, 4, 3, 61, 2117, 6907, 1356, 17, 111, 1018, 19, 3, 1522, 8368, 83, 6539, 3, 1621, 15306, 302, 9349, 4, 81, 3, 88, 23292, 5, 2, 238, 1364, 40, 116, 2634, 824, 5, 12444, 6990, 184478, 24, 8926, 13, 93, 5472, 13, 6188, 39, 337, 25, 2, 131, 77, 25, 74, 914, 28, 436, 6, 337, 5546, 2022, 610, 6, 2, 1750, 2226, 2528, 2922, 6, 750, 11, 186, 40, 186, 23, 26, 698, 6, 121, 11, 186, 40, 186, 23, 28, 3512, 4, 2, 468, 4671, 20468, 5689, 67265, 8129, 23371, 43, 8, 7376, 1019, 5, 269, 638, 893, 23273, 37, 1202, 20541, 13, 184479, 7, 3, 61, 780, 53650, 24, 5372, 36, 10832, 29, 89, 37558, 4, 7, 3483, 31, 3, 636, 1336, 24, 1187, 36, 22, 206, 6765, 6907, 4, 240, 48, 268, 37, 61, 97, 89247, 781, 24952, 41, 32, 55434, 263, 15, 48, 175, 37, 2, 92629, 2, 24, 1187, 175, 6, 779, 101, 16, 45, 307, 4, 1139, 47, 72, 12212, 3143, 382, 6, 10381, 473, 92622, 2, 8428, 122, 7, 3, 92630, 4953, 36, 10172, 184480, 72, 1060, 4, 24008, 2, 420, 2837, 45644, 1148, 222, 67266, 7, 162, 302, 9441, 31, 2, 704, 452, 34, 21, 82760, 7678, 40, 2, 1171, 16624, 5, 21, 24318, 35, 5, 424, 495, 73372, 5, 184481, 52, 22, 45, 80, 121, 11, 5419, 134, 970, 3, 20558, 39065, 222, 3982, 1148, 36, 81, 7, 35, 1113, 16, 12536, 9, 5750, 27, 41, 3, 28807, 9, 367, 61, 125, 848, 47, 48, 27, 62, 6, 85, 17, 2, 75, 557, 303, 62, 6, 779, 222, 3982, 149, 96, 4, 25, 367, 91, 3, 20, 191, 457, 52, 7, 81, 3, 8667, 2792, 2, 8667, 290, 4194, 137, 8, 2, 172, 1796, 493, 83, 891, 2546]
# 将张量转换为Tensor类型
# sentences = paddle.to_tensor(ss)
ss = ss[:min(max_seq_len, len(ss))]
sss = [[word_id] for word_id in ss]
sss = [sss for i in range(0, 128)]
sentences = paddle.to_tensor(sss)
# 获取模型对当前batch的输出结果
logits = sentiment_classifier(sentences)

# 使用softmax进行归一化
probs = F.softmax(logits)

# 把输出结果转换为numpy array数组，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
probs = probs.numpy()
for prob in probs:
    logger.info("%s-----%s",round(prob[0], 3),round(prob[1], 3))
    # for p in prob:
    #     print(type(p))
    #     print(p)
    #     p1 = round(p, 3)
    #     print(p1)
    #     logger.info(p1)
    #     print(type(p1))
    # print(round(p, 3) for p in prob)


