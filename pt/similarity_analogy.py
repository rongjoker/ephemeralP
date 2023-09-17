import os
import torch
from torch import nn
from d2l import torch as d2l


# 14.7. 词的相似性和类比任务
# https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/similarity-analogy.html

# 以下列出维度为50、100和300的预训练GloVe嵌入，可从GloVe网站下载。预训练的fastText嵌入有多种语言。这里我们使用可以从fastText网站下载300维度的英文版本（“wiki.en”）。


# @save
class TokenEmbedding:
    """GloVe嵌入"""

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe网站：https://nlp.stanford.edu/projects/glove/
        # fastText网站：https://fasttext.cc/
        with open(file=os.path.join(data_dir, 'vec.txt'), mode='r', errors='ignore') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息，例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


glove_6b50d = TokenEmbedding('glove.6b.50d')
print(len(glove_6b50d))
print(glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])


# 为了根据词向量之间的余弦相似性为输入词查找语义相似的词，我们实现了以下knn（k近邻）函数
def knn(W, x, k):
    # 增加1e-9以获得数值稳定性
    cos = torch.mv(W, x.reshape(-1, )) / (
            torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
            torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]


def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 排除输入词
        print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')


# glove_6b50d中预训练词向量的词表包含400000个词和一个特殊的未知词元。排除输入词和未知词元后，我们在词表中找到与“chip”一词语义最相似的三个词。
get_similar_tokens('chip', 3, glove_6b50d)


# 除了找到相似的词，我们还可以将词向量应用到词类比任务中。 例如，“man” : “woman” :: “son” : “daughter”是一个词的类比。 “man”是对“woman”的类比，“son”是对“daughter”的类比
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # 删除未知词


print(get_analogy('man', 'woman', 'son', glove_6b50d))

print(get_analogy('beijing', 'china', 'tokyo', glove_6b50d))

print(get_analogy('bad', 'worst', 'big', glove_6b50d))

print(get_analogy('do', 'did', 'go', glove_6b50d))
