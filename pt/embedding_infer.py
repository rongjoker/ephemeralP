import torch
from torch import nn
from d2l import torch as d2l
import rnn_base

# torch.multiprocessing.set_start_method('spawn')
# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
# 虽然独热向量很容易构建，但它们通常不是一个好的选择。一个主要原因是独热向量不能准确表达不同词之间的相似度
# 由于任意两个不同词的独热向量之间的余弦相似度为0，所以独热向量不能编码词之间的相似性
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)

embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')


# 在前向传播中，跳元语法模型的输入包括形状为（批量大小，1）的中心词索引center和形状为（批量大小，max_len）的上下文与噪声词索引contexts_and_negatives
# 输出中的每个元素是中心词向量和上下文或噪声词向量的点积。
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


loss = SigmoidBCELoss()

embed_size = 100


# net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
#                                  embedding_dim=embed_size),
#                     nn.Embedding(num_embeddings=len(vocab),
#                                  embedding_dim=embed_size))


def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                 / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
    torch.save(net, 'model/embedding.pth')


# lr, num_epochs = 0.002, 5
# train(net, data_iter, lr, num_epochs)


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
    print(query_token)
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')
    print('--------')


def infer_similar(query_token: 'joker'):
    net = torch.load('model/embedding.pth')
    i = 0
    net = net.to(torch.device(f'cuda:{i}'))
    get_similar_tokens(query_token, 3, net[0])


def train_embedding():
    lr, num_epochs = 0.002, 20

    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size))
    train(net, data_iter, lr, num_epochs)


if __name__ == '__main__':
    # train_embedding()
    words = ['joker', 'chip', 'lincoln', 'world war', 'rifle', 'china', 'usa']
    for word in words:
        infer_similar(word)
