# # 绘制Epoch级图像函数
def report_plot_epoch(num_epochs, report_epoch):
    # 更新训练损失、训练准确度、测试损失、测试准确度图像
    # 转置一下report列表
    report_epoch_t = list(map(list, zip(*report_epoch)))

    plt.close('all')

    # 绘制第一张子图
    loss_plot = plt.subplot(2, 1, 1)
    loss_plot.plot(report_epoch_t[0], report_epoch_t[1],
                   color='tab:blue', label='train_loss')
    loss_plot.plot(report_epoch_t[0], report_epoch_t[3],
                   color='tab:orange', label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(1, num_epochs)
    # plt.ylim(0, 1)
    plt.xticks(range(1, num_epochs + 1, math.ceil(num_epochs / 10)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()

    # 绘制第二张子图
    acc_plot = plt.subplot(2, 1, 2)
    acc_plot.plot(report_epoch_t[0], report_epoch_t[2],
                  color='tab:blue', label='train_acc')
    acc_plot.plot(report_epoch_t[0], report_epoch_t[4],
                  color='tab:orange', label='test_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim(1, num_epochs)
    plt.ylim(0, 1)
    plt.xticks(range(1, num_epochs + 1, math.ceil(num_epochs / 10)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.pause(0.1)  # 图片停留0.1s


# # 绘制Batch级图像函数
def report_plot_batch(num_epochs, report_batch, report_epoch):
    # 更新训练损失、训练准确度、测试损失、测试准确度图像
    # 转置一下report列表
    report_batch_t = list(map(list, zip(*report_batch)))
    report_epoch_t = list(map(list, zip(*report_epoch)))

    plt.close('all')

    # 绘制第一张子图
    loss_plot = plt.subplot(2, 1, 1)
    loss_plot.plot(report_batch_t[0], report_batch_t[1],
                   color='tab:blue', label='train_loss')
    if not report_epoch_t == []:
        loss_plot.plot(report_epoch_t[0], report_epoch_t[3],
                       color='tab:orange', label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0, num_epochs)
    # plt.ylim(0, 1)
    plt.xticks(range(0, num_epochs + 1, math.ceil(num_epochs / 10)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()

    # 绘制第二张子图
    acc_plot = plt.subplot(2, 1, 2)
    acc_plot.plot(report_batch_t[0], report_batch_t[2],
                  color='tab:blue', label='train_acc')
    if not report_epoch_t == []:
        acc_plot.plot(report_epoch_t[0], report_epoch_t[4],
                      color='tab:orange', label='test_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim(0, num_epochs)
    plt.ylim(0, 1)
    plt.xticks(range(0, num_epochs + 1, math.ceil(num_epochs / 10)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.pause(0.1)  # 图片停留0.1s





# # 定义累加器的类，用于累加每个batch的运行状态数据（损失和准确度）
class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):  # 初始化，n为累加器的列数
        self.data = [0.0] * n  # list * int 意思是将数组重复 int 次并依次连接形成一个新数组

    def add(self, *args):  # data和args对应列累加
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):  # 重置
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):  # 索引
        return self.data[idx]


# # 定义记录多次运行时间的Timer类
class Timer:
    """记录多次运行时间"""

    def __init__(self):
        """初始化"""
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并在列表中记录时间"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间的总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()



# # 计算一个batch中预测正确的个数
# 将预测值y_hat与真实值y进行比较，获取预测正确的个数
# accuracy_num返回的结果除以len(y)，则为准确率
def accurate_num(y_hat, y):
    """计算预测正确的数量"""
    # 这里的y_hat的行数为样本数，列数为分类数，即一行表示某个样本的计算结果（经过softmax后每行所有元素的和都为1）
    # y是1维向量，元素的个数对应样本数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 判断检查y_hat是一个矩阵
        y_hat = y_hat.argmax(axis=1)  # 获取每一行最大元素值的下标，即预测的分类的类别
    cmp = y_hat.type(y.dtype) == y  # 将y_hat转变为y的数据类型，然后作比较，cmp为布尔类型
    return float(cmp.type(y.dtype).sum())


# # 计算准确率
# 对于任意数据迭代器dataloader可访问的数据集，评估在任意模型上的准确率
# 实际上是计算一个epoch中的准确率
def evaluate_accuracy(test_dataloader, model, device):
    """计算在指定数据集上模型的精度"""
    if isinstance(model, torch.nn.Module):  # 判断检查model是不是torch.nn.Module类型
        model.eval()  # 将模型设置为评估模式

    # 实例化Accumulator对象，用于存储测试过程中的状态数据
    metric = Accumulator(3)  # 实例化Accumulator对象，累加器列数为3，测试损失总和、测试准确度总和、样本数

    # 执行测试
    with torch.no_grad():  # 不计算梯度，只前向传播
        for X, y in test_dataloader:  # 使用dataloader配合for循环,遍历每个batch
            # 将X，y放入GPU
            if isinstance(X, list):  # 如果X是list类型则按元素移至GPU
                X = [x.to(device) for x in X]
            else:  # 如果X是tensor类型则一次全部移动至GPU
                X = X.to(device)
            y = y.to(device)
            # 计算预测值
            y_hat = model(X)
            # 计算损失
            loss_value = loss_fun(y_hat, y)
            # 将（batch)测试损失总和、（batch)测试准确个数、（batch)样本数放入累加器
            metric.add(float(loss_value.sum()), accurate_num(y_hat, y), y.numel())
    # 返回测试损失和测试精度
    return metric[0] / metric[2], metric[1] / metric[2]



def train_from_scratch_main(model, train_dataloader, test_dataloader, num_epochs, loss_fun, optimizer, device, model_path):
    """在GPU中训练模型"""
    # 定义存储训练损失、训练精度、测试损失、测试精度的列表
    report_epoch = []  # Epoch级的报告,一个epoch占用一行
    report_batch = []  # Batch级的报告，一个batch占用一行，但不是每一个batch都有，仅包含训练损失、训练精度

    # Xavier Uniform模型初始化，在每一层网络保证输入和输出的方差相同
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)  # 应用Xavier Uniform初始化

    model=model.to(device)  # 将模型放入GPU

    timer, num_batches = Timer(), len(train_dataloader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2)

    #  执行每个epoch的训练，得到当前epoch的训练损失和训练精度
    for epoch in range(num_epochs):
        # 打印Epoch状态
        print(f"Epoch {epoch + 1}\n-------------------------------")
        print("learning rate:", optimizer.param_groups[0]['lr'])
        # 将模型设置为训练模式
        model.train()  # 将模型设置为训练模式

        # 实例化Accumulator对象，用于存储训练过程中的状态数据
        metric = Accumulator(3)  # 实例化Accumulator对象，累加器列数为3，训练损失总和、训练准确度总和、样本数

        # 执行每一个batch的训练
        for batch, (X, y) in enumerate(train_dataloader):  # 使用dataloader配合for循环，遍历每个batch
            timer.start()  # 开启定时器
            X, y = X.to(device), y.to(device)  # 将X， y移动至GPU
            # 计算预测值
            y_hat = model(X)
            # 计算损失
            loss_value = loss_fun(y_hat, y)
            # 计算梯度并更新预测值
            if isinstance(optimizer, torch.optim.Optimizer):  # 判断检查updater是否为torch.nn.Module类型
                """使用PyTorch内置的优化器和损失函数"""
                # 清除梯度
                optimizer.zero_grad()
                # 反向传播（计算梯度）
                loss_value.mean().backward()  # 计算整个batch中的损失平均值，mean()与交叉熵中reduction='none'有关
                # 更新参数
                optimizer.step()
            else:
                """使用定制的优化器和损失函数"""
                pass

            with torch.no_grad():
                # 将（batch)训练损失总和、（batch)训练准确度总和、（batch)样本数放入累加器
                metric.add(float(loss_value.sum()), accurate_num(y_hat, y), y.numel())

            timer.stop()  # 关闭定时器

            # 计算一个batch训练loss和accuracy
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            num_batches = len(train_dataloader)
            # %取模:返回除法的余数  //取整除:取商的整数部分
            if (batch + 1) % (num_batches // 5) == 0 or batch == num_batches - 1:
                # 存储至Batch级的报告
                report_batch.append([epoch + (batch + 1) / num_batches, train_loss, train_acc])
                # Batch级绘图
                report_plot_batch(num_epochs, report_batch, report_epoch)

        # 利用测试集，评估当前模型，返回当前epoch的测试损失和测试精度
        test_loss, test_acc = evaluate_accuracy(test_dataloader, model, device)

        scheduler.step()

        # 打印Epoch级的报告
        print('train_loss:', train_loss, '\ttrain_acc', train_acc)
        print('test_loss:', test_loss, '\ttest_acc', test_acc)
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')

        # # 存储至Epoch级的报告
        report_epoch.append([epoch + 1, train_loss, train_acc, test_loss, test_acc])
        report_plot_batch(num_epochs, report_batch, report_epoch)

        # 每个Epoch将报告存储至相应csv文件中(路径：/save/文件名.csv)
        save_path = os.path.join(data_dir, 'save')
        print(os.path.exists(save_path))
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
                     data=report_epoch).to_csv(save_path + "/report_epoch.csv", index=False)
        pd.DataFrame(columns=["epoch", "train_loss", "train_acc"],
                     data=report_batch).to_csv(save_path + "/report_batch.csv", index=False)
        print("report was saved\n")

        # # 保存每个Epoch的模型参数至文件(路径：/save/model_params_epoch下)
        epoch_params_file_path = os.path.join(save_path, 'model_params_epoch')
        if not os.path.exists(epoch_params_file_path):
            os.makedirs(epoch_params_file_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        # epoch_params_file = os.path.join(epoch_params_file_path, 'epoch%d_params.pth' % (epoch + 1))
        epoch_state = {'model_params': model.state_dict(),
                       'optimizer_params': optimizer.state_dict(),
                       'epoch': epoch + 1}
        torch.save(epoch_state, epoch_params_file_path + '/epoch%d_params.pth' % (epoch + 1))
        print("model_params_epoch was saved\n")

        # Epoch级绘图
        report_plot_epoch(num_epochs, report_epoch)
    torch.save(model.state_dict(), model_path)