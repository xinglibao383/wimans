import os
import datetime
import torch
from torch import nn
from tools.accumulator import Accumulator
from wimans import WiMANS, create_dataloaders
from timesformer.models.vit import MyTimeSformer, MyTimeSformerV2
from tools.logger import Logger


def accuracy(y1, y2, y1_hat, y2_hat):
    batch_size, num_users, num_classes = y2.shape

    y1, y1_hat = torch.argmax(y1, dim=2), torch.argmax(y1_hat, dim=2)
    y2, y2_hat = torch.argmax(y2, dim=2), torch.argmax(y2_hat, dim=2)

    acc1 = float((y1 == y1_hat).type(y1.dtype).sum() / (batch_size * num_users))
    acc2 = float((y2 == y2_hat).type(y1.dtype).sum() / (batch_size * num_users))

    return acc1, acc2


def evaluate(net, data_iter, loss_func):
    # y1训练损失之和, y2训练损失之和, 训练损失之和, y1正确预测的样本数, y2正确预测的样本数, 样本数
    metric = Accumulator(6)
    device = next(iter(net.parameters())).device
    net.eval()
    with torch.no_grad():
        for x, y1, y2 in data_iter:
            batch_size, num_users, _ = y2.shape
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            y1_hat, y2_hat = net(x)
            y1, y2, y1_hat, y2_hat = (y1.view(batch_size * num_users, 2), y2.view(batch_size * num_users, 10), 
                                      y1_hat.view(batch_size * num_users, 2), y2_hat.view(batch_size * num_users, 10))
            y1_indices = torch.argmax(y1, dim=1)
            y2_indices = torch.argmax(y2, dim=1)
            loss1 = loss_func(y1_hat, y1_indices)  # 二分类损失
            loss2 = loss_func(y2_hat, y2_indices)  # 多分类损失
            y1, y2, y1_hat, y2_hat = (y1.view(batch_size, num_users, 2), y2.view(batch_size, num_users, 10), 
                                      y1_hat.view(batch_size, num_users, 2), y2_hat.view(batch_size, num_users, 10))
            loss = loss1 + loss2
            eval_acc1, eval_acc2 = accuracy(y1, y2, y1_hat, y2_hat)
            metric.add(loss1.item() * x.shape[0], loss2.item() * x.shape[0], loss.item() * x.shape[0],
                       eval_acc1 * x.shape[0], eval_acc2 * x.shape[0], x.shape[0])

    return (metric[0] / metric[5], metric[1] / metric[5],
            metric[2] / metric[5], metric[3] / metric[5],
            metric[4] / metric[5])


def train(net, train_iter, eval_iter, learning_rate, num_epochs, patience, devices, checkpoint_save_dir_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # 在多个GPU上并行训练模型
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    best_state_dict = net.state_dict()
    min_eval_loss = float('inf')
    min_eval_loss_epoch = 0
    current_patience = 0

    for epoch in range(num_epochs):
        # y1训练损失之和, y2训练损失之和, 训练损失之和, y1正确预测的样本数, y2正确预测的样本数, 样本数
        metric = Accumulator(6)
        net.train()
        for i, (x, y1, y2) in enumerate(train_iter):
            batch_size, num_users, _ = y2.shape
            optimizer.zero_grad()
            x, y1, y2 = x.to(devices[0]), y1.to(devices[0]), y2.to(devices[0])
            y1_hat, y2_hat = net(x)

            y1, y2, y1_hat, y2_hat = y1.view(batch_size * num_users, 2), y2.view(batch_size * num_users, 10), y1_hat.view(batch_size * num_users, 2), y2_hat.view(batch_size * num_users, 10)
            # print(y1.shape, y2.shape, y1_hat.shape, y2_hat.shape)
            # 将 y1 和 y2 从 one-hot 编码转换为类别索引
            y1_indices = torch.argmax(y1, dim=1)
            y2_indices = torch.argmax(y2, dim=1)

            y2_indices = torch.where(
                y1_indices == 0,
                torch.zeros_like(y2_indices),  # 如果没有活动，y2_indices 设置为 0
                torch.argmax(y2[:, 1:], dim=1) + 1  # 如果有活动，取 y2 中除了第一个类别以外的最大值
            )

            loss1 = loss_func(y1_hat, y1_indices)  # 二分类损失
            loss2 = loss_func(y2_hat, y2_indices)  # 多分类损失

            y1, y2, y1_hat, y2_hat = y1.view(batch_size, num_users, 2), y2.view(batch_size, num_users, 10), y1_hat.view(batch_size, num_users, 2), y2_hat.view(batch_size, num_users, 10)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_acc1, train_acc2 = accuracy(y1, y2, y1_hat, y2_hat)
                metric.add(loss1.item() * x.shape[0], loss2.item() * x.shape[0], loss.item() * x.shape[0],
                           train_acc1 * x.shape[0], train_acc2 * x.shape[0], x.shape[0])
                
                if i % 20 == 0:
                    train_loss1, train_loss2, train_loss, _, _ = (metric[0] / metric[5], metric[1] / metric[5],
                                                                            metric[2] / metric[5], metric[3] / metric[5],
                                                                            metric[4] / metric[5])
                    print(f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.3f}, train loss1: {train_loss1:.3f}, train loss2: {train_loss2:.3f}')
                

        train_loss1, train_loss2, train_loss, train_acc1, train_acc2 = (metric[0] / metric[5], metric[1] / metric[5],
                                                                        metric[2] / metric[5], metric[3] / metric[5],
                                                                        metric[4] / metric[5])
        eval_loss1, eval_loss2, eval_loss, eval_acc1, eval_acc2 = evaluate(net, eval_iter, loss_func)
        logger.record([
            f'Epoch: {epoch}, current patience: {current_patience + 1}',
            f'train loss1: {train_loss1:.3f}, train loss2: {train_loss2:.3f}, train loss: {train_loss:.3f}',
            f'train acc1: {train_acc1:.3f}, train acc2: {train_acc2:.3f}',
            f'eval loss1: {eval_loss1:.3f}, eval loss2: {eval_loss2:.3f}, eval loss: {eval_loss:.3f}',
            f'eval acc1: {eval_acc1:.3f}, eval acc2: {eval_acc2:.3f}'
        ])

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint_save_path = os.path.join(checkpoint_save_dir_path, f"checkpoint-{epoch}.pth")
            torch.save(net.state_dict(), checkpoint_save_path)
        if eval_loss < min_eval_loss:
            best_state_dict = net.state_dict()
            min_eval_loss = eval_loss
            min_eval_loss_epoch = epoch
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                logger.record([f'Early stopping after {epoch + 1} epochs'])
                break

    torch.save(best_state_dict, os.path.join(checkpoint_save_dir_path, "best_state_dict.pth"))
    logger.record([f"The best testing loss occurred in the {min_eval_loss_epoch} epoch"])

    return os.path.join(checkpoint_save_dir_path, "best_state_dict.pth")


def test(net, pth_path, test_iter, device, logger):
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(pth_path))
    net.to(device)  # 将模型加载到正确的设备
    net.eval()  # 设置模型为评估模式

    loss_func = nn.CrossEntropyLoss()

    test_loss1, test_loss2, test_loss, test_acc1, test_acc2 = evaluate(net, test_iter, loss_func)
    logger.record([
        f'Test loss1: {test_loss1:.3f}, test loss2: {test_loss2:.3f}, test loss: {test_loss:.3f}',
        f'test acc1: {test_acc1:.3f}, test acc2: {test_acc2:.3f}'
    ])


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = os.path.join('./', 'outputs', timestamp)
    logger = Logger(save_path=output_save_path)

    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    # devices = [torch.device("cpu")]

    npy_dir = '/home/dataset/XLBWorkSpace/wimans/wifi_csi/amp/'  # .npy 文件所在的目录
    csv_file = '/home/dataset/XLBWorkSpace/wimans/annotation.csv'  # CSV 文件路径
    dataset = WiMANS(npy_dir, csv_file)
    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=16)

    # net = MyTimeSformer()
    net = MyTimeSformerV2(img_size=96, num_classes=10, num_frames=3000, attention_type='divided_space_time')

    pth_path = train(net, train_loader, val_loader, 0.0001, 300, 20, devices, output_save_path, logger)

    test(net, pth_path, test_loader, devices[0], logger) 