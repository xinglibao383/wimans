import os
import datetime
import torch
from torch import nn
from tools.accumulator import Accumulator
from wimans_v3 import WiMANS, get_dataloaders, get_dataloaders_random_split, get_dataloaders_without_test
from tools.logger import Logger
from models.model_v1 import MyModel


def accuracy(y, y_hat):
    return (torch.argmax(y_hat, dim=1) == y).float().mean().item()


def compute_three_accuracy(y1, y2, y3, y1_hat, y2_hat, y3_hat):
    return accuracy(y1, y1_hat), accuracy(y2, y2_hat), accuracy(y3, y3_hat)


def evaluate(net, data_iter, loss_func):
    # 训练损失之和, y1训练损失之和, y2训练损失之和, y3训练损失之和, 样本数, y1正确预测的样本数, y2正确预测的样本数, y3正确预测的样本数, 受试者数量
    metric = Accumulator(9)
    device = next(iter(net.parameters())).device
    net.eval()

    num_users = 6
    num_locations = 5 + 1
    num_activities = 9 + 1

    with torch.no_grad():
        for i, (x1, x2, y1, y2, y3) in enumerate(data_iter):
            batch_size, _ = y1.shape
            x1, x2, y1, y2, y3 = x1.to(device), x2.to(device), y1.to(device), y2.to(device), y3.to(device)
            y1_hat, y2_hat, y3_hat = net(x1, x2)

            y1 = y1.view(batch_size * num_users)
            y2 = y2.view(batch_size * num_users)
            y3 = y3.view(batch_size * num_users)
            y1_hat = y1_hat.view(batch_size * num_users, 2)
            y2_hat = y2_hat.view(batch_size * num_users, num_locations)
            y3_hat = y3_hat.view(batch_size * num_users, num_activities)

            loss1 = loss_func[0](y1_hat, y1)
            loss2 = loss_func[1](y2_hat, y2)
            loss3 = loss_func[2](y3_hat, y3)

            loss = loss1 + loss2 + loss3

            identity_acc, location_acc, activity_acc = compute_three_accuracy(y1, y2, y3, y1_hat, y2_hat, y3_hat)
            metric.add(loss.item() * batch_size,
                       loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size,
                       batch_size,
                       identity_acc * batch_size * num_users, location_acc * batch_size * num_users,
                       activity_acc * batch_size * num_users,
                       batch_size * num_users)

    return (metric[0] / metric[4],
            metric[1] / metric[4], metric[2] / metric[4], metric[3] / metric[4],
            metric[5] / metric[8], metric[6] / metric[8], metric[7] / metric[8])


def train(net, train_iter, eval_iter, learning_rate, weight_decay, num_epochs, patience, 
          devices, checkpoint_save_dir_path, logger, use_scheduler=False, task=[True, True, True]):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # 在多个GPU上并行训练模型
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.75)

    identity_weights = torch.tensor([1.0, 1.375], dtype=torch.float32).to(devices[0])
    loss_func1 = nn.CrossEntropyLoss(weight=identity_weights)

    location_weights = torch.tensor([1.0, 6.661, 6.754, 6.981, 6.981, 6.981], dtype=torch.float32).to(devices[0])
    loss_func2 = nn.CrossEntropyLoss(weight=location_weights)

    activity_weights = torch.tensor([1.0, 12.375, 12.375, 12.375, 12.375, 12.375, 12.375, 12.375, 12.375, 12.375], dtype=torch.float32).to(devices[0])
    loss_func3 = nn.CrossEntropyLoss(weight=activity_weights)

    best_state_dict = net.state_dict()
    min_eval_loss = float('inf')
    min_eval_loss_epoch = 0
    current_patience = 0

    num_users = 6
    num_locations = 5 + 1
    num_activities = 9 + 1

    for epoch in range(num_epochs):
        # 训练损失之和, y1训练损失之和, y2训练损失之和, y3训练损失之和, 样本数, y1正确预测的样本数, y2正确预测的样本数, y3正确预测的样本数, 受试者数量
        metric = Accumulator(9)
        net.train()
        for i, (x1, x2, y1, y2, y3) in enumerate(train_iter):
            batch_size, _ = y1.shape
            optimizer.zero_grad()
            x1, x2, y1, y2, y3 = x1.to(devices[0]), x2.to(devices[0]), y1.to(devices[0]), y2.to(devices[0]), y3.to(devices[0])
            y1_hat, y2_hat, y3_hat = net(x1, x2)

            y1 = y1.view(batch_size * num_users)
            y2 = y2.view(batch_size * num_users)
            y3 = y3.view(batch_size * num_users)
            y1_hat = y1_hat.view(batch_size * num_users, 2)
            y2_hat = y2_hat.view(batch_size * num_users, num_locations)
            y3_hat = y3_hat.view(batch_size * num_users, num_activities)

            loss1 = loss_func1(y1_hat, y1)
            loss2 = loss_func2(y2_hat, y2)
            loss3 = loss_func3(y3_hat, y3)

            loss = loss1 * task[0] + loss2 * task[1] + loss3 * task[2]

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                identity_acc, location_acc, activity_acc = compute_three_accuracy(y1, y2, y3, y1_hat, y2_hat, y3_hat)
                metric.add(loss.item() * batch_size,
                           loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size,
                           batch_size,
                           identity_acc * batch_size * num_users, location_acc * batch_size * num_users, activity_acc * batch_size * num_users,
                           batch_size * num_users)

                if i % 10 == 0:
                    train_loss, train_loss1, train_loss2, train_loss3 = (metric[0] / metric[4],
                                                                         metric[1] / metric[4], metric[2] / metric[4],
                                                                         metric[3] / metric[4])
                    print(
                        f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.3f}, train identity loss: {train_loss1:.3f}, train location loss2: {train_loss2:.3f}, train activity loss: {train_loss3:.3f}')

        train_loss, train_loss1, train_loss2, train_loss3, train_acc1, train_acc2, train_acc3 = (metric[0] / metric[4],
                                                                                                 metric[1] / metric[4],
                                                                                                 metric[2] / metric[4],
                                                                                                 metric[3] / metric[4],
                                                                                                 metric[5] / metric[8],
                                                                                                 metric[6] / metric[8],
                                                                                                 metric[7] / metric[8])

        eval_loss, eval_loss1, eval_loss2, eval_loss3, eval_acc1, eval_acc2, eval_acc3 = evaluate(net, eval_iter,
                                                                                                  [loss_func1,
                                                                                                   loss_func2,
                                                                                                   loss_func3])
        if use_scheduler:
            scheduler.step(eval_loss)

        logger.record([
            # f"Epoch: {epoch}, current patience: {current_patience + 1}, learning rate: {optimizer.param_groups[0]['lr']:.6f}",
            f'Epoch: {epoch}, current patience: {current_patience + 1}',
            f'train loss: {train_loss:.3f}, train identity loss: {train_loss1:.3f}, train location loss: {train_loss2:.3f}, train activity loss: {train_loss3:.3f}',
            f'train identity acc: {train_acc1:.3f}, train location acc: {train_acc2:.3f}, train activity acc: {train_acc3:.3f}',
            f'eval loss: {eval_loss:.3f}, eval identity loss: {eval_loss1:.3f}, eval location loss: {eval_loss2:.3f}, eval activity loss: {eval_loss3:.3f}',
            f'eval identity acc: {eval_acc1:.3f}, eval location acc: {eval_acc2:.3f}, eval activity acc: {eval_acc3:.3f}',
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


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = os.path.join('./', 'outputs', timestamp)
    logger = Logger(save_path=output_save_path)

    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]

    dataset = WiMANS(root_path='/data/XLBWorkSpace/wimans', 
                     nperseg=1024, noverlap=256, nfft=2048, window='hamming', remove_static=True)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=64)
    # train_loader, val_loader, test_loader = get_dataloaders_random_split(dataset, batch_size=64)

    net = MyModel(hidden_dim=512, nhead=8, encoder_layers=6, dropout1=0.1, dropout2=0.1)

    pth_path = train(net, train_loader, val_loader, 0.0001, 1e-5, 300, 50, devices, output_save_path, logger)
