import os
import json
import datetime
import torch
from torch import nn
from tools.accumulator import Accumulator
from wimans_v3_adversarial import WiMANS, get_dataloaders_without_test
from tools.logger import Logger
from models.model_v3_adversarial import FeatureExtractor, Classifier, DomainDiscriminator


def accuracy(y, y_hat):
    return (torch.argmax(y_hat, dim=1) == y).float().mean().item()


def compute_four_accuracy(y1, y2, y3, domain, y1_hat, y2_hat, y3_hat, domain_hat):
    return accuracy(y1, y1_hat), accuracy(y2, y2_hat), accuracy(y3, y3_hat), accuracy(torch.argmax(domain.clone(), dim=1), domain_hat)


def evaluate(feature_extractor, classifier, domain_discriminator,
             data_iter, loss_func, task='123', lambda_d=0.1):
    # 验证损失之和, y1验证损失之和, y2验证损失之和, y3验证损失之和, 域验证损失之和, 样本数, y1正确预测的样本数, y2正确预测的样本数, y3正确预测的样本数, 域正确预测的样本数, 受试者数量
    metric = Accumulator(11)
    device = next(iter(feature_extractor.parameters())).device

    feature_extractor.eval()
    classifier.eval()
    domain_discriminator.eval()

    num_users = 6
    num_locations = 5 + 1
    num_activities = 9 + 1

    with torch.no_grad():
        for i, (x1, x2, y1, y2, y3, domain) in enumerate(data_iter):
            batch_size, _ = y1.shape
            x1, x2, y1, y2, y3, domain = x1.to(device), x2.to(device), y1.to(device), y2.to(device), y3.to(
                device), domain.to(device)

            x = feature_extractor(x1, x2)

            y1_hat, y2_hat, y3_hat = classifier(x)
            y1 = y1.view(batch_size * num_users)
            y2 = y2.view(batch_size * num_users)
            y3 = y3.view(batch_size * num_users)
            y1_hat = y1_hat.view(batch_size * num_users, 2)
            y2_hat = y2_hat.view(batch_size * num_users, num_locations)
            y3_hat = y3_hat.view(batch_size * num_users, num_activities)
            loss1 = loss_func[0](y1_hat, y1)
            loss2 = loss_func[1](y2_hat, y2)
            loss3 = loss_func[2](y3_hat, y3)

            domain_hat = domain_discriminator(x, domain)
            loss4 = loss_func[3](domain_hat, torch.argmax(domain.clone(), dim=1))

            if task == '123':
                loss = loss1 + loss2 + loss3 - lambda_d * loss4
            elif task == '3':
                loss = loss3 - lambda_d * loss4

            identity_acc, location_acc, activity_acc, domain_acc = compute_four_accuracy(y1, y2, y3, domain,
                                                                                         y1_hat, y2_hat, y3_hat,
                                                                                         domain_hat)
            metric.add(loss.item() * batch_size,
                       loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size,
                       loss4.item() * batch_size,
                       batch_size,
                       identity_acc * batch_size * num_users, location_acc * batch_size * num_users,
                       activity_acc * batch_size * num_users, domain_acc * batch_size,
                       batch_size * num_users)

    return (
        metric[0] / metric[5], metric[1] / metric[5], metric[2] / metric[5], metric[3] / metric[5],
        metric[4] / metric[5],
        metric[6] / metric[10], metric[7] / metric[10], metric[8] / metric[10],
        metric[9] / metric[5])


def train(feature_extractor, classifier, domain_discriminator,
          train_iter, eval_iter, learning_rate1, learning_rate2, weight_decay, num_epochs, patience,
          devices, checkpoint_save_dir_path, logger, task='123', lambda_d=0.1):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    feature_extractor.apply(init_weights)
    classifier.apply(init_weights)
    domain_discriminator.apply(init_weights)

    # 在多个GPU上并行训练模型
    feature_extractor = nn.DataParallel(feature_extractor, device_ids=devices).to(devices[0])
    classifier = nn.DataParallel(classifier, device_ids=devices).to(devices[0])
    domain_discriminator = nn.DataParallel(domain_discriminator, device_ids=devices).to(devices[0])

    optimizer_f = torch.optim.AdamW(feature_extractor.parameters(), lr=learning_rate1, weight_decay=weight_decay)
    optimizer_y = torch.optim.AdamW(classifier.parameters(), lr=learning_rate1, weight_decay=weight_decay)
    optimizer_d = torch.optim.AdamW(domain_discriminator.parameters(), lr=learning_rate2, weight_decay=weight_decay)

    identity_weights = torch.tensor([1.0, 1.375], dtype=torch.float32).to(devices[0])
    loss_func1 = nn.CrossEntropyLoss(weight=identity_weights)
    location_weights = torch.tensor([1.0, 6.661, 6.754, 6.981, 6.981, 6.981], dtype=torch.float32).to(devices[0])
    loss_func2 = nn.CrossEntropyLoss(weight=location_weights)
    activity_weights = torch.tensor([1.0, 12.375, 12.375, 12.375, 12.375, 12.375, 12.375, 12.375, 12.375, 12.375],
                                    dtype=torch.float32).to(devices[0])
    loss_func3 = nn.CrossEntropyLoss(weight=activity_weights)
    loss_func4 = nn.CrossEntropyLoss()

    min_eval_loss = float('inf')
    min_eval_loss_epoch = 0
    current_patience = 0

    num_users = 6
    num_locations = 5 + 1
    num_activities = 9 + 1

    for epoch in range(num_epochs):
        # 验证损失之和, y1验证损失之和, y2验证损失之和, y3验证损失之和, 域验证损失之和, 样本数, y1正确预测的样本数, y2正确预测的样本数, y3正确预测的样本数, 域正确预测的样本数, 受试者数量
        metric = Accumulator(11)
        feature_extractor.train()
        classifier.train()
        domain_discriminator.train()

        for i, (x1, x2, y1, y2, y3, domain) in enumerate(train_iter):
            batch_size, _ = y1.shape
            x1, x2, y1, y2, y3, domain = x1.to(devices[0]), x2.to(devices[0]), y1.to(devices[0]), y2.to(
                devices[0]), y3.to(devices[0]), domain.to(devices[0])

            domain_discriminator.zero_grad()
            x = feature_extractor(x1, x2)
            domain_hat = domain_discriminator(x.detach(), domain)  # 不计算梯度
            loss_d = loss_func4(domain_hat, torch.argmax(domain.clone(), dim=1))
            loss_d.backward()
            optimizer_d.step()
            feature_extractor.zero_grad()
            classifier.zero_grad()
            x = feature_extractor(x1, x2)
            y1_hat, y2_hat, y3_hat = classifier(x)
            domain_hat = domain_discriminator(x, domain)

            y1 = y1.view(batch_size * num_users)
            y2 = y2.view(batch_size * num_users)
            y3 = y3.view(batch_size * num_users)
            y1_hat = y1_hat.view(batch_size * num_users, 2)
            y2_hat = y2_hat.view(batch_size * num_users, num_locations)
            y3_hat = y3_hat.view(batch_size * num_users, num_activities)

            loss1 = loss_func1(y1_hat, y1)
            loss2 = loss_func2(y2_hat, y2)
            loss3 = loss_func3(y3_hat, y3)
            loss4 = loss_func4(domain_hat, torch.argmax(domain.clone(), dim=1))

            if task == '123':
                loss = loss1 + loss2 + loss3 - lambda_d * loss4
            elif task == '3':
                loss = loss3 - lambda_d * loss4

            loss.backward()
            optimizer_f.step()
            optimizer_y.step()

            with torch.no_grad():
                identity_acc, location_acc, activity_acc, domain_acc = compute_four_accuracy(y1, y2, y3, domain,
                                                                                             y1_hat, y2_hat, y3_hat,
                                                                                             domain_hat)
                metric.add(loss.item() * batch_size,
                           loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size,
                           loss4.item() * batch_size,
                           batch_size,
                           identity_acc * batch_size * num_users, location_acc * batch_size * num_users,
                           activity_acc * batch_size * num_users, domain_acc * batch_size,
                           batch_size * num_users)

                if i % 30 == 0:
                    train_loss, train_loss1, train_loss2, train_loss3, train_loss4 = (metric[0] / metric[5],
                                                                                      metric[1] / metric[5],
                                                                                      metric[2] / metric[5],
                                                                                      metric[3] / metric[5],
                                                                                      metric[4] / metric[5])
                    print(
                        f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.3f}, train identity loss: {train_loss1:.3f}, train location loss2: {train_loss2:.3f}, train activity loss: {train_loss3:.3f}, train domain loss: {train_loss4:.3f}')

        train_loss, train_loss1, train_loss2, train_loss3, train_loss4, train_acc1, train_acc2, train_acc3, train_acc4 = (
            metric[0] / metric[5],
            metric[1] / metric[5],
            metric[2] / metric[5],
            metric[3] / metric[5],
            metric[4] / metric[5],
            metric[6] / metric[10],
            metric[7] / metric[10],
            metric[8] / metric[10],
            metric[9] / metric[5])

        eval_loss, eval_loss1, eval_loss2, eval_loss3, eval_loss4, eval_acc1, eval_acc2, eval_acc3, eval_acc4 = evaluate(
            feature_extractor,
            classifier,
            domain_discriminator,
            eval_iter,
            [loss_func1,
             loss_func2,
             loss_func3,
             loss_func4], task, lambda_d)

        logger.record([
            # f"Epoch: {epoch}, current patience: {current_patience + 1}, learning rate: {optimizer.param_groups[0]['lr']:.6f}",
            f'Epoch: {epoch}, current patience: {current_patience + 1}',
            f'train loss: {train_loss:.3f}, train identity loss: {train_loss1:.3f}, train location loss: {train_loss2:.3f}, train activity loss: {train_loss3:.3f}, train domain loss: {train_loss4:.3f}',
            f'train identity acc: {train_acc1:.3f}, train location acc: {train_acc2:.3f}, train activity acc: {train_acc3:.3f}, train domain acc: {train_acc4:.3f}',
            f'eval loss: {eval_loss:.3f}, eval identity loss: {eval_loss1:.3f}, eval location loss: {eval_loss2:.3f}, eval activity loss: {eval_loss3:.3f}, eval domain loss: {eval_loss4:.3f}',
            f'eval identity acc: {eval_acc1:.3f}, eval location acc: {eval_acc2:.3f}, eval activity acc: {eval_acc3:.3f}, eval domain acc: {eval_acc4:.3f}',
        ])

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            min_eval_loss_epoch = epoch
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                logger.record([f'Early stopping after {epoch + 1} epochs'])
                break

    logger.record([f"The best testing loss occurred in the {min_eval_loss_epoch} epoch"])

    return os.path.join(checkpoint_save_dir_path, "best_state_dict.pth")


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = os.path.join('./', 'outputs', timestamp)
    logger = Logger(save_path=output_save_path)
    # devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2')]

    nperseg, noverlap, nfft, window, remove_static, remove_noise = 512, 384, 1024, 'hamming', True, True
    hidden_dim, nhead, encoder_layers, dropout1, dropout2, dropout3, dropout4 = 512, 8, 6, 0.3, 0.3, 0.3, 0.3
    batch_size, learning_rate1, learning_rate2, weight_decay, task = 8, 0.0001, 0.0003, 1e-4, '123'
    feature_extractor1_name, feature_extractor2_name = 'transformer', 'resnet'
    transformer_with_positional, lambda_d = True, 0.3
    num_epochs, patience = 1000, 500

    params = {
        'nperseg': nperseg,
        'noverlap': noverlap,
        'nfft': nfft,
        'window': window,
        'remove_static': remove_static,
        'remove_noise': remove_noise,
        'hidden_dim': hidden_dim,
        'nhead': nhead,
        'encoder_layers': encoder_layers,
        'dropout1': dropout1,
        'dropout2': dropout2,
        'dropout3': dropout3,
        'dropout4': dropout4,
        'batch_size': batch_size,
        'learning_rate1': learning_rate1,
        'learning_rate2': learning_rate2,
        'weight_decay': weight_decay,
        'task': task,
        'feature_extractor1_name': feature_extractor1_name,
        'feature_extractor2_name': feature_extractor2_name,
        'transformer_with_positional': transformer_with_positional,
        'lambda_d': lambda_d,
        'num_epochs': num_epochs,
        'patience': patience
    }
    logger.record([f'Adversarial training parameters: {json.dumps(params, indent=4)}'])

    dataset = WiMANS(root_path='/data/XLBWorkSpace/wimans',
                     nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window,
                     remove_static=remove_static, remove_noise=remove_noise)
    train_loader, val_loader = get_dataloaders_without_test(dataset, batch_size=batch_size)

    feature_extractor = FeatureExtractor(hidden_dim=hidden_dim, nhead=nhead, encoder_layers=encoder_layers,
                                         dropout1=dropout1, dropout2=dropout2,
                                         feature_extractor1_name=feature_extractor1_name,
                                         feature_extractor2_name=feature_extractor2_name,
                                         transformer_with_positional=transformer_with_positional)
    classifier = Classifier(hidden_dim=hidden_dim, dropout=dropout3)
    domain_discriminator = DomainDiscriminator(hidden_dim=hidden_dim, dropout=dropout4)

    train(feature_extractor, classifier, domain_discriminator, train_loader, val_loader, learning_rate1,
          learning_rate2,
          weight_decay, num_epochs,
          patience, devices, output_save_path, logger, task, lambda_d)
