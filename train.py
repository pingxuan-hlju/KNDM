import os
import torch
import time
import numpy as np
import torch.nn as nn
import model
import argparse
import warnings
from scipy.io import loadmat
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict as ddict, Counter
from torch.utils.data import DataLoader, Dataset
from early_stopping import EarlyStopping
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(1206)

class MDDataset(Dataset):
    def __init__(self, triples, split, params):
        '''
        Args:
        ----------
            triples: tuple
                三元组(头实体，关系， 尾实体)
            split: str
                判断是否拆分
            params : dict
                参数
        '''
        self.triples = triples
        self.split = split
        self.params = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)
        # 标签平滑
        if self.split == 'train' and self.params.lbl_smooth != 0.0:
            label = (1.0 - self.params.lbl_smooth) * \
                label + (1.0 / self.params.num_ent)
        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.params.num_ent], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


def load_data(params, train_data, test_data):
    sr2d, data, tr_data = ddict(set), ddict(list), []
    for line in train_data:
        src_id, dst_id, rel_id = line
        src_id, rel_id, dst_id = int(float(src_id)), int(
            float(rel_id)), int(float(dst_id))
        data['train'].append((src_id, rel_id, dst_id))
    for line in test_data:
        src_id, dst_id, rel_id = line
        src_id, rel_id, dst_id = int(float(src_id)), int(
            float(rel_id)), int(float(dst_id))
        data['test'].append((src_id, rel_id, dst_id))

    # 获取train的所有值
    train_values = data.get('train')
    for value in train_values:
        src_id, rel_id, dst_id = value
        if rel_id == 0:
            tr_data.append((src_id, rel_id, dst_id))
            tr_data.append((dst_id, rel_id + 3, src_id))
            sr2d[(src_id, rel_id)].add(dst_id)
        if rel_id == 0:
            sr2d[(dst_id, rel_id + 3)].add(src_id)
    sr2d4tr = {k: list(v) for k, v in sr2d.items()}
    triples = ddict(list)
    for src_id, rel_id, dst_id in tr_data:
        # print(src_id, rel_id, dst_id)
        triples['train'].append(
            {'triple': (src_id, rel_id, dst_id), 'label': sr2d4tr[(src_id, rel_id)]})

    # 获取test的所有值
    test_values = data.get('test')
    for src_id, rel_id, dst_id in test_values:
        sr2d[(src_id, rel_id)].add(dst_id)
        sr2d[(dst_id, rel_id + 3)].add(src_id)
    sr2d4val_te = {k: list(v) for k, v in sr2d.items()}
    for src_id, rel_id, dst_id in test_values:
        triples['test'].append(
            {'triple': (src_id, rel_id, dst_id), 'label': sr2d4val_te[(src_id, rel_id)]})
    triples = dict(triples)

    def get_data_loader(dataset_class, split, batch_size, shuffle=True):
        return DataLoader(dataset_class(triples[split], split, params), batch_size=batch_size, shuffle=shuffle, collate_fn=dataset_class.collate_fn)
    data_iter = {
        'train': get_data_loader(MDDataset, 'train', params.batch_size),
        'test': get_data_loader(MDDataset, 'test', params.batch_size)}
    return data_iter, triples


# 知识图谱 评价指标
def evaluate(net, data_iter, params, split='test'):
    net.eval()
    with torch.no_grad():
        results = {}  # 存放结果的空字典
        train_iter = iter(data_iter[split])  # 迭代器 -> 返回每个批次
        for step, batch in enumerate(train_iter):
            triple, label = [_.to(params.device) for _ in batch]
            # (头实体，关系，0), label
            src, rel, dst, label = triple[:,
                                          0], triple[:, 1], triple[:, 2], label
            pred = net.pred(src, rel)  # (batch, 128, 1535)
            # 为目标预测创建索引 -> 创建一个从0到pred第一维度大小-1的张量;这用于索引批次中的每个样本
            b_range = torch.arange(pred.size()[0], device=params.device)
            # 获取目标预测值 ->  使用索引提取每个样本的目标预测值;这通常是为了计算损失或准确度
            target_pred = pred[b_range, dst]
            # 如果标签为真（非零），则将预测值设置为零；否则，保持原预测值不变。这通常用于忽略某些样本或关系在损失计算中的影响
            pred = torch.where(label.byte(), torch.zeros_like(pred), pred)
            # 将目标预测值放回预测张量中 -> 将之前提取的目标预测值放回预测张量的正确位置
            pred[b_range, dst] = target_pred
            pred = pred.cpu().numpy()
            dst = dst.cpu().numpy()
            for i in range(pred.shape[0]):
                # 提取当前样本的预测分数和目标
                scores = pred[i]
                target = dst[i]
                # 提取目标位置的预测分数并删除原分数
                tar_scr = scores[target]
                scores = np.delete(scores, target)
                # 随机插入目标分数
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                # 对分数进行排序并记录目标位置
                sorted_indices = np.argsort(-scores, kind='stable')
                _filter = np.where(sorted_indices == rand)[0][0]
                # 样本计数（count）、平均排名（MR）和平均倒数排名（MRR）
                results['count'] = 1 + results.get('count', 0.0)
                results['mr'] = (_filter + 1) + results.get('mr', 0.0)
                results['mrr'] = (1.0 / (_filter + 1)) + \
                    results.get('mrr', 0.0)
                # 计算Hits@k指标
                for k in range(10):
                    if _filter <= k:
                        results[f'hits@{k+ 1}'] = 1 + \
                            results.get(f'hits@{k+ 1}', 0.0)
    # 计算平均值
    results['mr'] = round(results['mr'] / float(results['count']), 5)
    results['mrr'] = round(results['mrr'] / float(results['count']), 5)
    for k in range(10):
        results[f'hits@{k+1}'] = round(results.get(f'hits@{k+ 1}',
                                                   0) / float(results['count']), 5)
    return results

def ecst_train(net, params, data_iter, cross):

    optimizer = torch.optim.Adam(
        net.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    early_stopping = EarlyStopping(
        patience=10, save_path='./checkpoint/best_network.pth')
    for epoch in range(params.max_epochs):
        # train
        net.train()
        losses = []
        train_iter = iter(data_iter['train'])
        for step, batch in enumerate(train_iter):
            optimizer.zero_grad()
            triple, label = [_.to(params.device) for _ in batch]
            src, rel, dst, label = triple[:,
                                          0], triple[:, 1], triple[:, 2], label
            pred = net.pred(src, rel)
            loss1 = net.kg_loss(pred, label)
            loss2 = net.TransE(src, rel, dst)
            loss = loss1 + 0.1 * loss2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 1000 == 0:
                print(
                    f'Cross:{cross + 1}, epoch: {epoch+1}, Train loss: {np.mean(losses)}, rel ratio: {((rel== 0)| (rel== 5)).sum()/ len(rel)}, acc: {(torch.where(pred< 0.5, 0, 1)== torch.where(label< 0.5, 0, 1)).sum()/ label.shape[0]/ label.shape[1]}')

        # test
        results = evaluate(net, data_iter, params, 'test')
        print(
            f'val mr: {results["mr"]}, mrr: {results["mrr"]}, hits10: {results["hits@10"]}')
        early_stopping(-results['mrr'], net)
        if early_stopping.flag == True:
            break


def hncl_train(model, temperature, epoch, learn_rate, cross):
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    for ep in range(epoch):
        time_start = time.time()
        loss = model(temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_end = time.time()
        print(
            f'Cross: {cross+1},epoch: {ep+ 1}, loss:{loss}, time: {time_end- time_start}')


if __name__ == "__main__":
    dim = 32
    learn_rate = 0.001
    epoch = 30
    temperature = 0.1
    parser = argparse.ArgumentParser(description='Parser for Arguments')
    parser.add_argument('-num_ent', type=int, default=1373 + 173)
    parser.add_argument('-num_rel', type=int, default=4)
    parser.add_argument('-num_node', type=int, default=2)
    parser.add_argument('-num_drug', type=int, default=1373)
    parser.add_argument('-num_micr', type=int, default=173)
    parser.add_argument('-tr_val_te_path', type=str, default='kg_data/')
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-lbl_smooth', type=float, default=0.2,
                        help='Label smoothing enable or disable')
    # parser.add_argument('-embed_dim', type= int, default= 128)
    parser.add_argument('-embed_dim', type=int, default=128)  # 1546
    parser.add_argument('-rel_dim', type=int, default=128)
    parser.add_argument('-node_dim', type=int, default=32)
    parser.add_argument('-dp', type=float, default=0.1)
    parser.add_argument('-device', type=str, default='cuda:0')  # cuda:0
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-patience', type=int, default=40)    # 早停
    parser.add_argument('-max_epochs', type=int, default=1000)  # 训练轮次
    params = parser.parse_args([])

    # ECST_train
    dics = torch.load('data/top_k.pth')
    train = torch.load('data/train.pth')
    test = torch.load('data/test.pth')
    ecst_fea = []
    for i in range(5):
        t_idxs = dics[i].to(params.device)
        data_iter, triples = load_data(params, train[i], test[i])
        net = model.ECST(params, t_idxs).to(params.device)
        ecst_train(net, params, data_iter, i)
        for params_tensor in net.state_dict():
            print(params_tensor, '\t', net.state_dict()[params_tensor].size())
        emb = net.state_dict()['ent_embed.weight']
        ecst_fea.append(emb)

    torch.save(ecst_fea, 'data/ecst_fea.pth')
    
    # gump
    gump_fea = []
    drug_simi_mat, micr_simi_mat, asso_mat_mask = torch.load(
        'data/gump_embeds.pth')
    for i in range(5):
        # print(f'第{i + 1}折')
        net = model.GUMP(drug_simi_mat, micr_simi_mat,
                  asso_mat_mask[i].to(drug_simi_mat.dtype), 32)
        fea_g = net()
        gump_fea.append(fea_g)

    torch.save(gump_fea, 'data/gump_fea.pth')

    # hncl_train
    ecst_fea = torch.load('data/ecst_fea.pth')
    gump_fea = torch.load('data/gump_fea.pth')
    para = []
    CL_ecst_fea, CL_gump_fea = [], []
    for i in range(5):
        print(f'第{i + 1}折')
        attenkg, gum = ecst_fea[i].to(device), gump_fea[i].to(device)
        net = model.HNCL(attenkg, gum).to(device)
        hncl_train(net, temperature, epoch, learn_rate, i)
        meter = net.state_dict()['w1'], net.state_dict()['w2']
        fea0 = net.state_dict()['kg_fea']
        fea1 = net.state_dict()['pt_fea']
        para.append(meter)
        CL_ecst_fea.append(fea0)
        CL_gump_fea.append(fea1)
    torch.save([CL_ecst_fea, CL_gump_fea, para],
               'data/con_embed_parameter.pth')