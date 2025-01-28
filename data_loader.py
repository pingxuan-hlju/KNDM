import os
import torch
import warnings
import numpy as np
from scipy.io import loadmat
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold


# 设置随机种子
torch.manual_seed(1206)

#  归一化的邻接矩阵


def Regularization(adj):
    row = torch.zeros(1373)
    col = torch.zeros(173)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                row[i] += 1
                col[j] += 1
    row = torch.sqrt(row)
    col = torch.sqrt(col)
    a = torch.Tensor([1])
    ADJ = torch.zeros(size=(1373, 173))
    for m in range(adj.shape[0]):
        for n in range(adj.shape[1]):
            if adj[m][n] == 1:
                temp = row[m] * col[n]
                ADJ[m][n] = torch.div(a, temp)

    return ADJ


# 获取特征矩阵
drug_simi_mat = torch.from_numpy(np.loadtxt('data/drugsimilarity.txt'))
micr_simi_mat = torch.from_numpy(
    np.loadtxt('data/microbe_microbe_similarity.txt'))
asso_mat = torch.from_numpy(
    loadmat('data/net1.mat', mat_dtype=True)['interaction'])

# 获取索引值
drug_inter_adj = drug_simi_mat.nonzero()
micr_inter_adj = micr_simi_mat.nonzero()
drug_micr_adj = asso_mat.nonzero()

# 添加offset
drug_offset, micr_offset = 0, 1373
drug_inter_adj = drug_inter_adj + torch.tensor([drug_offset, drug_offset])
micr_inter_adj = micr_inter_adj + torch.tensor([micr_offset, micr_offset])
drug_micr_adj = drug_micr_adj + torch.tensor([drug_offset, micr_offset])

# @ tensor 版本的shuffle 按维度0


def tensor_shuffle(ts, dim=0):
    return ts[torch.randperm(ts.shape[dim])]


pos_xy = asso_mat.nonzero()  # 所有正例坐标 torch.Size([2470, 2])
# 所有负例坐标 torch.Size([235059, 2])
neg_xy = tensor_shuffle((asso_mat == 0).nonzero(), dim=0)
rand_num_4940 = torch.randperm(4940)  # 2470* 2
neg_xy, rest_neg_xy = neg_xy[0: len(pos_xy)], neg_xy[len(pos_xy):]  # 打乱之后的负例
pos_neg_xy = torch.cat((pos_xy, neg_xy), dim=0)[rand_num_4940]

kflod = KFold(n_splits=5, shuffle=False)


def split_dataset(data, rel):
    train_list = []
    test_list = []
    for fold, (train_xy_idx, test_xy_idx) in enumerate(kflod.split(data)):
        # print(f'第{fold + 1}折')
        train = data[train_xy_idx, ]
        train_rels = np.ones(len(train)) * rel
        train_list.append(np.insert(train, 2, train_rels, axis=1))
        test = data[test_xy_idx]
        test_rels = np.ones(len(test)) * rel
        test_list.append(np.insert(test, 2, test_rels, axis=1))
    return train_list, test_list


# 划分数据集
drug_micr_train, drug_micr_test = split_dataset(drug_micr_adj, 0)
drug_inter_train, _ = split_dataset(drug_inter_adj, 1)
micr_inter_train, _ = split_dataset(micr_inter_adj, 2)

# 生成文件 -> 三元组(h, t, r)
test = drug_micr_test
train = []
for i in range(5):
    train_triad = np.concatenate(
        [drug_micr_train[i], drug_inter_train[i], micr_inter_train[i]], axis=0)
    train.append(train_triad)

# @ mask test
train_xy = []
test_xy = []
asso_mat_mask = []
fea = []
for fold, (train_xy_idx, test_xy_idx) in enumerate(kflod.split(pos_neg_xy)):
    # print(f'第{fold + 1}折')
    train_xy.append(pos_neg_xy[train_xy_idx, ])  # 每折的训练集坐标 len=5
    ts = pos_neg_xy[test_xy_idx]
    test_all = torch.cat([ts, rest_neg_xy], dim=0)
    test_xy.append(test_all)        # 每折的测试集坐标 len=5
    # @ mask test
    asso_mat_zy = asso_mat.clone()
    for index in ts:
        if asso_mat[index[0]][index[1]] == 1:
            asso_mat_zy[index[0]][index[1]] = 0
    asso_mat_zy = Regularization(asso_mat_zy)
    DD_DM = torch.cat([drug_simi_mat, asso_mat_zy], dim=1)
    DM_MM = torch.cat([asso_mat_zy.T, micr_simi_mat], dim=1)
    embed = torch.cat([DD_DM, DM_MM], dim=0)  # 生成embedding -> [1546,1546]
    asso_mat_mask.append(asso_mat_zy)
    fea.append(embed)


# 头实体前top_k 个邻居
dics = []
for i, x in enumerate(fea):
    # print(f'第{i + 1}折')
    t_idxs = []
    for row_index, row in enumerate(x):
        top_values, top_indices = torch.topk(row, k=10, dim=0)
        t_idxs.append(top_indices.tolist())
    t_idxs = torch.Tensor(t_idxs)
    dics.append(t_idxs)

torch.save([fea, train_xy, test_xy, asso_mat, asso_mat_mask],
           'data/embed_index_adj_asso.pth')
torch.save(dics, 'data/top_k.pth')
torch.save(train, 'data/train.pth')
torch.save(test, 'data/test.pth')
torch.save([drug_simi_mat, micr_simi_mat, asso_mat_mask],
           'data/gump_embeds.pth')
