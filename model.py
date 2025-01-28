import os
import torch
import time
import warnings
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ordered_set import OrderedSet
from collections import defaultdict as ddict, Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(1206)

class ECST(torch.nn.Module):
    def __init__(self, params, t_idxs):
        super().__init__()
        self.params = params
        self.t_idxs = t_idxs
        self.fc = nn.Sequential(nn.Linear(2 * self.params.embed_dim, self.params.embed_dim),
                                nn.ReLU(), nn.Dropout(0.2), nn.Linear(self.params.embed_dim, self.params.embed_dim))

        self.ent_embed = torch.nn.Embedding(
            self.params.num_ent, self.params.embed_dim, padding_idx=None)
        nn.init.xavier_normal_(self.ent_embed.weight)
        self.rel_embed = torch.nn.Embedding(
            self.params.num_rel, self.params.embed_dim, padding_idx=None)
        nn.init.xavier_normal_(self.rel_embed.weight)
        self.nod_embed = torch.nn.Embedding(2, 32, padding_idx=None)
        nn.init.xavier_normal_(self.nod_embed.weight)

        self.W_Q, self.W_K, self.W_V = nn.Linear(
            288, 128), nn.Linear(160, 128), nn.Linear(160, 128)
        self.bceloss, self.mseloss = torch.nn.BCELoss(), torch.nn.MSELoss(reduction='mean')
        self.init_para()

    def init_para(self):
        nn.init.xavier_normal_(
            self.fc[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc[3].weight)
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_V.weight)

    def TransE(self, src, rel, dst):
        V_src, rel_emb, dst_emb = self.gengxin(
            src), self.rel_embed(rel), self.ent_embed(dst)
        # 计算正样本得分
        positive_scores = torch.norm(V_src + rel_emb - dst_emb, dim=1, p=2)
        # 随机替换尾实体->生成负样本
        neg_dst_list = []
        for i in range(len(src)):
            head, relation, tail = src[i], rel[i], dst[i]
            while True:
                # 随机生成一个尾实体的索引，确保不和原始正样本重复
                negative_entity = torch.randint(
                    0, self.params.num_ent, (1,)).item()
                if negative_entity != tail and negative_entity not in self.t_idxs[head]:
                    break
            # 将生成的负样本三元组添加到负样本列表中
            neg_dst_list.append(negative_entity)
        neg_dst = torch.Tensor(neg_dst_list).long()
        # 计算负样本得分
        neg_dst_emb = self.ent_embed(neg_dst.cuda())

        negative_scores = torch.norm(V_src + rel_emb - neg_dst_emb, dim=1, p=2)
        # 返回正样本和负样本得分
        # 计算损失
        loss_2 = torch.sum(-torch.log(torch.sigmoid(negative_scores - positive_scores)))
        return loss_2

    def pred(self, src, rel):
        V_src, rel_emb = self.gengxin(src), self.rel_embed(rel)  # 128*128
        # print(V_src.shape, rel_emb.shape, dst_emb.shape)
        # 主损失
        yc = torch.sigmoid(torch.matmul(self.fc(
            torch.cat((V_src, rel_emb), dim=1)), self.ent_embed.weight.transpose(1, 0)))
        return yc

    def kg_loss(self, pred, true_label):
        loss_1 = self.bceloss(pred, true_label)
        return loss_1

    def gengxin(self, src):
        src_emb = self.ent_embed(src)
        hnr_all = []
        for h_id in src:
            neighbor = self.t_idxs[h_id]
            a = []
            for i, t_id in enumerate(neighbor):
                # print(f'第{i+ 1}个邻居')
                h_emb = self.ent_embed(h_id).unsqueeze(0)
                n_emb = self.get_node(h_id).unsqueeze(0)
                r_emb = self.get_rel(h_id, t_id).unsqueeze(0)
                hnr_emb = torch.cat((h_emb, n_emb, r_emb), dim=1)
                a.append(hnr_emb)
            b = torch.cat(a, dim=0)
            q = self.W_Q(b).unsqueeze(0)
            # print(q.shape)
            hnr_all.append(q)
        hnr_emb = torch.cat(hnr_all, dim=0)  # (128,10, 288)

        list_fea = self.get_tail_node()
        k_list = []
        v_list = []         # len:128 , [x,y,z,... ]  x:10*160
        for i in src:
            k = self.W_K(list_fea[i]).unsqueeze(0)
            v = self.W_V(list_fea[i]).unsqueeze(0)
            k_list.append(k)
            v_list.append(v)
        fea2 = torch.cat(k_list, dim=0)
        fea3 = torch.cat(v_list, dim=0)

        fea_q, fea_k, fea_v = hnr_emb, fea2.transpose(1, 2), fea3
        q_k = torch.matmul(fea_q, fea_k)
        # 取对角线上的元素
        a = q_k[:, torch.arange(10), torch.arange(10)].unsqueeze(1)
        att_mat = torch.softmax(a / torch.sqrt(torch.tensor(128)), dim=1)
        # 残差
        V_src = src_emb + torch.matmul(att_mat, fea_v).squeeze()  # (128,128)
        return V_src

    # 准备 K,V
    def get_tail_node(self):
        tn_all = []
        for row_index, row in enumerate(self.t_idxs):  # 尾实体的索引集合
            # print(row)
            row_fea = []
            for index in row:
                index = index.long()
                # print(index) # gpu
                n_emb = self.get_node(index).unsqueeze(0)  # 1*32
                t_fea = self.ent_embed(index).unsqueeze(0)  # 1*128
                # print(t_fea.shape)
                tn_emb = torch.cat((t_fea, n_emb), dim=1)  # 1*160
                row_fea.append(tn_emb)
            row_fea = torch.cat(row_fea)
            tn_all.append(row_fea)   # [x,y...,z] ,x,y...z:10*160
        return tn_all

    # 获取节点属性的嵌入向量
    def get_node(self, t_id):
        if t_id < 1373:
            n_idx = 0
        else:
            n_idx = 1
        n_idx = torch.tensor(n_idx).long()  # cpu
        # print(n_idx.device)
        n_emb = self.nod_embed(n_idx.cuda())
        return n_emb
    # 获取关系

    def get_rel(self, src, t_id):
        if src < 1373:
            if t_id < 1373:
                r_idx = 0
            else:
                r_idx = 2
        else:
            if t_id < 1373:
                r_idx = 3
            else:
                r_idx = 1
        r_idx = torch.tensor(r_idx).long()  # cpu
        # print(n_idx.device)
        r_emb = self.rel_embed(r_idx.cuda())
        return r_emb
    
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class gnconv(nn.Module):
    def __init__(self, dim, order=3, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)][::-1]
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.dwconv = get_dwconv(sum(self.dims), 7, True)
        self.projs = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)])
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.scale = s

    def forward(self, x):
        # 输入通过输入通道数到两倍输出通道数的投影层
        # (batch_size, 32,1,128) -> (batch_size, 64, 1,128)
        x = self.proj_in(x)
        y, x = torch.split(x, (self.dims[0], sum(self.dims)), dim=1)
        # # 残差连接的部分y: (batch_size, 8, 1,128); 深度可分离卷积的输入x:(batch_size, 56, 1,128)
        # print(y.shape,x.shape)
        # 深度可分离卷积 x:(batch_size, 56, 1,128)
        x = self.dwconv(x) * self.scale  # (batch_size, 56, 1,128)
        # print(x.shape)
        x_list = torch.split(x, self.dims, dim=1)
        # [(batch_size, 8, 1,128),(batch_size, 16, 1,128),(batch_size, 32, 1,128)]
        # 第一个阶段的输出与投影层进行元素级别的相乘
        x = y * x_list[0]
        # 从第二个阶段开始，每个阶段的输出与对应的投影层进行元素级别的相乘
        for i in range(self.order - 1):  # 2
            conv = self.projs[i]
            # print(conv(x).shape,x_list[i + 1].shape)
            x = conv(x) * x_list[i + 1]
            # print(x.shape)
        # 最终的输出通过输出投影层
        return self.proj_out(x)


class GUMP(nn.Module):
    def __init__(self, fea1, fea2, fea3, dim, gnconv=gnconv):
        super(GUMP, self).__init__()
        self.dd, self.mm, self.dm = fea1, fea2, fea3
        self.X_D, self.X_M = torch.cat(
            (fea1, fea3), dim=1), torch.cat((fea3.T, fea2), dim=1)
        self.conv = nn.Conv2d(3, dim, 1)
        self.gnconv = gnconv(dim)
        self.fc1 = nn.Sequential(
            nn.Linear(1546, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128))
        self.fc2 = nn.Sequential(
            nn.Linear(1546, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128))
        self.fc3 = nn.Sequential(
            nn.Linear(1546, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128))
        self.fc4 = nn.Sequential(
            nn.Linear(1546, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128))
        self.fc5 = nn.Sequential(
            nn.Linear(1546, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128))
        self.fc6 = nn.Sequential(
            nn.Linear(1546, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128))
        self.cgb_mlp = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 1000), nn.ReLU(),
                                     nn.Dropout(0.5), nn.Linear(1000, 128))
        self.init_para()

    def init_para(self):
        nn.init.xavier_normal_(
            self.fc1[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc1[3].weight)
        nn.init.xavier_normal_(
            self.fc2[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc2[3].weight)
        nn.init.xavier_normal_(
            self.fc3[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc3[3].weight)
        nn.init.xavier_normal_(
            self.fc4[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc4[3].weight)
        nn.init.xavier_normal_(
            self.fc5[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc5[3].weight)
        nn.init.xavier_normal_(
            self.fc6[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc6[3].weight)
        nn.init.xavier_normal_(
            self.cgb_mlp[1].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.cgb_mlp[4].weight)

    def new_fea(self):
        # 获取 drug_fea;
        dd = (self.dd @ self.X_D).float()
        ddd = (self.dd @ self.dd @ self.X_D).float()
        dmd = (self.dm @ self.dm.T @ self.X_D).float()
        d1 = self.fc4(dd)
        d2 = self.fc5(ddd)
        d3 = self.fc6(dmd)
        # 获取 mirc_fea;
        mm = (self.mm @ self.X_M).float()
        mmm = (self.mm @ self.mm @ self.X_M).float()
        mdm = (self.dm.T @ self.dm @ self.X_M).float()
        m1 = self.fc1(mm)
        m2 = self.fc2(mmm)
        m3 = self.fc3(mdm)
        fea1 = torch.cat((d1, m1), dim=0)
        fea2 = torch.cat((d2, m2), dim=0)
        fea3 = torch.cat((d3, m3), dim=0)
        return fea1, fea2, fea3

    def forward(self):
        x1, x2, x3 = self.new_fea()
        x1, x2, x3 = F.normalize(x1, dim=1), F.normalize(
            x2, dim=1), F.normalize(x3, dim=1)
        # print(x1.shape,x2.shape,x3.shape)
        z1, z2, z3 = x1[None, None, :, :], x2[None,
                                              None, :, :], x3[None, None, :, :]
        z = torch.cat((z1, z2, z3), dim=1)
        # 升通道 64 2C
        z = self.conv(z)
        #  循环门控
        z_gnconv = self.gnconv(z)
        p_se = self.cgb_mlp(z_gnconv.reshape(1546, -1))
        return p_se
    
    
class HNCL(nn.Module):
    def __init__(self, kg_fea, pt_fea):
        super(HNCL, self).__init__()
        self.w1 = nn.Parameter(torch.rand(1))
        self.w2 = nn.Parameter(torch.rand(1))
        self.kg_fea = nn.Parameter(kg_fea)
        self.pt_fea = nn.Parameter(pt_fea)
        # 投影到损失空间
        self.fc_kg = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.fc_kg.weight)
        self.fc_pt = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.fc_pt.weight)
        self.relu = nn.ReLU()

    def forward(self, T):
        kg_fea = self.relu(self.fc_kg(self.kg_fea))
        pt_fea = self.relu(self.fc_pt(self.pt_fea))
        kg_fea, fea_x = F.normalize(
            kg_fea, dim=1), F.normalize(pt_fea, dim=1)  # 归一化
        kg_abs, x_abs = kg_fea.norm(dim=1), fea_x.norm(dim=1)  # L2范数
        sim_matrix = kg_fea @ fea_x.T / (kg_abs * x_abs + 1e-5)
        sim_matrix = torch.exp(sim_matrix / T)  # [1546,1546]
        pos_sim = torch.diag(sim_matrix)  # 分子--> 正例 [1546]
        # 根据节点类型不同--> 权重不同
        drug_M, micr_M = sim_matrix[:1373, :], sim_matrix[1373:, :]
        # 节点类型分类 drug[1373], micr[173]
        drug_pos, micr_pos = pos_sim[:1373], pos_sim[1373:]

        loss_dd = drug_pos / (drug_M[:, :1373].sum(dim=1))  # 与同种类型节点计算
        loss_dm = drug_pos / (drug_M[:, 1373:].sum(dim=1) + drug_pos)  # 与不同类型节点计算
            
        loss_mm = micr_pos / (micr_M[:, 1373:].sum(dim=1))  # 与同种类型节点计算
        loss_md = micr_pos / (micr_M[:, :1373].sum(dim=1) + micr_pos)  # 与不同类型节点计算
        
        alpha = torch.softmax(torch.cat([self.w1,self.w2]),dim=0)
        # loss_drug = self.w1 * loss_dd + self.w2 * loss_dm
        loss_drug = alpha[0]*loss_dd+alpha[1]*loss_dm
        loss_micr = alpha[0] * loss_mm + alpha[1] * loss_md
        # loss_drug = loss_dd+ loss_dm
        # loss_micr = loss_mm + loss_md
        loss = torch.cat((loss_drug, loss_micr), dim=0)
        loss = - torch.log(loss)
        return torch.mean(loss)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=(2, 1), stride=1, padding=1),
                                   # (batch_size, 32, 3, 1537)
                                   nn.ReLU(),
                                   nn.Conv2d(10, 20, kernel_size=(
                                       2, 2), stride=2, padding=1),
                                   # (batch_size, 16, 2, 769)
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        # (batch_size, 16, 1, 384)
        self.mlp = nn.Sequential(nn.Dropout(0.5), nn.Linear(
            20 * 451, 150), nn.ReLU(), nn.Dropout(0.5), nn.Linear(150, 2))
        self.init_para()

    def init_para(self):
        nn.init.xavier_normal_(
            self.mlp[1].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.mlp[4].weight)

    def forward(self, x, y, fea):
        x_drug = fea[x][:, None, None, :]  # x_lnc [32 1 1 1546]
        x_microbe = fea[y + 1373][:, None, None, :]  # x_dis [32 1 1 1546]
        x = torch.cat([x_drug, x_microbe], dim=2)  # [32 1 2 1546]
        return self.mlp(self.conv1(x).view(x.shape[0], -1))
