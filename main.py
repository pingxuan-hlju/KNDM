import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from early_stopping import EarlyStopping
import model

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(1206)


def train(model, train_set, test_set, embed, epoch, learn_rate, cross):

    optimizer = torch.optim.Adam(
        model.parameters(), learn_rate, weight_decay=5e-3)
    cost = nn.CrossEntropyLoss()
    embeds = embed.float().cuda()

    # early_stopping = EarlyStopping(
    #     patience=10, save_path='./checkpoint/best_network.pth')

    for i in range(epoch):
        model.train()
        LOSS = 0
        for x1, x2, y in train_set:
            x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
            out = model(x1, x2, embeds)
            loss = cost(out, y)
            LOSS += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Cross: %d  Epoch: %d / %d Loss: %0.5f" %
              (cross + 1, i + 1, epoch, LOSS))

        # early_stopping(LOSS, model)
        # if early_stopping.flag:
        #     print(f'early_stopping!')
        #     early_stop = 1
        #     test(model, test_set, cross, embeds)
        #     break
        # 如果到最后一轮了，保存测试结果
        if i + 1 == epoch:
            test(model, test_set, cross, embeds)


def test(model, test_set, cross, embeds):
    correct = 0
    total = 0

    predall, yall = torch.tensor([]), torch.tensor([])
    model.eval()  # 使Dropout失效

    # model.load_state_dict(torch.load('./checkpoint/best_network.pth'))
    for x1, x2, y in test_set:
        x1, x2, y = x1.long().to(device), x2.long().to(device), y.long().to(device)
        with torch.no_grad():
            pred = model(x1, x2, embeds)
            a = torch.max(pred, 1)[1]
        total += y.size(0)
        correct += (a == y).sum()
        predall = torch.cat(
            [predall, torch.as_tensor(pred, device='cpu')], dim=0)
        yall = torch.cat([yall, torch.as_tensor(y, device='cpu')])

    torch.save((predall, yall), './result/CNN_%d' % cross)  # 存放每折结果和标签
    print('Test_acc: ' + str((correct / total).item()))


class MyDataset(Dataset):
    def __init__(self, tri, dm):
        self.tri = tri
        self.dm = dm

    def __getitem__(self, idx):
        x, y = self.tri[idx, :]

        label = self.dm[x][y]
        return x, y, label

    def __len__(self):
        return self.tri.shape[0]


if __name__ == "__main__":
    learn_rate = 0.001
    epoch = 80
    batch = 128

    fea, train_xy, test_xy, asso_mat, _ = torch.load(
        'data/embed_index_adj_asso.pth')
    CL_ecst_fea, CL_gump_fea, _ = torch.load('data/con_embed_parameter.pth')
    #  451
    for i in range(5):
        ecst_fea, gump_fea = F.normalize(
            CL_ecst_fea[i], dim=1), F.normalize(CL_gump_fea[i], dim=1)
        X = torch.cat([fea[i].cuda(), ecst_fea, gump_fea], dim=1)
        net =model.CNN().to(device)
        train_set = DataLoader(
            MyDataset(train_xy[i], asso_mat), batch, shuffle=True)
        test_set = DataLoader(
            MyDataset(test_xy[i], asso_mat), batch, shuffle=False)
        train(net, train_set, test_set, X, epoch, learn_rate, i)