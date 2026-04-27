import copy
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from model import Model

from numpy.core import multiarray
import matplotlib.pyplot as plt
from hypergraph_utils import *

import os
from function import create_resultlist
from utils import f1_score_binary, precision_binary, recall_binary, mcc_binary, accuracy_binary
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from NMF import *
import randomfusion
from scipy import interp

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 设置随机数种子
seed = 48
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作的确定性
torch.backends.cudnn.benchmark = False  # 禁用卷积算法选择，确保结果可重复
# 检查 PyTorch 是否支持 GPU
print("CUDA 是否可用:", torch.cuda.is_available())

# 查看当前 GPU 设备的名称
if torch.cuda.is_available():
    print("GPU 设备名称:", torch.cuda.get_device_name(0))
else:
    print("当前运行在 CPU 上")


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def contrastive_loss(h1, h2, tau=0.4):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


def train(epochs):
    auc1 = 0
    aupr1 = 0
    recall1 = 0
    precision1 = 0
    f11 = 0
    mcc1 = 0
    accuracy1 = 0

    if epoch != epochs - 1:
        model.train()
        reconstructionG, reconstructionVAE, result, recover, mir_feature_1, mir_feature_2 = model(
            m_fusion_sim, d_fusion_sim, mir_feat, dis_feat, PCAassociationT, PCAassociation)

        outputs = recover.t().cpu().detach().numpy()
        test_predict = create_resultlist(outputs, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,
                                         Index_zeroCol, len(test_p), zero_length, test_f)

        # 掩码矩阵 - 使用所有训练样本
        MA = torch.masked_select(A, train_mask_tensor)

        # 总损失 - 使用所有训练样本
        rec = torch.masked_select(recover.t(), train_mask_tensor)

        # 超图的重构 - 使用所有训练样本
        reH = torch.masked_select(reconstructionG.t(), train_mask_tensor)

        # 计算正样本权重，用于处理类别不平衡
        pos_weight_value = torch.tensor([pos_weight]).cuda() if args.cuda else torch.tensor([pos_weight])

        # 超图损失 - 使用所有样本
        lossH = F.binary_cross_entropy_with_logits(reH.t(), MA, pos_weight=pos_weight_value)

        # 总损失 - 使用所有样本
        loss = lossH + F.binary_cross_entropy_with_logits(rec.t(), MA, pos_weight=pos_weight_value)

        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()

        auc_val = roc_auc_score(label, test_predict)
        aupr_val = average_precision_score(label, test_predict)

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.5f}'.format(loss.data.item()),
              'auc_val: {:.5f}'.format(auc_val),
              'aupr_val: {:.5f}'.format(aupr_val),
              )
        max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),
                                                  torch.from_numpy(test_predict).float())
        print("//////////max_f1_score", max_f1_score)
        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////precision:", precision)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////recall:", recall)

        mcc = mcc_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////mcc:", mcc)
        accuracy = accuracy_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////accuracy:", accuracy)

        print(
            '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    fpr, tpr = [], []
    if epoch == args.epochs - 1:
        auc1 = auc_val
        aupr1 = aupr_val
        recall1 = recall
        precision1 = precision
        f11 = max_f1_score
        mcc1 = mcc
        accuracy1 = accuracy

        print('auc_test: {:.5f}'.format(auc1),
              'aupr_test: {:.5f}'.format(aupr1),
              'precision_test: {:.5f}'.format(precision1),
              'recall_test: {:.5f}'.format(recall1),
              'f1_test: {:.5f}'.format(f11),
              'mcc_test: {:.5f}'.format(mcc1),
              'accuracy_test: {:.5f}'.format(accuracy1),
              )

        # 为了画图
        fpr, tpr, thresholds = roc_curve(label, test_predict)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(roc_auc)

    return auc1, aupr1, recall1, precision1, f11, mcc1, accuracy1, all_fpr, all_tpr, all_auc, fpr, tpr


# circ-disease(533,89)
MD = np.loadtxt("circRNAdisease(533,89)/association.txt")
C1 = np.loadtxt("circRNAdisease(533,89)/GKGIP_circRNA.txt")
D1 = np.loadtxt("circRNAdisease(533,89)/GKGIP_disease.txt")

C2 = np.loadtxt("circRNAdisease(533,89)/LKGIP_circRNA.txt")
D2 = np.loadtxt("circRNAdisease(533,89)/LKGIP_disease.txt")

# ------------------------------------------------------------------------
# 以下是超图相似性随机游走融合模块
# 高斯核相似性
HHMMG = construct_H_with_KNN(C1)
HMM1 = generate_G_from_H(HHMMG)  # circRNA相似性
HMM1 = HMM1.double()

HHDDG = construct_H_with_KNN(D1)  # 药物相似性
HDD1 = generate_G_from_H(HHDDG)
HDD1 = HDD1.double()

# 拉普拉斯相似性
HHMML = construct_H_with_KNN(C2)
HMM2 = generate_G_from_H(HHMML)  # circRNA相似性
HMM2 = HMM2.double()

HHDDL = construct_H_with_KNN(D2)  # 药物相似性
HDD2 = generate_G_from_H(HHDDL)
HDD2 = HDD2.double()

HHMG = construct_H_with_KNN(MD)  # 全局circRNA-药物关联
HMG = generate_G_from_H(HHMG)
HMG = HMG.double()

HHDG = construct_H_with_KNN(MD.T)  # 全局药物-circRNA关联
HDG = generate_G_from_H(HHDG)
HDG = HDG.double()

D_S_list = [HMM1, HMM2, HMG]
M_S_list = [HDD1, HDD2, HDG]
# 设置参数
beta1, beta2 = 0.3, 0.3
max_iter1, max_iter2 = 1, 4

# 调用两层双随机游走函数
m_fusion_sim, d_fusion_sim = randomfusion.two_tier_bi_random_walk(D_S_list, M_S_list, beta1, beta2, max_iter1,
                                                                  max_iter2)

[row, col] = np.shape(MD)
indexn = np.argwhere(MD == 0)
Index_zeroRow = indexn[:, 0]
Index_zeroCol = indexn[:, 1]
indexp = np.argwhere(MD == 1)
Index_PositiveRow = indexp[:, 0]
Index_PositiveCol = indexp[:, 1]
totalassociation = np.size(Index_PositiveRow)  # 7694
fold = int(totalassociation / 5)  # 1538

zero_length = np.size(Index_zeroRow)  # 321601

seed = 47
alpha = 0.7
n = 1
hidden1 = 512
hidden2 = 128
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cv_num', type=int, default=5, help='number of fold')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

AAuc_list1 = []
f1_score_list1 = []
precision_list1 = []
recall_list1 = []
aupr_list1 = []

auc_sum = 0
aupr_sum = 0
AUC = 0
AUPR = 0
recall_sum = 0
precision_sum = 0
f1_sum = 0
mcc_sum = 0
acc_sum = 0
accuracy_sum = 0

# 绘制ROC曲线
tprs = []
aucs = []
all_fpr, all_tpr, all_auc = [], [], []
mean_fpr = np.linspace(0, 1, 100)
# 绘制PR曲线
all_precision, all_recall, all_aupr = [], [], []
aupr_sum, time = 0, 0

for time in range(1, n + 1):
    Auc_per = []
    f1_score_per = []
    precision_per = []
    recall_per = []
    aupr_per = []
    p = np.random.permutation(totalassociation)

    AUC = 0
    aupr = 0
    rec = 0
    pre = 0
    f1 = 0
    mcc = 0
    accuracy = 0

    # 5-折
    for f in range(1, args.cv_num + 1):
        print("cross_validation:", '%01d' % (f))

        if f == args.cv_num:
            testset = p[((f - 1) * fold): totalassociation + 1]
        else:
            testset = p[((f - 1) * fold): f * fold]

        all_f = np.random.permutation(np.size(Index_zeroRow))

        test_p = list(testset)
        test_f = all_f[0:len(test_p)]

        difference_set_f = list(set(all_f).difference(set(test_f)))
        train_f = difference_set_f
        train_p = list(set(p).difference(set(testset)))

        X = copy.deepcopy(MD)
        Xn = copy.deepcopy(X)

        zero_index = []
        for ii in range(len(train_f)):
            zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])

        true_list = multiarray.zeros((len(test_p) + len(test_f), 1))
        for ii in range(len(test_p)):
            Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            true_list[ii, 0] = 1

        train_mask = np.ones(shape=Xn.shape)
        for ii in range(len(test_p)):
            train_mask[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            train_mask[Index_zeroRow[test_f[ii]], Index_zeroCol[test_f[ii]]] = 0

        train_mask_tensor = torch.from_numpy(train_mask).to(torch.bool)
        label = true_list

        A = copy.deepcopy(Xn)
        AT = A.T

        # 这一步是主成分分析法
        PCAassociation = run_MC_2(A)

        PCAassociation = torch.from_numpy(PCAassociation)
        PCAassociationT = PCAassociation.T

        rr = MD.shape[0]
        cc = MD.shape[1]
        mir_feat = torch.eye(rr)
        dis_feat = torch.eye(cc)
        parameters = [cc, rr]

        model = Model()
        optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        A = torch.from_numpy(A)
        AT = torch.from_numpy(AT)
        XX = copy.deepcopy(Xn)
        XX = torch.from_numpy(XX)

        XXN = A

        # 计算正样本权重，保持原始的正负样本比例
        pos_weight = float(XXN.shape[0] * XXN.shape[1] - XXN.sum()) / XXN.sum()

        mir_feat, dis_feat = Variable(mir_feat), Variable(dis_feat)
        if args.cuda:
            model.cuda()
            XX = XX.cuda()
            A = A.cuda()
            AT = AT.cuda()
            m_fusion_sim = m_fusion_sim.cuda()
            d_fusion_sim = d_fusion_sim.cuda()
            PCAassociation = PCAassociation.cuda()
            PCAassociationT = PCAassociationT.cuda()
            mir_feat = mir_feat.cuda()
            dis_feat = dis_feat.cuda()
            train_mask_tensor = train_mask_tensor.cuda()

        for epoch in range(args.epochs):
            auc1, aupr1, recall1, precision1, f11, mcc1, accuracy1, all_fpr, all_tpr, all_auc, fpr, tpr = train(epoch)

            AUC = AUC + auc1
            aupr = aupr + aupr1
            rec = rec + recall1
            pre = pre + precision1
            f1 = f1 + f11
            mcc = mcc + mcc1
            accuracy = accuracy + accuracy1

        print(auc)
        if f == args.cv_num:
            print('AUC: {:.4f}'.format(AUC / args.cv_num),
                  'aupr: {:.4f}'.format(aupr / args.cv_num),
                  'precision: {:.4f}'.format(pre / args.cv_num),
                  'recall: {:.4f}'.format(rec / args.cv_num),
                  'f1_score: {:.4f}'.format(f1 / args.cv_num),
                  'mcc_score: {:.4f}'.format(mcc / args.cv_num),
                  'accuracy_score: {:.4f}'.format(accuracy / args.cv_num),
                  )

            a = AUC / args.cv_num
            b = aupr / args.cv_num
            c = pre / args.cv_num
            d = rec / args.cv_num
            e = f1 / args.cv_num
            f = mcc / args.cv_num
            g = accuracy / args.cv_num

    auc_sum = auc_sum + a
    aupr_sum = aupr_sum + b
    precision_sum = precision_sum + c
    recall_sum = recall_sum + d
    f1_sum = f1_sum + e
    mcc_sum = mcc_sum + f
    accuracy_sum = accuracy_sum + g

# 绘制ROC曲线
plt.figure(figsize=(8, 6))

for i in range(len(all_fpr)):
    plt.plot(all_fpr[i], all_tpr[i], label=f'ROC fold {i + 1} (AUC = {all_auc[i]:.4f})', linestyle='-',
             linewidth=2)

min_length = min(len(fpr) for fpr in all_fpr)

all_fpr_fixed = [np.interp(np.linspace(0, 1, min_length), fpr, fpr) for fpr in all_fpr]
all_tpr_fixed = [np.interp(np.linspace(0, 1, min_length), tpr, tpr) for tpr in all_tpr]

mean_fpr = np.mean(all_fpr_fixed, axis=0)
mean_tpr = np.mean(all_tpr_fixed, axis=0)
mean_auc = np.mean(all_auc)

plt.plot(fpr, tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})', linestyle='-')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for 10-Fold CV')
plt.legend(loc='lower right')
plt.savefig('roc_10fold-514,62.png')
plt.show()

print(
    'auc_ave: {:.5f}'.format(auc_sum / n),
    'aupr_ave: {:.5f}'.format(aupr_sum / n),
    'precision_ave: {:.5f}'.format(precision_sum / n),
    'recall_ave: {:.5f}'.format(recall_sum / n),
    'f1_ave: {:.5f}'.format(f1_sum / n),
    'mcc_ave: {:.5f}'.format(mcc_sum / n),
    'accuracy_ave: {:.5f}'.format(accuracy_sum / n),
)
