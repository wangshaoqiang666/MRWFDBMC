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
from hypergraph_utils import _generate_G_from_H
import os
from kl_loss import kl_loss
from function import create_resultlist
from utils import f1_score_binary,precision_binary,recall_binary, mcc_binary, accuracy_binary
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
torch.backends.cudnn.benchmark = False  #禁用卷积算法选择，确保结果可重复
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

def contrastive_loss(h1, h2, tau = 0.4):
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
            m_fusion_sim, d_fusion_sim, mir_feat, dis_feat, PCAassociationT, PCAassociation)  # 将数据传入模型

        outputs = recover.t().cpu().detach().numpy()
        test_predict = create_resultlist(outputs, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,
                                         Index_zeroCol, len(test_p), zero_length, test_f)

        #掩码矩阵
        MA = torch.masked_select(A, train_mask_tensor)
        #总损失
        rec = torch.masked_select(recover.t(), train_mask_tensor)

        #超图的重构
        reH = torch.masked_select(reconstructionG.t(), train_mask_tensor)

        #超图损失
        lossH = F.binary_cross_entropy_with_logits(reH.t(), MA,pos_weight=pos_weight)
        #

        loss = lossH + F.binary_cross_entropy_with_logits(rec.t(), MA,pos_weight=pos_weight)
        # loss = F.binary_cross_entropy_with_logits(rec.t(), MA,pos_weight=pos_weight)
        # loss = lossH

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
        max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),torch.from_numpy(test_predict).float())
        print("//////////max_f1_score",max_f1_score)
        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////precision:", precision)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////recall:", recall)

        mcc = mcc_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////mcc:", mcc)
        accuracy = accuracy_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////accuracy:", accuracy)

        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    fpr, tpr = [], []
    if epoch == args.epochs - 1:
        auc1 = auc_val
        aupr1 = aupr_val
        recall1 = recall
        precision1 = precision
        f11 = max_f1_score
        mcc1= mcc
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
        # 绘制AUPR曲线
        # precision, recall, thresholds = precision_recall_curve(label, test_predict)
        #
        # # Interpolate the recall-precision values for consistent comparison across folds
        # interp_recall = np.linspace(0, 1, 100)  # Uniformly spaced recall values
        # interp_precision = np.interp(interp_recall, recall[::-1],
        #                             precision[::-1])  # Reverse recall and precision for interpolation
        # interp_precision[0] = 1.0  # Ensure the starting precision is 1.0 for recall = 0
        #
        # # Store interpolated values
        # all_precision.append(interp_precision)
        # all_recall.append(interp_recall)
        #
        # # Calculate AUPR for this fold
        # aupr = auc(recall, precision)
        # all_aupr.append(aupr)
        #
        # for i in range(len(recall)):
        #     if recall[i] == 1:
        #         precision[i] = 0
        #
        # aupr = auc(recall, precision)
        # aupr_sum = aupr_sum + aupr
        # time += 1
        # s = aupr_sum / time
        # print(f'Fold {time}: AUPR = {aupr:.4f}, Cumulative AUPR = {s:.4f}')
    return auc1,aupr1,recall1,precision1,f11, mcc1, accuracy1,all_fpr,all_tpr,all_auc,fpr, tpr
    # return auc1, aupr1, recall1, precision1, f11, mcc1, accuracy1, all_recall, all_precision, all_aupr, recall, precision

    # return auc1,aupr1,recall1,precision1,f11, mcc1, accuracy1

#miRNA-disease
# MD = np.loadtxt("data/md_delete.txt")  #miRNA-disease 关联
# MM = np.loadtxt("data/mm_delete.txt")  #miRNA相似性
# DD = np.loadtxt("data/dd_delete.txt")  #疾病相似性
# DG = np.loadtxt("data/dg_delete.txt")  #基因-疾病关联
# MG = np.loadtxt("data/mg_delete.txt")  #miRNA-基因关联

#circRNA-drug
# MD = np.loadtxt("circRNAdrug/association.txt")
# # MM = np.loadtxt("circRNAdrug/integration_circRNA.txt")
# # DD = np.loadtxt("circRNAdrug/integration_drug.txt")
# C1 = np.loadtxt("circRNAdrug/GKGIP_circRNA.txt")
# D1 = np.loadtxt("circRNAdrug/GKGIP_drug.txt")
#
# C2 = np.loadtxt("circRNAdrug/LKGIP_circRNA.txt")
# D2 = np.loadtxt("circRNAdrug/LKGIP_drug.txt")

#lnc-disease
# MD = np.loadtxt("lncRNAdisease/association.txt")
# # MM = np.loadtxt("circRNAdrug/integration_circRNA.txt")
# # DD = np.loadtxt("circRNAdrug/integration_drug.txt")
# C1 = np.loadtxt("lncRNAdisease/GKGIP_lncRNA.txt")
# D1 = np.loadtxt("lncRNAdisease/GKGIP_disease.txt")
#
# C2 = np.loadtxt("lncRNAdisease/LKGIP_circRNA.txt")
# D2 = np.loadtxt("lncRNAdisease/LKGIP_disease.txt")
#circ-disease
# MD = np.loadtxt("circRNAdisease(514,62)/association.txt")
# C1 = np.loadtxt("circRNAdisease(514,62)/GKGIP_circRNA.txt")
# D1 = np.loadtxt("circRNAdisease(514,62)/GKGIP_disease.txt")
#
# C2 = np.loadtxt("circRNAdisease(514,62)/LKGIP_circRNA.txt")
# D2 = np.loadtxt("circRNAdisease(514,62)/LKGIP_disease.txt")
#circ-disease
# MD = np.loadtxt("circRNAdisease(312,40)/association.txt")
# C1 = np.loadtxt("circRNAdisease(312,40)/GKGIP_circRNA.txt")
# D1 = np.loadtxt("circRNAdisease(312,40)/GKGIP_disease.txt")
#
# C2 = np.loadtxt("circRNAdisease(312,40)/LKGIP_circRNA.txt")
# D2 = np.loadtxt("circRNAdisease(312,40)/LKGIP_disease.txt")
# #circ-disease(533,89)
MD = np.loadtxt("circRNAdisease(533,89)/association.txt")
C1 = np.loadtxt("circRNAdisease(533,89)/GKGIP_circRNA.txt")
D1 = np.loadtxt("circRNAdisease(533,89)/GKGIP_disease.txt")

C2 = np.loadtxt("circRNAdisease(533,89)/LKGIP_circRNA.txt")
D2 = np.loadtxt("circRNAdisease(533,89)/LKGIP_disease.txt")

# #circ-disease(923,104)
# MD = np.loadtxt("circRNAdisease(923,104)/association.txt")
# C1 = np.loadtxt("circRNAdisease(923,104)/GKGIP_circRNA.txt")
# D1 = np.loadtxt("circRNAdisease(923,104)/GKGIP_disease.txt")
#
# C2 = np.loadtxt("circRNAdisease(923,104)/LKGIP_circRNA.txt")
# D2 = np.loadtxt("circRNAdisease(923,104)/LKGIP_disease.txt")

#drug-disease
# MD = np.loadtxt("drugdisease/dataset2(269,598)/association.txt")
#
# C1 = np.loadtxt("drugdisease/dataset2(269,598)/GKGIP_drug.txt")
# D1 = np.loadtxt("drugdisease/dataset2(269,598)/GKGIP_disease.txt")
#
# C2 = np.loadtxt("drugdisease/dataset2(269,598)/LKGIP_drug.txt")
# D2 = np.loadtxt("drugdisease/dataset2(269,598)/LKGIP_disease.txt")

#------------------------------------------------------------------------
#以下是超图相似性随机游走融合模块
# #高斯核相似性
HHMMG = construct_H_with_KNN(C1)
HMM1 = generate_G_from_H(HHMMG)  # circRNA相似性
HMM1 = HMM1.double()

HHDDG = construct_H_with_KNN(D1)  # 药物相似性
HDD1 = generate_G_from_H(HHDDG)
HDD1 = HDD1.double()

# #拉普拉斯相似性
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
max_iter1, max_iter2 = 1,4

# 调用两层双随机游走函数
m_fusion_sim, d_fusion_sim = randomfusion.two_tier_bi_random_walk(D_S_list, M_S_list, beta1, beta2, max_iter1,
                                                                  max_iter2)

# m_fusion_sim = (HMM1 + HMM2 + HMG)/3
# d_fusion_sim = (HDD1+HDD2+HDG)/3
#----------------------------------------------------------------------------------------------------------------------


[row, col] = np.shape(MD)
indexn = np.argwhere(MD == 0)
Index_zeroRow = indexn[:, 0]
Index_zeroCol = indexn[:, 1]
indexp = np.argwhere(MD == 1)
Index_PositiveRow = indexp[:, 0]
Index_PositiveCol = indexp[:, 1]
totalassociation = np.size(Index_PositiveRow) #7694
fold = int(totalassociation / 5) #1538

zero_length = np.size(Index_zeroRow)#321601

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

#绘制ROC曲线
tprs=[]
aucs=[]
all_fpr, all_tpr, all_auc = [], [], []
mean_fpr=np.linspace(0,1,100)
#绘制PR曲线
all_precision, all_recall, all_aupr = [], [], []
aupr_sum, time = 0, 0

for time in range(1,n+1):
    Auc_per = []
    f1_score_per = []
    precision_per = []
    recall_per = []
    aupr_per = []
    p = np.random.permutation(totalassociation)
    # print(p)

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


        # X_1 = copy.deepcopy(M_1)  # 深拷贝 M_1
        # Xn_1 = copy.deepcopy(X_1)  # 初始化 Xn_1
        # # 局部
        # X_2 = copy.deepcopy(M_2)  # 深拷贝 M_2
        # Xn_2 = copy.deepcopy(X_2)  # 初始化 Xn_2


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
        PCAassociation =run_MC_2(A)
#---------------------------------------------------------------------------------------------------
        #以下是基于奇异值分解的双主成分分析模块
        #全局超图
        PCAassociation = torch.from_numpy(PCAassociation)
        PCAassociationT =  PCAassociation.T
        #
        # GSM = G_similarity.compute_global_similarity_matrix(MM)
        # # 计算disease的全局相似性
        # GMM = G_similarity.compute_global_similarity_matrix(DD)
        # # # #计算局部图推理
        # # # # 计算circRNA的局部相似性
        # LSM = L_similarity.row_normalization(MM, 10)
        # # # 计算disease的局部相似性
        # LMM = L_similarity.row_normalization(DD, 10)

        # H1 = np.hstack((MM, A))  # 将参数元组的元素数组按水平方向进行叠加
        #
        # H2 = np.vstack((DD, A))  # 将参数元组的元素数组按垂直方向进行叠加
        #
        # L1 = run_MC(H1)
        #
        # L2 = run_MC(H2)
        # #
        # M_1 = L1[0:MM.shape[0], MM.shape[0]: L1.shape[1]]  # 把补充的关联矩阵原来A位置给取出来。
        # # # #列块
        # M_2 = L2[DD.shape[0]:L2.shape[0], 0:DD.shape[0]]  # 把补充的关联矩阵原来A位置给取出来。
        #
        # A = (M_1 + M_2)/2
        # #lncRNA-disease聚合用的随机初始特征
        rr = MD.shape[0]
        cc = MD.shape[1]
        mir_feat = torch.eye(rr)
        dis_feat = torch.eye(cc)
        parameters = [cc, rr]

        # #超图
        # HHMD = construct_H_with_KNN(A)
        # HMD = generate_G_from_H(HHMD) #757*757 miRNA-疾病关联
        # HMD = HMD.double()
        # # 打印张量757,757
        # print(HMD)


        # HHDM = construct_H_with_KNN(AT)  #疾病-miRNA关联
        # HDM = generate_G_from_H(HHDM) #435*435
        # HDM = HDM.double()

#去掉两个相似性视图
        # HHMM = construct_H_with_KNN(MM)
        # HMM = generate_G_from_H(HHMM)   #miRNA相似性
        # HMM = HMM.double()

        # HHDD = construct_H_with_KNN(DD) #疾病相似性
        # HDD = generate_G_from_H(HHDD)
        # HDD = HDD.double()

        model = Model()
        optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)




        A = torch.from_numpy(A)
        AT = torch.from_numpy(AT)
        XX = copy.deepcopy(Xn)
        XX = torch.from_numpy(XX)

        #消融实验
        XXN = A


        pos_weight = float(XXN.shape[0] * XXN.shape[1] - XXN.sum()) / XXN.sum()
        # norm = A.shape[0] * A.shape[1] / float((A.shape[0] * A.shape[1] - A.sum()) * 2)
        mir_feat, dis_feat = Variable(mir_feat), Variable(dis_feat)
        # loss_kl = kl_loss(218, 271)

        #lncRNAdisease
        # loss_kl = kl_loss(157, 82)
        #drugdisease
        # loss_kl = kl_loss(514, 62)
        #circdisease
        loss_kl = kl_loss(82, 157)


        if args.cuda:
            model.cuda()

            XX = XX.cuda()

            A = A.cuda()
            AT = AT.cuda()

            m_fusion_sim = m_fusion_sim.cuda()
            d_fusion_sim = d_fusion_sim.cuda()
            PCAassociation = PCAassociation.cuda()
            PCAassociationT = PCAassociationT.cuda()

            # HMM = HMM.cuda()
            # HDD = HDD.cuda()

            mir_feat = mir_feat.cuda()
            dis_feat = dis_feat.cuda()

            train_mask_tensor = train_mask_tensor.cuda()

        for epoch in range(args.epochs):
            # print(epoch)
            auc1,aupr1,recall1,precision1,f11,mcc1, accuracy1,all_fpr,all_tpr,all_auc,fpr, tpr = train(epoch)
            # auc1, aupr1, recall1, precision1, f11, mcc1, accuracy1, all_recall, all_precision, all_aupr, recall, precision = train(epoch)

            # auc1,aupr1,recall1,precision1,f11,mcc1, accuracy1  = train(epoch)
            AUC = AUC + auc1
            aupr = aupr + aupr1
            rec = rec + recall1
            pre = pre + precision1
            f1 = f1 + f11
            mcc = mcc + mcc1
            accuracy = accuracy + accuracy1
        print(auc)
        if f == args.cv_num:
            print('AUC: {:.4f}'.format(AUC/args.cv_num),
                  'aupr: {:.4f}'.format(aupr/args.cv_num),
                  'precision: {:.4f}'.format(pre / args.cv_num),
                  'recall: {:.4f}'.format(rec / args.cv_num),
                  'f1_score: {:.4f}'.format(f1 / args.cv_num),
                  'mcc_score: {:.4f}'.format(mcc / args.cv_num),
                  'accuracy_score: {:.4f}'.format(accuracy / args.cv_num),
                      )

            a = AUC/args.cv_num
            b = aupr/args.cv_num
            c = pre / args.cv_num
            d = rec / args.cv_num
            e = f1 / args.cv_num
            f = mcc / args.cv_num
            g = accuracy / args.cv_num


    auc_sum = auc_sum + a
    aupr_sum = aupr_sum + b
    precision_sum= precision_sum +c
    recall_sum = recall_sum + d
    f1_sum = f1_sum + e
    mcc_sum = mcc_sum + f
    accuracy_sum = accuracy_sum + g

#绘制ROC曲线
plt.figure(figsize=(8, 6))

for i in range(len(all_fpr)):
    plt.plot(all_fpr[i], all_tpr[i], label=f'ROC fold {i + 1} (AUC = {all_auc[i]:.4f})', linestyle='-',
             linewidth=2)
# 先找到最小的长度
min_length = min(len(fpr) for fpr in all_fpr)

# 对每个子数组进行截断或插值，使它们具有相同的长度
all_fpr_fixed = [np.interp(np.linspace(0, 1, min_length), fpr, fpr) for fpr in all_fpr]
all_tpr_fixed = [np.interp(np.linspace(0, 1, min_length), tpr, tpr) for tpr in all_tpr]
# 然后计算平均值
mean_fpr = np.mean(all_fpr_fixed, axis=0)
mean_tpr = np.mean(all_tpr_fixed, axis=0)
mean_auc = np.mean(all_auc)
# np.savetxt(r'mean_fpr.txt', mean_fpr, delimiter='\t', fmt='%.9f')
# np.savetxt(r'mean_tpr.txt', mean_tpr, delimiter='\t', fmt='%.9f')

plt.plot(fpr, tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})', linestyle='-')
# Plot the diagonal chance line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for 10-Fold CV')
plt.legend(loc='lower right')
# 保存图像到文件
plt.savefig('roc_10fold-514,62.png')
plt.show()
# 绘制Precision-Recall曲线
# plt.figure(figsize=(8, 6))
#
# for i in range(len(all_precision)):
#     plt.plot(all_recall[i], all_precision[i], label=f'PR fold {i + 1} (AUPR = {all_aupr[i]:.4f})',
#              linestyle='-')
#
# # 先找到最小的长度
# min_length = min(len(recall) for recall in all_recall)
#
# # 对每个子数组进行截断或插值，使它们具有相同的长度
# all_recall_fixed = [np.interp(np.linspace(0, 1, min_length), recall, recall) for recall in all_recall]
# all_precision_fixed = [np.interp(np.linspace(0, 1, min_length), precision, precision) for precision in
#                        all_precision]
#
# # 然后计算平均值
# mean_recall = np.mean(all_recall_fixed, axis=0)
# mean_precision = np.mean(all_precision_fixed, axis=0)
# mean_aupr = np.mean(all_aupr)

# plt.plot(recall, precision, label=f'Mean PR (AUPR = {mean_aupr:.4f})', linestyle='-')
# plt.plot([0, 1], [1, 0], linestyle='--', color='gray')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('PR Curve for 5-Fold CV')
# plt.legend(loc='lower left')
# # 保存图像到文件
# plt.savefig('pr_curve_5fold.png')
# plt.show()

print(
      'auc_ave: {:.5f}'.format(auc_sum/n),
      'aupr_ave: {:.5f}'.format(aupr_sum/n),
      'precision_ave: {:.5f}'.format(precision_sum / n),
      'recall_ave: {:.5f}'.format(recall_sum / n),
      'f1_ave: {:.5f}'.format(f1_sum / n),
      'f1_ave: {:.5f}'.format(mcc_sum / n),
      'f1_ave: {:.5f}'.format(accuracy_sum / n),
                      )

















