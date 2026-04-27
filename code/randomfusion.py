import numpy as np

def two_tier_bi_random_walk(D_S_list, M_S_list, beta1, beta2, max_iter1=10, max_iter2=10):
    """
    两层双随机游走算法

    参数:
    - D_S_list: 包含疾病相似性矩阵的列表 [D_S1, D_S2, D_S3]
    - M_S_list: 包含微生物相似性矩阵的列表 [M_S1, M_S2, M_S3]
    - beta1: 第一层双随机游走的衰减因子
    - beta2: 第二层双随机游走的衰减因子
    - max_iter1: 第一层双随机游走的最大迭代次数
    - max_iter2: 第二层双随机游走的最大迭代次数

    返回:
    - R_D_final: 整合后的疾病相似性矩阵
    - R_M_final: 整合后的微生物相似性矩阵
    """
    def normalize_similarity_matrix(S):
        """归一化相似性矩阵"""
        row_sum = np.array(S.sum(axis=1)).flatten()
        # row_sum[row_sum == 0] = 1  # 避免除以0
        return S / row_sum[:, np.newaxis]

    def bi_random_walk(S1, S2, beta, max_iter):
        """双随机游走"""
        R = S1
        for _ in range(max_iter):
            R = (1 - beta) * S1 @ R + beta * S2
        return R

    # 第一层双随机游走
    D_S1, D_S2 = D_S_list[0], D_S_list[1]
    M_S1, M_S2 = M_S_list[0], M_S_list[1]

    # 归一化相似性矩阵
    D_S1_norm = normalize_similarity_matrix(D_S1)
    D_S2_norm = normalize_similarity_matrix(D_S2)
    M_S1_norm = normalize_similarity_matrix(M_S1)
    M_S2_norm = normalize_similarity_matrix(M_S2)

    # 第二层双随机游走
    D_S3 = D_S_list[2]
    M_S3 = M_S_list[2]

    # 归一化GIP核相似性矩阵
    D_S3_norm = normalize_similarity_matrix(D_S3)
    M_S3_norm = normalize_similarity_matrix(M_S3)



    # 第一层双随机游走
    R_D = bi_random_walk(D_S1_norm, D_S2_norm, beta1, max_iter1)
    R_M = bi_random_walk(M_S1_norm, M_S2_norm, beta1, max_iter1)



    # 第二层双随机游走
    R_D_final = bi_random_walk(R_D, D_S3_norm, beta2, max_iter2)
    R_M_final = bi_random_walk(R_M, M_S3_norm, beta2, max_iter2)

    return R_D_final, R_M_final