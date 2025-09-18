import numpy as np
import copy
r=62
# 交替最小二乘更新函数
def DC(D,mu,T0):
    U,S,V = np.linalg.svd(D)
    print(V.shape[0])
    print(V.shape[1])
    T1 = np.zeros(np.size(T0))
    for i in range(1,500):
        T1 = DCInner(S,mu,T0)
        err = np.sum(np.square(T1-T0))
        if err < 1e-6:
            break
        T0 = T1


    #求行块结果
    # V = V[:82, :]
    # print(U.shape[0])
    # print(U.shape[1])
    #求列块结果
    U = U[:, :r]
    l_1 = np.dot(U, np.diag(T1))
    l = np.dot(l_1, V)
    return l,T1


def DCInner(S,mu,T_k):
    lamb = 1/mu
    grad = 1/(1+np.square(T_k))
    T_k1 = S-lamb*grad
    T_k1[T_k1<0]=0
    return T_k1

#上面是求奇异值的，下面是求L2，1范数的

def GAMA(H):
    muzero =2
    mu = muzero
    rho = 1
    tol = 1e-3
    alpha =0.01

    m, n = np.shape(H)
    L = copy.deepcopy(H)
    S = np.zeros((m,n))
    Y = np.zeros((m,n))  #这个保存，正常更新

    omega = np.zeros(H.shape)
    omega[H.nonzero()] = 1

    for i in range(0, 500):


        #这些代码是求W的
        tran = (1/mu) * (Y+alpha*(H*omega))+L
        W = tran - (alpha/(alpha+mu))*omega*tran
        # W[W < 0] = 0
        # W[W > 1] = 1

        #这三项整体算是求奇异值的,也就是X,在这里L就相当于X了
        D = W-Y/mu  #更新C
        sig = np.zeros(min(m, n)) #存奇异值的
        L, sig = DC(copy.deepcopy(D),mu,copy.deepcopy(sig)) #求奇异值的

        #求Y
        Y= Y+mu*(L-W)     #更新Y
        mu = mu*rho         #更新u
        sigma = np.linalg.norm(L-W,'fro')
        RRE = sigma/np.linalg.norm(H,'fro')
        if RRE < tol:
            break
    # M_1 = L[0:SM.shape[0], SM.shape[0]:H.shape[1]]
    # print(M_1)
    return W




def run_MC_2(Y):
    #SVD
    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    #s1=np.diag(np.sqrt(S))
    U= U[:, 0:r]
    V= V[0:r,:]
    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    C=Wt
    A0 = GAMA(A)
    B0 = GAMA(B)
    lty = A0 @ B0.T

    # lty = A @ B.T
    return lty


