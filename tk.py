import numpy as np
from scipy.special import psi, digamma
import warnings
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans


def tkmeans(data_set, k, nu_fixed=None, max_iter=500, tol=1e-3, patience=10):
    
    # ------------------------------------------------------------------
    # 数据预处理：确保 data_set 是 (n_samples, n_features)
    # ------------------------------------------------------------------
    X = np.array(data_set, dtype=np.float64)
    if X.shape[0] < X.shape[1]:
        X = X.T  # 转置为 (n_samples, n_features)
    n, p = X.shape                     # n = 样本数, p = 维度
    g = k                              # 簇数量
    
    # ------------------------------------------------------------------
    # 初始化参数
    # ------------------------------------------------------------------
    np.random.seed(42)  # 可复现

    # 使用 k-means++ 初始化，提高稳定性
    try:
        kinit = KMeans(n_clusters=g, n_init=10, random_state=42)
        mu = kinit.fit(X).cluster_centers_
    except Exception:
        # 回退到均匀采样
        mu = np.zeros((g, p))
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        for i in range(g):
            mu[i] = np.random.uniform(xmin, xmax)
    
    nu = 15
    alpha = 0.05
    
    if nu_fixed is not None:
        nu = float(nu_fixed)
    
    c = (p + nu) / 2.0
    
    # 用于存储中间变量 (g, n)
    tau  = np.zeros((g, n))   # 责任度 (变分后验概率)
    rho  = np.zeros((g, n))
    D    = np.zeros((g, n))   # Mahalanobis 距离平方 (未除 alpha)
    
    Q_old = -np.inf
    count_stable = 0
    dQ_prev = np.inf
    
    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    for itr in range(1, max_iter + 1):
        # ====================== E-step ======================
        # 计算距离 D(i,j) = ||x_j - mu_i||^2
        for i in range(g):
            diff = X - mu[i]                     # (n, p)
            D[i] = np.sum(diff**2, axis=1)       # (n,)
        
        # temp(i,j) = [1 + D/(nu*alpha)]^(-c)  (log-safe)
        log_temp = -c * np.log1p(D / (nu * alpha))
        log_temp = np.clip(log_temp, -700, 50)  # 避免下溢/溢出
        temp = np.exp(log_temp)
        
        # 归一化得到责任 tau
        sum_temp = temp.sum(axis=0, keepdims=True) + 1e-12
        tau = temp / sum_temp
        
        # rho 和 log rho（用于后续计算）
        rho = (nu + p) / (nu + D / alpha)
        # lrho = log(rho) - r   （代码里虽然算了但没用到，这里保留以防后续需要）
        
        # ====================== 计算目标函数 Q ======================
        Q = np.sum(tau * np.log(temp + 1e-12))   # 防止 log(0)
        
        # ====================== M-step ======================
        # 更新 mu
        for i in range(g):
            weights = tau[i] / rho[i]            # (n,)
            mu[i] = (weights @ X) / (weights.sum() + 1e-12)
        
        # 更新 alpha
        frac_a0 = np.sum(tau * rho * D)
        frac_a1 = np.sum(tau)
        alpha = frac_a0 / (frac_a1 * p + 1e-12)
        alpha = min(max(alpha, 1e-4), 1e6)  # 防止 alpha 过小或过大
        
        # 更新自由度 nu（仅当未固定时）
        if nu_fixed is None:
            # 计算 star（原代码中的平均 digamma 差）
            star = 0.0
            for i in range(g):
                diff = X - mu[i]
                dist = np.sum(diff**2, axis=1) / alpha
                term = np.log((nu + p) / (nu + dist)) - (nu + p) / (nu + dist)
                star += np.sum(tau[i] * term) / (tau[i].sum() + 1e-12)
            star /= g
            
            # 固定点迭代求解 nu（更稳定方式）
            nu_old = nu
            for _ in range(50):  # 最多迭代50次
                deriv = -digamma(nu) + np.log(nu) + 1 + star - np.log((nu_old + p)/2) + digamma((nu_old + p)/2)
                if abs(deriv) < 1e-6:
                    break
                nu = nu - deriv * 0.3   # 阻尼步长更小
                nu = float(np.clip(nu, 2.01, 100.0))      # 限制范围
        
        # ====================== 打印信息 ======================
        # print(f"itr: {itr:3d}, Q: {Q:.6f}, nu: {nu:.4f}, alpha: {alpha:.8f}")
        
        # ====================== 收敛判断 ======================
        if np.isnan(Q):
            # print("Q became NaN, stopping.")
            break
            
        dQ = abs(Q - Q_old)
        if dQ < tol or abs(dQ - dQ_prev) < tol:
            count_stable += 1
        else:
            count_stable = 0
            
        if count_stable >= patience:
            # print(f"Converged after {itr} iterations.")
            break
            
        Q_old = Q
        dQ_prev = dQ
    
    # ====================== 最终硬分配 ======================
    idx = np.argmax(tau, axis=0) + 1      # 标签从 1 开始
    mu_final = mu                       # (k, p)
    
    return idx, mu_final, itr

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 加载 Iris 数据集
    iris = datasets.load_iris()
    X = iris.data      # (150, 4)
    y_true = iris.target + 1   # 标签从 1 开始，方便对比

    # 数据标准化（对这类算法很重要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # print("Iris 数据集形状:", X_scaled.shape)

    # 注意：这里我们不固定 nu，让它自己学习（对重尾数据更鲁棒）
    labels_sa, centers_sa, iters_sa = tkmeans(X, k=3, nu_fixed=None)


    # 可视化
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=labels_sa, s=10, cmap='viridis')
    plt.scatter(centers_sa[:,0], centers_sa[:,1], c='red', s=200, marker='X')
    plt.title(f'Sigma-Alpha Clustering (iter={iters_sa})')
    plt.show()
    
    # print(f"Sigma-Alpha Clustering 运行完成，共迭代 {iters_sa} 次")
 
