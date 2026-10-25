import os
import time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler


def run_fold_leakfree_rfe(X_f_raw, y, X_p, train_idx, test_idx, fold=0,
                          n_features=500, step=100, cache_dir="./fold_leakfree_large8_combat"):
    """
    严格折内无泄漏 RF-RFE (7381 -> 500) 与标准化
    提取自 run7_large8_with_combat_3.py
    """
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, f"X_train_{n_features}_fold{fold}.npy")
    test_cache = os.path.join(cache_dir, f"X_test_{n_features}_fold{fold}.npy")

    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print("⚡ 读取折内无泄露特征缓存...")
        X_train_500 = np.load(train_cache)
        X_test_500 = np.load(test_cache)
    else:
        print(f"🔄 正在执行折内严格无泄露 RFE (7381 -> {n_features})...")
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=rf, n_features_to_select=n_features, step=step)

        X_train_500 = rfe.fit_transform(X_f_raw[train_idx], y[train_idx])
        X_test_500 = rfe.transform(X_f_raw[test_idx])

        np.save(train_cache, X_train_500)
        np.save(test_cache, X_test_500)
        print(f"✅ 折内 RFE 完成 (耗时: {time.time() - t0:.1f}s)")

    X_all = np.zeros((len(y), n_features), dtype=np.float32)
    X_all[train_idx] = X_train_500
    X_all[test_idx] = X_test_500

    sc_f, sc_p = StandardScaler(), StandardScaler()
    sc_f.fit(X_all[train_idx])
    sc_p.fit(X_p[train_idx])

    xf_t = torch.FloatTensor(sc_f.transform(X_all))
    xp_t = torch.FloatTensor(sc_p.transform(X_p))
    y_train_t = torch.FloatTensor(y[train_idx]).view(-1, 1)

    return xf_t, xp_t, y_train_t