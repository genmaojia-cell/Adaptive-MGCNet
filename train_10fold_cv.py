import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

# 导入封装好的模块
from models import AdaptiveMGCNet
from preprocessing import run_fold_leakfree_rfe
from utils import compute_metrics, set_seed

# ==============================================================================
# 路径配置（接入 ComBat 特征，使用独立缓存）
# ==============================================================================
FMRP_PATH = "X_fmri_7381_large_sites_combat.npy"
PHENO_PATH = "pheno_large_sites.csv"
CACHE_DIR = "./fold_leakfree_large8_combat"
os.makedirs(CACHE_DIR, exist_ok=True)


def main():
    # 锁定随机种子保证完全可复现
    set_seed(42)

    print("=" * 80)
    print("🚀 启动【ComBat 校正版】8 站点队列（493 例，严格平衡）10 折基准评测")
    print("=" * 80)

    if not os.path.exists(FMRP_PATH):
        raise FileNotFoundError(f"未找到神经影像特征文件: {FMRP_PATH}")
    if not os.path.exists(PHENO_PATH):
        raise FileNotFoundError(f"未找到表型数据文件: {PHENO_PATH}")

    X_f_raw = np.load(FMRP_PATH)
    pheno_df = pd.read_csv(PHENO_PATH)
    y = np.where(pheno_df['DX_GROUP'].values == 1, 1, 0).astype(np.float32)

    # 剔除辅助和非数值列，保留 31 维临床表型特征
    drop_cols = ['SUB_ID', 'DX_GROUP', 'REAL_SITE']
    X_p = pheno_df.drop(columns=[c for c in drop_cols if c in pheno_df.columns]).values.astype(np.float32)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {'acc': [], 'pre': [], 'sen': [], 'spe': [], 'f1': [], 'auc': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_f_raw, y)):
        print(f"\n>>>>>> Fold {fold + 1:02d} / 10 <<<<<<")

        # 严格折内无泄漏 RFE (7381 -> 500) 与标准化
        xf_t, xp_t, y_train_t = run_fold_leakfree_rfe(
            X_f_raw=X_f_raw,
            y=y,
            X_p=X_p,
            train_idx=train_idx,
            test_idx=test_idx,
            fold=fold,
            n_features=500,
            step=100,
            cache_dir=CACHE_DIR
        )

        model = AdaptiveMGCNet(fmri_dim=500, pheno_dim=X_p.shape[1], hidden_dim=64, k_neighbors=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2e-3)
        criterion = nn.BCELoss()

        # 训练循环 (70 Epochs)
        for epoch in range(70):
            model.train()
            optimizer.zero_grad()
            out = model(xf_t, xp_t)
            loss = criterion(out[train_idx], y_train_t)
            loss.backward()
            optimizer.step()

        # 评估阶段
        model.eval()
        with torch.no_grad():
            out = model(xf_t, xp_t).cpu().numpy()
            probs = out[test_idx].flatten()
            y_test = y[test_idx]

        fold_res = compute_metrics(y_test, probs, threshold=0.5)
        for k in metrics:
            metrics[k].append(fold_res[k])

        print(
            f"Fold {fold + 1:02d} | ACC: {fold_res['acc'] * 100:.2f}% | "
            f"AUC: {fold_res['auc'] * 100:.2f}% | SEN: {fold_res['sen'] * 100:.2f}% | "
            f"SPE: {fold_res['spe'] * 100:.2f}%"
        )

    print("\n" + "=" * 75)
    print("🎯【高质量 8 站点 (ComBat 校正版) 最终 10 折性能】")
    print("=" * 75)
    for m in ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']:
        print(f"{m.upper()}: {np.mean(metrics[m]) * 100:.2f}% ± {np.std(metrics[m]) * 100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()