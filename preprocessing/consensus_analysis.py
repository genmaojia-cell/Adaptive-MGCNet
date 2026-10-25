import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# 路径配置（只读现有数据与缓存，完全不改动原始文件）
# ==============================================================================
FMRP_PATH = "X_fmri_7381_large_sites.npy"
PHENO_PATH = "pheno_large_sites.csv"
CACHE_DIR = "./fold_leakfree_large8"

# 全新保存的目标文件名（不覆盖原有任何 .npy 文件）
NEW_OUTPUT_NAME = "rfe_consensus_frequency_7381_combat_new.npy"

def main():
    print("=" * 80)
    print("🔍 启动独立提取程序：生成全新 ComBat 版 10 折 RFE 特征共识频次矩阵")
    print("=" * 80)

    if not os.path.exists(FMRP_PATH):
        raise FileNotFoundError(f"❌ 未找到大中心特征文件: {FMRP_PATH}")
    if not os.path.exists(PHENO_PATH):
        raise FileNotFoundError(f"❌ 未找到大中心表型文件: {PHENO_PATH}")

    # 1. 载入经 ComBat 校正的 493 例 x 7381 维原始特征
    X_f_raw = np.load(FMRP_PATH)
    pheno_df = pd.read_csv(PHENO_PATH)
    y = np.where(pheno_df['DX_GROUP'].values == 1, 1, 0).astype(np.float32)

    n_samples, n_features = X_f_raw.shape
    print(f"✅ 载入基准队列: {n_samples} 例受试者, 全脑特征维度: {n_features}")

    # 2. 完全复现 run6 的 10 折切分，确保每折的样本索引严格一致
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    consensus_freq = np.zeros(n_features, dtype=int)
    matched_folds = 0

    for fold, (train_idx, _) in enumerate(skf.split(X_f_raw, y)):
        train_cache = os.path.join(CACHE_DIR, f"X_train_500_fold{fold}.npy")
        if not os.path.exists(train_cache):
            print(f"⚠️ 未找到缓存文件: {train_cache}，请确认 ./fold_leakfree_large8 目录状态")
            continue

        # 载入该折 RFE 筛选出来的 (N_train, 500) 特征矩阵
        X_train_500 = np.load(train_cache)

        # 取当前训练折的首个样本向量进行精准列匹配
        probe_sample_idx = train_idx[0]
        raw_probe_vec = X_f_raw[probe_sample_idx]  # 7381 维原始值
        selected_probe_vec = X_train_500[0]       # 500 维选中值

        # 在 7381 维中检索出这 500 个特征所对应的真实列索引
        fold_selected_indices = []
        for val in selected_probe_vec:
            matches = np.where(np.isclose(raw_probe_vec, val, atol=1e-5))[0]
            if len(matches) > 0:
                fold_selected_indices.append(matches[0])

        if len(fold_selected_indices) == 500:
            consensus_freq[fold_selected_indices] += 1
            matched_folds += 1
            print(f"✅ Fold {fold + 1:02d} / 10: 成功精准回溯 500 个特征索引")
        else:
            print(f"⚠️ Fold {fold + 1:02d}: 匹配到 {len(fold_selected_indices)} 个特征（预期 500）")

    # 3. 校验并另存为新文件
    print("-" * 80)
    if matched_folds == 10:
        np.save(NEW_OUTPUT_NAME, consensus_freq)
        print(f"🎉 导出完成！新文件已保存至: {NEW_OUTPUT_NAME}")
        print(f"📊 统计概览:")
        print(f"   - 单条连接最高被选中频次: {np.max(consensus_freq)} / 10 折")
        print(f"   - 至少在 1 折中被选中的特征数: {np.sum(consensus_freq > 0)}")
        print(f"   - 跨 10 折全部稳定入选的核心特征数: {np.sum(consensus_freq == 10)}")
    else:
        print(f"❌ 只有 {matched_folds}/10 个折匹配成功，未执行最终保存，请检查缓存路径。")
    print("=" * 80)

if __name__ == "__main__":
    main()