import numpy as np
import pandas as pd

FMRP_7381_PATH = "X_fmri_7381.npy"
PHENO_CSV_PATH = "processed_data_advanced.csv"

# 1. 加载数据
pheno_df = pd.read_csv(PHENO_CSV_PATH)
X_f = np.load(FMRP_7381_PATH)

# 2. 从 One-Hot 列还原出每个被试的真实站点名称
site_cols = [c for c in pheno_df.columns if c.startswith('SITE_ID_')]
print(f'检测到 {len(site_cols)} 个站点 One-Hot 列: {site_cols}\n')


def get_site_name(row):
  for col in site_cols:
    if row[col] == True or str(row[col]).upper() == 'TRUE' or row[col] == 1:
      return col.replace('SITE_ID_', '')
  return 'UNKNOWN'


pheno_df['REAL_SITE'] = pheno_df.apply(get_site_name, axis=1)

# 3. 统计真实的 17 站点分布
print('=' * 75)
print('📊 真实 17 站点分布与样本量统计：')
print('=' * 75)
site_counts = pheno_df['REAL_SITE'].value_counts()
for site, count in site_counts.items():
  dx_counts = pheno_df[pheno_df['REAL_SITE'] == site][
      'DX_GROUP'
  ].value_counts()
  asd = dx_counts.get(1, 0)
  tc = dx_counts.get(2, dx_counts.get(0, 0))
  print(f'  - 站点 {site:<16}: 总被试={count:3d} (ASD: {asd:2d}, TC: {tc:2d})')

# 4. 设定高质量大中心纳入标准（门槛: 样本量 >= 30）
THRESHOLD = 30
valid_sites = site_counts[site_counts >= THRESHOLD].index.tolist()
filter_mask = pheno_df['REAL_SITE'].isin(valid_sites).values

filtered_pheno = pheno_df[filter_mask].reset_index(drop=True)
filtered_X_f = X_f[filter_mask]

print('\n' + '=' * 75)
print(f'🎯 筛选后保留的高质量大中心 (样本量 >= {THRESHOLD})：')
print('=' * 75)
print(f'入选站点: {valid_sites} (共 {len(valid_sites)} 个高质量中心)')
print(
    f'队列总样本量: {len(filtered_pheno)} 例 (剔除了'
    f' {len(pheno_df) - len(filtered_pheno)} 例长尾微型样本)'
)
print(f'类别平衡性: {dict(filtered_pheno["DX_GROUP"].value_counts())}')

# 5. 保存干净的多中心数据集
np.save('X_fmri_7381_large_sites.npy', filtered_X_f)
filtered_pheno.to_csv('pheno_large_sites.csv', index=False)
print('\n✅ 已保存高质量基准文件:')
print('   - X_fmri_7381_large_sites.npy')
print('   - pheno_large_sites.csv')