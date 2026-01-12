import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('/mnt/dataset2/Datasets/TUEV/v3.0.0/DOCS/metadata_v00r.xlsx')

# 保存为 CSV 文件
df.to_csv('/mnt/dataset4/cx/code/EEG_LLM_text/metadata_v00r.csv', index=False)
