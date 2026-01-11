import pandas as pd
import numpy as np

# 读两个文件
df1 = pd.read_csv("/mnt/dataset4/cx/code/EEG_LLM_text/output_sub2/trial0/segment1.csv")
df2 = pd.read_csv("/mnt/dataset4/cx/extracted_features/sub_2/sub2_trial0_segment1.csv")

# 保证按列名对齐
common_cols = sorted(set(df1.columns) & set(df2.columns))

diff_report = []

for col in common_cols:
    if col in ["trial_id", "segment_id", "session_id", "label", "start_time", "end_time"]:
        continue

    v1 = df1[col].values
    v2 = df2[col].values

    if len(v1) != len(v2):
        diff_report.append((col, "length mismatch", None))
        continue

    abs_diff = np.abs(v1 - v2)

    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    same = np.allclose(v1, v2, atol=1e-6)

    diff_report.append((
        col,
        "same" if same else "DIFFERENT",
        max_diff,
        mean_diff
    ))

# 输出差异列
diff_df = pd.DataFrame(
    diff_report,
    columns=["feature", "status", "max_abs_diff", "mean_abs_diff"]
)

print(diff_df.sort_values("max_abs_diff", ascending=False))
