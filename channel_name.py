import h5py
import numpy as np

h5_path = "/mnt/dataset2/hdf5_datasets/Workload_MATB/sub_1.h5"

with h5py.File(h5_path, "r") as f:
    chn_name_attr = f.attrs["chn_name"]

    # 统一转成 Python str
    chn_name = [
        x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in chn_name_attr
    ]

print("通道数:", len(chn_name))
print("通道名列表:")
print(chn_name)
