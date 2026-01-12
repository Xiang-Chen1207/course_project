
import h5py

def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name} | shape={obj.shape} | dtype={obj.dtype}")
    else:
        print(f"[GROUP]   {name}")

with h5py.File("/mnt/dataset2/hdf5_datasets/Workload_MATB/sub_10.h5", "r") as f:
    f.visititems(show)
