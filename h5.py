import json

import h5py

data_path = "/home/chenkangxin/data/compass/modelnet40_ply_hdf5_2048/ply_data_test0.h5"
f = h5py.File(data_path, 'r')

print(f['data'])

f2 = open("/home/chenkangxin/data/compass/modelnet40_ply_hdf5_2048/ply_data_test_0_id2file.json")

data = json.load(f2)


print(len(data))