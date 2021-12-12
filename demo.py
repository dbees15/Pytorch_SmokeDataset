#======Description=====
#Author: Daniel Beeston
#Purpose: Output demo for SmokeVideoDataLoader. Prints tensor data from __getitem__ and __getbatch__

import torch
import dataset

ds = dataset.SmokeVideoDataLoader('DemoDataset','labels.csv','Videofiles',100,15,10,2)

print("Example using dataset of 5 video files\n")

print(f"__getitem__ Example Output")
print("Item at index 0:")
item = ds.__getitem__(0)
print(f"Tensor size: {item[0].size()}")
print(f"Label: {item[1]}")

print("\n__getbatch__ Example Output")
for x in range(0,5):
    batch = ds.__getbatch__()
    if torch.is_tensor(batch[0]):
        print("Batch #"+str(ds.current_batch))
        print("Batch Index: "+str(ds.batch_index-ds.batch_size))
        print("Batch Labels: "+str(batch[1]))
        print(f"Batch Size: {batch[0].size()}")
        print("----------------------------")
    else:
        print(batch[0])
        print("----------------------------")
