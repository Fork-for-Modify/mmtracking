import torch
import numpy as np
import cv2
import os
from os.path import join as opj
import scipy.io as scio

# ---------------
# image & data save
# ---------------

## ==== save image from torch sensor ====
# import torch
# import numpy as np
# import cv2
tensor = torch.zeros(16, 3, 256, 256, device='cuda:0')  # a tensor for test
# tensor = xxx

# save N*3*H*W tensor to RGB image: save $tensor[0]
tmp_ = tensor[0].detach().cpu().numpy()[::-1].transpose(1, 2, 0)
tmp_ = 255*(tmp_ - tmp_.min())/(tmp_.max() - tmp_.min())
cv2.imwrite('tensor_0.png', tmp_.astype(np.uint8))

# save N*C*H*W tensor to RGB image: save $tensor[0,0]
tmp_ = tensor[0, 8].detach().cpu().numpy()
tmp_ = 255*(tmp_ - tmp_.min())/(tmp_.max() - tmp_.min())
cv2.imwrite('tensor_0.png', tmp_.astype(np.uint8))

## ==== save tensor data to .mat ====
# import scipy.io as scio
# import numpy as np

data1 = sci_mask[0]
data1 = data1.detach().cpu().numpy()
data1 = np.transpose(data1, (2, 3, 1, 0))

# data = data.detach().cpu().numpy()
# data = np.transpose(data, (2, 3, 1, 0))

save_path = './test_sci_data.mat'
scio.savemat(save_path, {'mask': data1, 'meas': data2, 'orig': data3})
