from six.moves import cPickle as pickle
from PIL import Image
import numpy as np

# Good Source: https://github.com/NVIDIA/DIGITS/blob/master/tools/download_data/cifar10.py

# Load Dataset
fo = open("test_batch", 'rb')
dict = pickle.load(fo)
fo.close()
data = dict["data"]
grayData = np.empty((10000, 1024), dtype=np.uint8)


# Split data into pictures, channels, pixel coordinates
data = data.reshape((10000, 3, 32, 32))
# Change to convenient order
data = data.transpose((0, 2, 3, 1))


for i in range(len(data)):
    im = Image.fromarray(data[i]).convert("L")
    tmp = np.array(im).reshape((1024))
    grayData[i] = tmp

dict["data"] = grayData

fd = open("test_batch_gray", 'wb')
pickle.dump(dict, fd, protocol=pickle.HIGHEST_PROTOCOL)