from __future__ import print_function

import numpy
import matplotlib.pyplot as plt

from bhtsne import bh_tsne
from data import load_data
from PIL import Image
from tile_raster_images import scale_to_unit_interval

def run_bhtsne(data, out_size=6000, save_name='test.png', invert=True, theta=0.5):

    data = test_set_x[:3000]

    f, ax = plt.subplots()

    results = numpy.zeros((data.shape[0], 2))

    for res, save in zip(bh_tsne(numpy.copy(data), theta=theta), results):
        save[...] = res

    # normalize results
    res_min = numpy.min(results, axis=0)
    results = results - res_min
    res_max = numpy.max(results, axis=0)
    results = results / res_max

    outarray = numpy.zeros((out_size, out_size), dtype='uint8')
    outarray[...] = 255

    for i in xrange(results.shape[0]):
        xpos = int(results[i][0] * (out_size - 1000) + 500)
        ypos = int(results[i][1] * (out_size - 1000) + 500)
        pic = scale_to_unit_interval(data[i].reshape(28, 28))
        if invert:
            pic = 1 - pic
        outarray[xpos:xpos + 28, ypos:ypos + 28] = pic * 255

    image = Image.fromarray(outarray)

    image.save(save_name)


if __name__ == '__main__':
    pass