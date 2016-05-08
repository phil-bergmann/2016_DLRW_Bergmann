from __future__ import print_function

import numpy
import timeit
import os
import sys

from bhtsne import bh_tsne
from data import load_cifar
from PIL import Image
from tile_raster_images import scale_to_unit_interval


def run_bhtsne(data, out_size=8000, save_name='test.png', invert=False, theta=0.5, shape=(28, 28)):

    results = numpy.zeros((data.shape[0], 2))

    print("[*] Processing data ...")

    start_time = timeit.default_timer()

    for res, save in zip(bh_tsne(numpy.copy(data), theta=theta), results):
        save[...] = res

    end_time = timeit.default_timer()

    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fs' % (end_time - start_time)), file=sys.stderr)

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
        pic = scale_to_unit_interval(data[i].reshape(shape))
        if invert:
            pic = 1 - pic
        outarray[xpos:xpos + shape[0], ypos:ypos + shape[1]] = pic * 255

    image = Image.fromarray(outarray)

    image.save(save_name)


if __name__ == '__main__':
    cifar = load_cifar(concat=False)

    data = cifar[0][0].astype('float64')

    # cifar theta 0.5
    run_bhtsne(data, save_name='cifar_theta0.5.png', theta=0.5, shape=(32, 32))

    # cifar theta 0.75
    run_bhtsne(data, save_name='cifar_theta0.75.png', theta=0.75, shape=(32, 32))

    # cifar theta 1
    run_bhtsne(data, save_name='cifar_theta1.0.png', theta=1.0, shape=(32, 32))