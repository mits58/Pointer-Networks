import collections

import chainer
import numpy as np

np.random.seed(1124) # noa himesaka is so cute :)

class PlaneData(chainer.dataset.DatasetMixin):
    """
    A class for point data.
    Attributes
    ----------
    must implement len, get_example, converter
    """

    def __init__(self, dataset, device):
        # points -> x, perms -> t
        self.points, self.perms = PlaneData.load_data(dataset, device)

    def __len__(self):
        return len(self.points)

    def get_example(self, i):
        return self.points[i], self.perms[i]

    def converter(self, datalist, device):
        xp = chainer.get_device(device).xp
        batch_points = [data[0] for data in datalist]
        batch_perms = [data[1] for data in datalist]
        return batch_points, batch_perms

    def load_data(dataset, device):
        """
        Only for 10 points TSP...
        """

        if device == -1:
            xp = np
        else:
            xp = device.xp


        with open('./data/{}'.format(dataset), 'r') as f:
            data = f.readlines()

        points = []
        perms = []

        for d in data:
            pt, pm = [tmp.strip() for tmp in d.split("output")]
            pt = xp.array([float(val) for val in pt.split(" ")], dtype=xp.float32)
            pm = xp.array([int(num) for num in pm.split(" ")])

            points.append(pt)
            perms.append(pm)

        return points, perms



if __name__ == '__main__':
    PlaneData.load_data("hoge", -1)
