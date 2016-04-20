#
# Basic K-Nearest Neighbors
#
# @ author becxer
# @ reference Machine Learning in Action by Peter Harrington
# @ e-mail becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
import operator


class KNN:
    def __init__(self, mat_data, label_data, k):
        self.mat_data = convert.list2npfloat(mat_data)
        self.label_data = label_data
        self.train_size = self.mat_data.shape[0]
        self.k = k

    def fit(self):
        pass

    # compare distance from all mat_data rows and choose most closer one
    def predict(self, array_input):
        array_input = convert.list2npfloat(array_input)
        diff_mat = tile(array_input, (self.train_size,1)) - self.mat_data
        pow_diff_mat = diff_mat ** 2
        pow_distances = pow_diff_mat.sum(axis=1)
        distances = pow_distances ** 0.5
        sorted_distances = distances.argsort()
        class_count = {}
        for i in range(self.k):
            kth_label = self.label_data[sorted_distances[i]]
            class_count[kth_label] = class_count.get(kth_label, 0) + 1
        sorted_class_count = sorted(class_count.iteritems()
            , key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

