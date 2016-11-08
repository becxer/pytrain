#
# Basic K-Nearest Neighbors
#
# @ author becxer
# @ reference Machine Learning in Action by Peter Harrington
# @ e-mail becxer87@gmail.com
#

import numpy as np
from pytrain.lib import convert
from pytrain.lib import ptmath
import operator

class KNN:
    def __init__(self, mat_data, label_data, k, dist_func):
        self.mat_data = convert.list2npfloat(mat_data)
        self.dist_func = ptmath.distfunc(dist_func)
        self.label_data = label_data
        self.train_size = self.mat_data.shape[0]
        self.k = k

    def fit(self):
        pass

    # compare distance from all mat_data rows and choose most closer one
    def predict(self, array_input):
        array_input = convert.list2npfloat(array_input)

        distances = []
        for trg in self.mat_data:
            distances.append(self.dist_func(array_input, trg))
        distances = np.array(distances)
        
        sorted_distances = distances.argsort()
        class_count = {}
        for i in range(self.k):
            kth_label = str(self.label_data[sorted_distances[i]])
            class_count[kth_label] = class_count.get(kth_label, 0) + 1
        sorted_class_count = sorted(class_count.iteritems()
            , key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

