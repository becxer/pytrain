#
# Gaussian Naive Bayes
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
import operator

class GaussianNaiveBayes:

    def __init__(self, mat_data, label_data):
        self.mat_data = convert.list2npfloat(mat_data)
        self.label_data = label_data

        self.mat_mean = {}
        self.mat_variance = {}
        self.label_count = {}
        self.label_count_sum = 0

    # To calculate Gaussian, we have to get mean & variance for each Label
    def fit(self):
        self.col_size = len(self.mat_data[0])
        for i, label in enumerate(self.label_data):
            self.mat_mean[label] = self.mat_mean.get(label,\
                    zeros(self.col_size)) + self.mat_data[i]
            self.label_count[label] = self.label_count.get(label,0) + 1
            self.label_count_sum += 1

        self.num_label = len(self.label_count)
        self.label_map = self.label_count.keys()
        self.label_count_arr = array([self.label_count.values()])

        self.mat_mean_arr = array(self.mat_mean.values()) /\
                tile(self.label_count_arr.T,(1,self.col_size))
        self.mat_mean_arr += finfo(float).eps

        for i, label in enumerate(self.label_data):
            self.mat_variance[label] = \
                    self.mat_variance.get(label, zeros(self.col_size)) + \
                    ((self.mat_data[i] \
                    - self.mat_mean_arr[self.label_map.index(label)]) ** 2)

        self.mat_variance_arr = (array(self.mat_variance.values())\
                / tile(self.label_count_arr.T,(1,self.col_size))) ** 0.5
        self.mat_variance_arr += finfo(float).eps

    # Calculate gaussian probability
    #
    # ARG_MAX -> Label, 
    #    P( Label_i | X )
    #        = -1
    #          * SIGMA(j to n) {(x_j - mean_j)^2 / 2 * vari_j^2 + log(vari_j)}
    #          + log ( Count_Label_i ) - log ( Count_Label_all )
    #
    def predict(self, array_input):
        array_input = convert.list2npfloat(array_input)
        deviate_arr = self.mat_mean_arr - tile(array_input,(self.num_label,1))
        gaussian_bayes = (deviate_arr ** 2 / ((self.mat_variance_arr ** 2) * 2)) * -1 \
                - log(self.mat_variance_arr)
        gaussian_prob = gaussian_bayes.sum(axis=1) + log(self.label_count_arr)\
            -log(tile(array(self.label_count_sum),(1,self.num_label)))
        best_label_index = gaussian_prob[0].argsort()[::-1][0]
        return self.label_map[ best_label_index]

