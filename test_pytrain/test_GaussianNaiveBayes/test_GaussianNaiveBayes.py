#
# test Gaussian Naive Bayes
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.GaussianNaiveBayes import GaussianNaiveBayes
from pytrain.lib import nlp
from pytrain.lib import fs
from pytrain.lib import batch

import json
from numpy import *

class test_GaussianNaiveBayes(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        train_mat = [\
                [-65,-55,-42],[-20,-59,-71],[-43,-49,-69],\
                [-61,-30,-74],[-79,-81,-40],[-71,-57,-24],\
                [-67,-19,-58],[-57,-73,-83],[-68,-74,-59],\
                [-80,-85,-79]
            ]
        train_label = ['B','A','A','A','B','B','A','C','C','C']
        
        test_mat = [\
                [-45,-47,-74],[-77,-69,-25],[-64,-71,-59],\
                [-85,-85,-25],[-85,-85,-85]
            ]
        test_label = ['A','B','C','B','C']

        gnb = GaussianNaiveBayes(train_mat,train_label)
        gnb.fit()
        error_rate = batch.eval_predict(gnb, test_mat, test_label, self.logging)
        self.tlog("strength of signal predict error rate : " + str(error_rate))

class test_GaussianNaiveBayes_rssi(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        MAJOR_AP_COUNT = 17
        BAD_SIGNAL = -100
        
        areaf = open("sample_data/rssi/rssi.dat")
        area_json_list = areaf.readlines()
        areaf.close()
        area_set = {}
        ap_set = {}

        def compare(x,y):
            if x['rssi'] < y['rssi']:
                return 1
            elif x['rssi'] == y['rssi']:
                return 0
            else:
                return -1

        for aobj in area_json_list:
            area = json.loads(aobj)
            label = area["areaID"]
            aplist = area["apList"]
            aplist.sort(compare)
            for ap in aplist[:MAJOR_AP_COUNT]:
                ap_set[ap['bssid']] = 1
            area_set[label] = area_set.get(label,[])
            area_set[label].append(aplist[:MAJOR_AP_COUNT])
        
        ap_vector_column = ap_set.keys()

        train_mat = []
        train_label = []

        test_mat = []
        test_label = []

        count = 0;
        for label in area_set:
            for aps in area_set[label]:
                ap_vector = tile(BAD_SIGNAL, len(ap_vector_column))
                for ap in aps:
                    ap_vector[ap_vector_column.index(ap['bssid'])] = ap['rssi']
                
                count += 1
                if count % 10 == 0:
                    test_label.append(label)
                    test_mat.append(ap_vector)
                else :
                    train_label.append(label)
                    train_mat.append(ap_vector)

        gnb = GaussianNaiveBayes(train_mat,train_label)
        gnb.fit()
        error_rate = batch.eval_predict(gnb, test_mat, test_label, self.logging)
        self.tlog("rssi predict (with GaussianNaiveBayes) error rate : " + str(error_rate))

