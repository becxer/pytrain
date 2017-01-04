#
# HMM
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

import numpy as np
from pytrain.lib import convert

class HMM:

    def __init__(self, mat_data, label_data, hidden_state_labeled = True, hidden_state = -1):
        self.mat_data = mat_data
        self.label_data = label_data
        self.n = len(mat_data[0][0])
        if hidden_state_labeled :
            self.label_set = []
            for seq_label in label_data:
                self.label_set.extend(seq_label)
            self.label_set = list(set(self.label_set))
            self.label_idx = { x:i for i, x in enumerate(self.label_set)}
            self.m = len(self.label_set)
            self.make_freqtable()
        elif not hidden_state_labeled:
            self.m = hidden_state
            self.label_set = list(range(hidden_state))
            self.make_randomtable()

    def make_randomtable(self):
        self.a = np.random.random([self.m, self.m])
        self.b = np.random.random([self.m, self.n])
        self.a = np.log(self.a / self.a.sum(axis=1).reshape((self.m,1)))
        self.b = np.log(self.b / self.b.sum(axis=1).reshape((self.m,1)))
        
    def make_freqtable(self):
        self.a = np.zeros([self.m, self.m]) + 0.000001
        self.b = np.zeros([self.m, self.n]) + 0.000001
        for seq_idx, seq_label in enumerate(self.label_data):
            for i in range(1, len(seq_label)):
                now = seq_label[i]
                prev = seq_label[i-1]
                now_ob = self.mat_data[seq_idx][i]
                self.b[self.label_idx[now]] += now_ob
                self.a[self.label_idx[prev]][self.label_idx[now]] += 1
        self.b = np.log(self.b / self.b.sum(axis=1).reshape((self.m,1)))
        self.a = np.log(self.a / self.a.sum(axis=1).reshape((self.m,1)))
        
    def viterbi(self, array_input):
        t = len(array_input)
        # self.prob :: index[0] is prob, index[1] is from idx
        self.prob = np.zeros([t, self.m, 2]) - 10000000
        first_ob_idx = np.nonzero(array_input[0])[0]
        first = self.b[:,first_ob_idx].sum(axis=1)
        first_prob = np.transpose(np.tile(first,(self.m,1)))
        first_prob[:,1:] = -1
        self.prob[0] = first_prob[:,:2]
        for i in range(1,t):
            now_ob_idx = np.nonzero(array_input[i])[0]
            for j in range(self.m):
                max_prob = self.prob[i][j][0]
                max_idx = self.prob[i][j][1]
                for k in range(self.m):
                    now_prob = self.prob[i-1][k][0] + \
                      self.a[k][j] + self.b[j,now_ob_idx].sum(axis=0)
                    if max_prob < now_prob:
                        max_prob = now_prob
                        max_idx = k
                self.prob[i][j][0] = max_prob
                self.prob[i][j][1] = max_idx
        last_idx = -1
        last_max = -10000000
        for j in range(self.m):
            if self.prob[t-1][j][0] > last_max:
                last_idx = int(j)
                last_max = self.prob[t-1][j][0]
        trace = []
        for at in range(t-1,-1,-1):
            trace.append(int(last_idx))
            last_idx = self.prob[at][int(last_idx)][1]
        return trace[::-1]
        
    def fit(self, toler, epoch):
        # TODO : Baum-welch EM algorithm implementation
        
        pass
    
    def predict(self, array_input):
        seq_of_label = self.viterbi(array_input)
        ret = [self.label_set[x] for x in seq_of_label]
        return ret
