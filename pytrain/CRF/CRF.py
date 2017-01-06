#
# CRF
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

import numpy as np
from pytrain.lib import convert

class CRF:

    def __init__(self, mat_data, label_data, hidden_state_labeled = True, hidden_state = -1):
        self.eps = np.finfo(np.float).eps / 1000000000000
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
        self.pi = np.zeros(self.m)
        self.a = np.random.random([self.m, self.m]) + self.eps
        self.b = np.random.random([self.m, self.n]) + self.eps
        self.a = np.log(self.a / self.a.sum(axis=1).reshape((self.m,1)))
        self.b = np.log(self.b / self.b.sum(axis=1).reshape((self.m,1)))
        
    def make_freqtable(self):
        self.pi = np.zeros(self.m)
        self.a = np.zeros([self.m, self.m]) + self.eps
        self.b = np.zeros([self.m, self.n]) + self.eps
        for seq_idx, seq_label in enumerate(self.label_data):
            for i in range(len(seq_label)):
                now = seq_label[i]
                now_ob = self.mat_data[seq_idx][i]
                self.b[self.label_idx[now]] += now_ob
                if i >= 1:
                    prev = seq_label[i-1]
                    self.a[self.label_idx[prev]][self.label_idx[now]] += 1
        self.b = np.log(self.b / (self.b.sum(axis=1).reshape((self.m,1))))
        self.a = np.log(self.a / (self.a.sum(axis=1).reshape((self.m,1))))
        
    def viterbi(self, array_input):
        T = len(array_input)
        # self.prob :: index[0] is prob, index[1] is from idx
        self.prob = np.log(np.zeros([T, self.m, 2]) + self.eps)
        first_ob_idx = np.nonzero(array_input[0])[0]
        first = self.pi + self.b[:,first_ob_idx].sum(axis=1)
        first_prob = np.transpose(np.tile(first,(self.m,1)))
        first_prob[:,1:] = -1
        self.prob[0] = first_prob[:,:2]
        for t in range(1,T):
            now_ob_idx = np.nonzero(array_input[t])[0]
            for j in range(self.m):
                max_prob = self.prob[t][j][0]
                max_idx = self.prob[t][j][1]
                for i in range(self.m):
                    now_prob = self.prob[t-1][i][0] + \
                      self.a[i][j] + self.b[j,now_ob_idx].sum(axis=0)
                    if max_prob < now_prob:
                        max_prob = now_prob
                        max_idx = i
                self.prob[t][j][0] = max_prob
                self.prob[t][j][1] = max_idx
        last_idx = -1
        last_max = -10000000
        for i in range(self.m):
            if self.prob[T-1][i][0] > last_max:
                last_idx = int(i)
                last_max = self.prob[T-1][i][0]
        trace = []
        for at in range(T-1,-1,-1):
            trace.append(int(last_idx))
            last_idx = self.prob[at][int(last_idx)][1]
        return last_max, trace[::-1]

    def baum_welch(self, x_input):
        T = len(x_input)
        # alpha : probability of state i when see time 1...t
        alpha = np.zeros([T, self.m])
        # beta : probability of state i when see time t...T
        beta = np.zeros([T, self.m])
        # gamma : probability of state i when t
        gamma = np.zeros([T, self.m])
        # eta : probability of moving i to j when time t
        eta = np.zeros([T, self.m, self.m])

        # Get alpha, beta
        first_ob_idx = np.nonzero(x_input[0])[0]
        alpha[0] = self.pi + self.b[:,first_ob_idx].sum(axis=1)
        beta[T-1] = np.ones(self.m)
        for t in range(1,T):
            tr = T - t - 1
            forw_ob_idx = np.nonzero(x_input[t])[0]
            back_ob_idx = np.nonzero(x_input[tr])[0]
            for j in range(self.m):
                sum_prob_forw = 0
                sum_prob_back = 0
                for i in range(self.m):
                    sum_prob_forw += np.exp(alpha[t-1][i] + self.a[i][j] + self.b[j, forw_ob_idx].sum(axis=0))
                    sum_prob_back += np.exp(beta[tr+1][i] + self.a[j][i] + self.b[i, back_ob_idx].sum(axis=0))
                alpha[t][j] = np.log(sum_prob_forw)
                beta[tr][j] = np.log(sum_prob_back)
        beta[0] += self.pi
        
        # Get gamma
        for t in range(T):
            gamma_denom = 0
            for i in range(self.m):
                gamma[t][i] = alpha[t][i] + beta[t][i]
                gamma_denom += np.exp(gamma[t][i])
            gamma_denom = np.log(gamma_denom)
            gamma[t] -= gamma_denom
        gamma = np.exp(gamma) # gamma is not log value !
            
        # Get eta
        for t in range(1, T):
            eta_denom = 0
            ob_idx = np.nonzero(x_input[t])[0]
            for i in range(self.m):
                for j in range(self.m):
                    eta[t-1][i][j] = alpha[t-1][i] + self.a[i][j] + \
                                 self.b[j, ob_idx].sum(axis=0) + beta[t][j]
                    eta_denom += np.exp(eta[t-1][i][j])
            eta_denom = np.log(eta_denom)
            eta[t-1] -= eta_denom
        eta = np.exp(eta) # eta is not log value !
        eta[T-1] = 0

        # Get expected pi, a, b
        Epi = gamma[0]
        Ea = eta.sum(axis=0)/np.transpose([gamma.sum(axis=0)])
        Eb = np.transpose(np.matmul(np.array(x_input).T, gamma) \
          / gamma.sum(axis=0))
        return Epi, Ea, Eb
        
    def fit(self, toler, epoch):
        for i in range(epoch):
            Epi = [];  Ea = []; Eb = []
            for idx in range(len(self.mat_data)):
                x_input = self.mat_data[idx]
                epi, ea, eb = self.baum_welch(x_input)
                Epi.append(epi)
                Ea.append(ea)
                Eb.append(eb)
            npi = np.log(np.array(Epi).sum(axis=0) + self.eps) - np.log(len(Epi)) 
            na = np.log(np.array(Ea).sum(axis=0) + self.eps) - np.log(len(Ea))
            nb = np.log(np.array(Eb).sum(axis=0) + self.eps) - np.log(len(Eb))

            tolpi = np.average(np.abs(np.exp(self.pi) - np.exp(npi)))
            tola = np.average(np.abs(np.exp(self.a) - np.exp(na)))
            tolb = np.average(np.abs(np.exp(self.b) - np.exp(nb)))

            if tolpi < toler and tola < toler and tolb < toler:
                break
            else:
                self.pi = npi
                self.a = na
                self.b = nb
            
    def predict(self, array_input, with_prob = False):
        prob, seq_of_label = self.viterbi(array_input)
        ret = [self.label_set[x] for x in seq_of_label]
        if with_prob:
            return ret, prob
        else:
            return ret
