#
# SVM (Binary classifier)
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

import numpy as np
from pytrain.lib import convert
from pytrain.lib import ptmath

class SVM:

    def __init__(self, mat_data, label_data):
        self.x = mat_data
        self.y = label_data
        self.len_row = self.x.shape[0]
        self.alphas = np.mat(np.zeros((self.len_row, 1)))
        self.E_cache = np.mat(np.zeros((self.len_row, 2))) # [0] is dirty bit
        self.b = 0

    def trans_data_with_kernel(self, kernel, kernel_params = {}):
        self.K = np.mat(np.zeros((self.len_row, self.len_row)))
        for i in range(self.len_row):
            self.K[:,i] = self.trans_array_with_kernel(self.x[i,:], kernel, kernel_params)
            
    def trans_array_with_kernel(self, array_input, kernel, kernel_params = {}):
        K = np.mat(np.zeros((self.len_row, 1)))
        if kernel == 'Linear':
            K = self.x * array_input.T
        elif kernel == 'Polynomial':
            degree = kernel_params['degree']
            K1 = self.x * array_input.T
            K = K1
            for d in range(degree):
                if d >= 1:
                    K = np.multiply(K, K1)
        elif kernel == 'RBF':
            gamma = kernel_params['gamma']
            for j in range(self.len_row):
                delta_row = self.x[j] - array_input
                K[j] = delta_row * delta_row.T
            K = np.exp(K / (-1 * gamma ** 2))
        return K

    def clip_alpha(self, a, H, L):
        if a > H:
            a = H
        elif a < L:
            a = L
        return a

    def select_j_random(self, i):
        j = i
        while (j == i):
            j = int(np.random.uniform(0, self.len_row))
        return j
    
    def select_j(self, i):
        max_j = i
        max_delta_E = -1
        Ei = self.calc_E(i)
        valid_cache_index = np.nonzero(self.E_cache[:,0].A)[0]
        if len(valid_cache_index) > 0:
            for idx in valid_cache_index:
                if idx == i: continue
                delta_E = abs(Ei - self.E_cache[idx,1])
                if delta_E > max_delta_E:
                    max_j = idx
                    max_delta_E = delta_E
        else:
            max_j = self.select_j_random(i)
        return max_j
        
    def calc_E(self, n):
        Fn = float(np.multiply(self.alphas, self.y).T * self.K[:,n] + self.b)
        En = Fn - float(self.y[n])
        return En

    def update_E_cache(self, n):
        En = self.calc_E(n)
        self.E_cache[n] = [1, En]

    def smo_loop(self, i, C, toler):
        Ei = self.calc_E(i)
        if (self.y[i] * Ei < -toler and self.alphas[i] < C) or (self.y[i] * Ei > toler and self.alphas[i] > 0):
            j = self.select_j(i)
            Ej = self.calc_E(j)
            if self.y[i] != self.y[j] :
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(C, C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - C)
                H = min(C, self.alphas[j] + self.alphas[i])
                
            if L == H :
                return 0
            eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j]
            if eta >= 0: # eta cannot be positive
                return 0
            
            old_alpha_j = self.alphas[j]
            old_alpha_i = self.alphas[i]

            delta_alpha_j = self.y[j] * (Ej - Ei) / eta
            
            self.alphas[j] += delta_alpha_j
            self.alphas[j] = self.clip_alpha(self.alphas[j],H,L)
            delta_alpha_j = self.alphas[j] - old_alpha_j
            self.update_E_cache(j)           
            
            delta_alpha_i =  -1 * self.y[i] * self.y[j] * delta_alpha_j
            self.alphas[i] += delta_alpha_i
            self.update_E_cache(i)
            
            delta_b_i = Ei + delta_alpha_i * self.y[i] * self.K[i,i] + delta_alpha_j * self.y[j] * self.K[i,j]
            delta_b_j = Ej + delta_alpha_i * self.y[i] * self.K[i,j] + delta_alpha_j * self.y[j] * self.K[j,j]
            
            if (self.alphas[i] > 0) and (self.alphas[i] < C):
                self.b = self.b - delta_b_i
            elif (self.alphas[j] > 0) and (self.alphas[j] < C):
                self.b = self.b - delta_b_j
            else:
                self.b = self.b - (delta_b_i + delta_b_j) / 2
            return 1
        else:
            return 0

    def smo_plat(self, C, toler, epoch):
        now_epoch = 0
        go_full_over = True
        alpha_changed = 0
        while (now_epoch < epoch) and ((alpha_changed > 0) or go_full_over):
            alpha_changed = 0
            if go_full_over:
                for i in range(self.len_row):
                    alpha_changed += self.smo_loop(i, C, toler)
            else :
                non_bounded_idx = np.nonzero((self.alphas.A > 0) * (self.alphas.A < C))[0]
                for i in non_bounded_idx:
                    alpha_changed += self.smo_loop(i, C, toler)
            if go_full_over :
                go_full_over = False
            elif alpha_changed == 0:
                go_full_over = True
            now_epoch += 1

    def fit(self, C, toler, epoch, kernel = 'Linear', kernel_params = {}):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.trans_data_with_kernel(kernel, kernel_params)
        self.smo_plat(C, toler, epoch)

    def predict(self, array_input):
        kernel_input = self.trans_array_with_kernel(array_input, self.kernel, self.kernel_params)
        Fn = float(np.multiply(self.alphas, self.y).T * kernel_input + self.b)
        return np.sign(Fn)
