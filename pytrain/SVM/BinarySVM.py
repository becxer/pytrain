#
# Binary SVM
#
# @ author becxer
# @ reference Machine Learning in Action by Peter Harrington
# @ e-mail becxer87@gmail.com
#

from numpy import *

class BinarySVM:

    def __init__(self, mat_data, label_data):
        self.mat_data = mat_data
        self.label_data = label_data

    def selectJrand(self, i, m):
        j = i
        while j == i :
            j = int(random.uniform(0, m))
        return j

    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj

    def fit(self, C, toler, epoch):
        b, alphas = self.smo(C, toler, epoch)
        self.b = b
        self.alphas = alphas

    def predict(self, array_input):
        array_input = mat(array_input)
        dataMatrix = mat(self.mat_data)
        labelMat = mat(self.label_data).transpose()
        fXinput = float(multiply(self.alphas, labelMat).T * \
                   (dataMatrix * array_input.T)) + self.b
        return sign(fXinput[0,0])
    
    def smo(self, C, toler, maxIter):
        dataMatrix = mat(self.mat_data)
        labelMat = mat(self.label_data).transpose()
        b = 0;
        m, n = shape(dataMatrix) # m -> row , n -> col
        alphas = mat(zeros((m,1)))
        iter = 0
        while (iter < maxIter):
            alphaPairsChanged = 0
            for i in range(m):
                fXi =  float(multiply(alphas, labelMat).T * \
                        (dataMatrix * dataMatrix[i,:].T)) + b
                Ei = fXi - float(labelMat[i])
                if((labelMat[i] * Ei < -toler and (alphas[i] < C)) or \
                   (labelMat[i] * Ei > toler and (alphas[i] > 0))):
                   j = self.selectJrand(i,m)
                   fXj = float(multiply(alphas, labelMat).T * \
                          (dataMatrix * dataMatrix[j,:].T)) + b
                   Ej = fXj - float(labelMat[j])
                   alphaIold = alphas[i].copy()
                   alphaJold = alphas[j].copy()
                   if (labelMat[i] != labelMat[j]):
                       L = max(0, alphas[j] - alphas[i])
                       H = min(C, C + alphas[j] - alphas[i])
                   elif (labelMat[i] == labelMat[j]):
                       L = max(0, alphas[j] + alphas[i] - C)
                       H = min(C, alphas[j] + alphas[i])
                   if L == H: continue # L == H
                   eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - \
                     dataMatrix[i,:] * dataMatrix[i,:].T -\
                     dataMatrix[j,:] * dataMatrix[j,:].T
                   if eta >= 0: continue # eta >= 0
                   alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                   alphas[j] = self.clipAlpha(alphas[j], H, L)
                   if (abs(alphas[j] - alphaJold < 0.00001)): continue # j not moving
                   alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                   b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i,:] * dataMatrix[i,:].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i,:] * dataMatrix[j,:].T
                   b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i,:] * dataMatrix[j,:].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j,:] * dataMatrix[j,:].T
                   if (0 < alphas[i]) and (C > alphas[i]): b = b1
                   elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                   else : b = (b1 + b2) / 2.0
                   alphaPairsChanged += 1
            if(alphaPairsChanged == 0): iter += 1
            else: iter = 0
        return b, alphas    
