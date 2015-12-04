import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, C = 200, toler = 0.001, sigma = 1.3, maxIter = 10000):
        self.C = C
        self.tol = toler
        self.sigma = sigma
        self.maxIter = maxIter
        
        self.x = []
        self.y = []
        self.alphas = []
        self.b = 0
        
        self.K = []
        
        self.s = set()
        
        self.x2 = []
        self.y2 = []
        self.alphas2 = []
    
    def getSupportVectors(self):
        return self.x2, self.y2, self.alphas2
    
    def getC(self):
        return self.C
    
    def getK(self, x):
        k = np.zeros(len(self.x))
        for i in xrange(len(self.x)):
            dist = self.x[i] - x
            dist = np.dot(dist, dist)
            k[i] = np.exp(-dist / pow(self.sigma, 2))
        return k
    
    def getK2(self, x):
        k = np.zeros(len(self.x2))
        for i in xrange(len(self.x2)):
            dist = self.x2[i] - x
            dist = np.dot(dist, dist)
            k[i] = np.exp(-dist / pow(self.sigma, 2))
        return k
    
    def predict(self, x):
        k = self.getK2(x)
        return np.dot(self.alphas2 * self.y2, k) + self.b
    
    def getE(self, i):
        fxi = np.dot(self.alphas * self.y, self.K[i]) + self.b
        e = fxi - self.y[i]
        return e
    
    def getJ(self, i, Ei):
        j = -1
        Ej = -1
        dE = -1
        
        for k in self.s:
            if (k != i):
                Ek = self.getE(k)
                dE2 = abs(Ei - Ek)
                
                if (dE2 > dE):
                    j = k
                    Ej = Ek
                    dE = dE2
            
        if (j == -1):                
            for k in xrange(len(self.x)):
                if (k != i):
                    Ek = self.getE(k)
                    dE2 = abs(Ei - Ek)
                    
                    if (dE2 > dE):
                        j = k
                        Ej = Ek
                        dE = dE2
        
        return j, Ej
    
    def updateAlpha(self, i):
        Ei = self.getE(i)
        if ((self.y[i] * Ei) < self.tol):
            j, Ej = self.getJ(i, Ei)
            
            oldI = self.alphas[i].copy()
            oldJ = self.alphas[j].copy()
            
            left = 0
            right = self.C
            if (self.y[i] != self.y[j]):
                left = max(0, self.alphas[j] - self.alphas[i])
                right = min(self.C, self.alphas[j] + self.C - self.alphas[i])
            else:
                left = max(0, self.alphas[j] - (self.C - self.alphas[i]))
                right = min(self.C, self.alphas[j] + self.alphas[i])
            
            if (left == right):
                return 0
            
            eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if (eta >= 0):
                return 0
            
            self.alphas[j] -= self.y[j] * (Ei - Ej) / eta
            self.alphas[j] = max(left, self.alphas[j])
            self.alphas[j] = min(right, self.alphas[j])
            
            self.s.add(j)
            if (self.alphas[j] == 0):
                self.s.remove(j)
             
            if (abs(self.alphas[j] - oldJ) < 0.00001):
                return 0
            
            self.alphas[i] += self.y[i] * self.y[j] * (oldJ - self.alphas[j])
            
            self.s.add(i)
            if (self.alphas[i] == 0):
                self.s.remove(i)
            
            b1 = self.b - Ei
            b1 -= self.y[i] * (self.alphas[i] - oldI) * self.K[i, i]
            b1 -= self.y[j] * (self.alphas[j] - oldJ) * self.K[i, j]
            b2 = self.b - Ej
            b2 -= self.y[i] * (self.alphas[i] - oldI) * self.K[i, j]
            b2 -= self.y[j] * (self.alphas[j] - oldJ) * self.K[j, j]
            
            if (self.alphas[i] > 0 and self.alphas[i] < self.C):
                self.b = b1
            elif (self.alphas[j] > 0 and self.alphas[j] < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            
            return 1
        else:
            return 0
            
    def fit(self, features, labels):
        self.x = np.array(features)
        self.y = np.array(labels)
        
        self.alphas = np.zeros(len(features))
        self.b = 0.0
        
        self.s = set()
        
        self.K = np.zeros((len(features), len(features)))
        for i in xrange(len(self.x)):
            self.K[i] = self.getK(self.x[i])
        
        iterNum = 0
        alphaChanged = 1
        
        while ((iterNum < self.maxIter) and (alphaChanged > 0)):
            alphaChanged = 0
            for i in xrange(len(self.x)):
                alphaChanged += self.updateAlpha(i)
            iterNum += 1
       
        self.x2 = []
        self.y2 = []
        self.alphas2 = []
        for i in xrange(len(self.alphas)):
            if (self.alphas[i] > 0):
                self.x2.append(self.x[i])
                self.y2.append(self.y[i])
                self.alphas2.append(self.alphas[i])
        self.x2 = np.array(self.x2)
        self.y2 = np.array(self.y2)
        self.alphas2 = np.array(self.alphas2)
        
        self.x = []
        self.y = []
        self.alphas = []
        self.K = []
        
        print 'Training down!'
        print
        

if __name__ == '__main__':    
    x = [[3.542485, 1.977398], [3.018896, 2.556416], [7.55151, -1.58003], [2.114999, -0.004466], 
         [8.127113, 1.274372], [7.108772, -0.986906], [8.610639, 2.046708], [2.326297, 0.265213], 
         [3.634009, 1.730537], [0.341367, -0.894998], [3.125951, 0.293251], [2.123252, -0.783563], 
         [0.887835, -2.797792], [7.139979, -2.329896], [1.696414, -1.212496], [8.117032, 0.623493], 
         [8.497162, -0.266649], [4.658191, 3.507396], [8.197181, 1.545132], [1.208047, 0.2131], 
         [1.928486, -0.32187], [2.175808, -0.014527], [7.886608, 0.461755], [3.223038, -0.552392], 
         [3.628502, 2.190585], [7.40786, -0.121961], [7.286357, 0.251077], [2.301095, -0.533988], 
         [-0.232542, -0.54769], [3.457096, -0.082216], [3.023938, -0.057392], [8.015003, 0.885325], 
         [8.991748, 0.923154], [7.916831, -1.781735], [7.616862, -0.217958], [2.450939, 0.744967], 
         [7.270337, -2.507834], [1.749721, -0.961902], [1.803111, -0.176349], [8.804461, 3.044301], 
         [1.231257, -0.568573], [2.074915, 1.41055], [-0.743036, -1.736103], [3.536555, 3.96496], 
         [8.410143, 0.025606], [7.382988, -0.478764], [6.960661, -0.245353], [8.23446, 0.701868], 
         [8.168618, -0.903835], [1.534187, -0.622492], [9.229518, 2.066088], [7.886242, 0.191813], 
         [2.893743, -1.643468], [1.870457, -1.04042], [5.286862, -2.358286], [6.080573, 0.418886], 
         [2.544314, 1.714165], [6.016004, -3.753712], [0.92631, -0.564359], [0.870296, -0.109952], 
         [2.369345, 1.375695], [1.363782, -0.254082], [7.27946, -0.189572], [1.896005, 0.51508], 
         [8.102154, -0.603875], [2.529893, 0.662657], [1.963874, -0.365233], [8.132048, 0.785914], 
         [8.245938, 0.372366], [6.543888, 0.433164], [-0.236713, -5.766721], [8.112593, 0.295839], 
         [9.803425, 1.495167], [1.497407, -0.552916], [1.336267, -1.632889], [9.205805, -0.58648], 
         [1.966279, -1.840439], [8.398012, 1.584918], [7.239953, -1.764292], [7.556201, 0.241185], 
         [9.015509, 0.345019], [8.266085, -0.230977], [8.54562, 2.788799], [9.295969, 1.346332], 
         [2.404234, 0.570278], [2.037772, 0.021919], [1.727631, -0.453143], [1.979395, -0.050773], 
         [8.092288, -1.372433], [1.667645, 0.239204], [9.854303, 1.365116], [7.921057, -1.327587], 
         [8.500757, 1.492372], [1.339746, -0.291183], [3.107511, 0.758367], [2.609525, 0.902979], 
         [3.263585, 1.367898], [2.912122, -0.202359], [1.731786, 0.589096], [2.387003, 1.573131]]
    y = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 
         1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 
         1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
         -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 
         1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 
         1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 
         -1.0, -1.0, -1.0, -1.0]
    
    x2 = [[-0.214824, 0.662756], [-0.061569, -0.091875], [0.406933, 0.648055], [0.22365, 0.130142], 
          [0.231317, 0.766906], [-0.7488, -0.531637], [-0.557789, 0.375797], [0.207123, -0.019463], 
          [0.286462, 0.71947], [0.1953, -0.179039], [-0.152696, -0.15303], [0.384471, 0.653336], 
          [-0.11728, -0.153217], [-0.238076, 0.000583], [-0.413576, 0.145681], [0.490767, -0.680029], 
          [0.199894, -0.199381], [-0.356048, 0.53796], [-0.392868, -0.125261], [0.353588, -0.070617], 
          [0.020984, 0.92572], [-0.475167, -0.346247], [0.074952, 0.042783], [0.394164, -0.058217], 
          [0.663418, 0.436525], [0.402158, 0.577744], [-0.449349, -0.038074], [0.61908, -0.088188], 
          [0.268066, -0.071621], [-0.015165, 0.359326], [0.539368, -0.374972], [-0.319153, 0.629673], 
          [0.694424, 0.64118], [0.079522, 0.193198], [0.253289, -0.285861], [-0.035558, -0.010086], 
          [-0.403483, 0.474466], [-0.034312, 0.995685], [-0.590657, 0.438051], [-0.098871, -0.023953], 
          [-0.250001, 0.141621], [-0.012998, 0.525985], [0.153738, 0.491531], [0.388215, -0.656567], 
          [0.049008, 0.013499], [0.068286, 0.392741], [0.7478, -0.06663], [0.004621, -0.042932], 
          [-0.7016, 0.190983], [0.055413, -0.02438], [0.035398, -0.333682], [0.211795, 0.024689], 
          [-0.045677, 0.172907], [0.595222, 0.20957], [0.229465, 0.250409], [-0.089293, 0.068198], 
          [0.3843, -0.17657], [0.834912, -0.110321], [-0.307768, 0.503038], [-0.777063, -0.348066], 
          [0.01739, 0.152441], [-0.293382, -0.139778], [-0.203272, 0.286855], [0.957812, -0.152444], 
          [0.004609, -0.070617], [-0.755431, 0.096711], [-0.526487, 0.547282], [-0.246873, 0.833713], 
          [0.185639, -0.066162], [0.851934, 0.456603], [-0.827912, 0.117122], [0.233512, -0.106274], 
          [0.583671, -0.709033], [-0.487023, 0.62514], [-0.448939, 0.176725], [0.155907, -0.166371], 
          [0.334204, 0.381237], [0.081536, -0.106212], [0.227222, 0.527437], [0.75929, 0.33072], 
          [0.204177, -0.023516], [0.577939, 0.403784], [-0.568534, 0.442948], [-0.01152, 0.021165], 
          [0.87572, 0.422476], [0.297885, -0.632874], [-0.015821, 0.031226], [0.541359, -0.205969], 
          [-0.689946, -0.508674], [-0.343049, 0.841653], [0.523902, -0.436156], [0.249281, -0.71184], 
          [0.193449, 0.574598], [-0.257542, -0.753885], [-0.021605, 0.15808], [0.601559, -0.727041], 
          [-0.791603, 0.095651], [-0.908298, -0.053376], [0.12202, 0.850966], [-0.725568, -0.292022]]

    y2 = [-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 
          -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 
          1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 
          1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 
          1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 
          -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    
    features = x2
    labels = y2
    
    cf = SVM(C = 100, sigma = 1.5, toler = 0.0001, maxIter = 1000)
    cf.fit(features, labels)
    
    x, y, alphas = cf.getSupportVectors()
    c = cf.getC()
    for i in xrange(len(x)):
        print x[i], y[i], alphas[i]
    print
    
    x2 = []
    y2 = []
    for i in xrange(len(x)):
        if (alphas[i] < c):
            x2.append(x[i])
            y2.append(y[i])
        
    cf2 = SVM()
    cf2.fit(x2, y2)
        
    x2, y2, alphas2 = cf2.getSupportVectors()
    for i in xrange(len(x2)):
        print x2[i], y2[i], alphas2[i]
    print
    
    for i in xrange(len(features)):
        print cf2.predict(features[i]) * labels[i]
    print
