import numpy as np

class adaBoost:
    def __init__(self):
        self.h = [(0.0)]
        self.hw = [0.0]
        
    def getHt(self, x, h):
        num = 0.0
        for i in xrange(len(x)):
            num = num + x[i] * h[i]
        num = num + h[-1]
        
        if (num > 0):
            return 1
        else:
            return -1
        
    def predict(self, x):
        num = 0.0
        for i in xrange(len(self.h)):
            num = num + self.getHt(x, self.h[i]) * self.hw[i]
        
        if (num > 0):
            return 1
        else:
            return -1
        
    def getError(self, x, y, h):
        num = self.getHt(x, h)
        
        if (num * y < 0):
            return 1.0
        else:
            return 0.0
        
    def fit(self, x, y):
        self.dimension = len(x[0])
        
        hs = []
        for i in xrange(self.dimension):
            arr = []
            for j in xrange(len(x)):
                arr.append(x[j][i])
            arr.sort()
            
            for j in xrange(len(arr) - 1):
                hij = []
                
                for k in xrange(self.dimension):
                    if (k == i):
                        hij.append(1.0)
                    else:
                        hij.append(0.0)
                
                hij.append(-1.0 * (arr[j] + arr[j + 1]) / 2.0)
                hij2 = []
                for num in hij:
                    hij2.append(-1 * num)
                
                hs.append(hij)
                hs.append(hij2)
                
        xweights = []
        for i in xrange(len(x)):
            xweights.append(1.0 / len(x))
            
        self.h = []
        self.hw = []
        
        steps = len(hs) / 2
        for step in xrange(steps):
            arr = []
            
            for i in xrange(len(hs)):
                error = 0.0
                for j in xrange(len(x)):
                    error = error + self.getError(x[j], y[j], hs[i]) * xweights[j]
                    
                arr.append((error, i))
                
            a = min(arr)
            error = a[0]
            num = a[1]
            
            if error > 0.5:
                break
            
            ht = hs[num]
            hwt = 0.5 * np.log((1.0 - error) / error)
            
            self.h.append(ht)
            self.hw.append(hwt)
            hs.remove(ht)
            
            xweights2 = []
            for i in xrange(len(x)):
                hti = self.getHt(x[i], ht)
                xweights2.append(xweights[i] * np.exp(-1 * hwt * y[i] * hti))
             
            zt = sum(xweights2)
            for i in xrange(len(xweights2)):
                xweights2[i] = xweights2[i] / zt
                
            xweights = xweights2
            
            totalError = 0
            for i in xrange(len(x)):
                num = self.predict(x[i])
                if (num * y[i] < 0):
                    totalError = totalError + 1
              
            print 'Error number:', totalError
            if totalError == 0:
                break
        
        print 'Training down!'
        print 
        
    def getH(self):
        return self.h
    
    def getHw(self):
        return self.hw

dataFeatures = [[-2, 19], [14, 16], [-9, 12], [-15, 15], [7, 20], [-13, 5], [16, 14], [25, 12], 
               [-18, 2], [23, 8], [13, -4], [4, -3], [0, 5], [-7, 6], [6, 4], [10, -6],
               [-6, 10], [5, -4], [16, 0], [15, -9]]

dataClass = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

cf = adaBoost()
cf.fit(dataFeatures, dataClass)
h = cf.getH()
hw = cf.getHw()

for i in xrange(len(h)):
    print h[i], hw[i]
