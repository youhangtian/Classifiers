import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate = 0.1, steps = 10000):
        self.weights = []
        self.bs = []
        for i in xrange(len(layers) - 1):
            w = 2 * np.random.random((layers[i], layers[i + 1])) - 1
            b = 2 * np.random.random(layers[i + 1]) - 1
            self.weights.append(w)
            self.bs.append(b)
         
        self.learning_rate = learning_rate
        self.steps = steps
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def getPrime(self, x):
        return x * (1.0 - x)
            
    def predict(self, x, inLabel = False):
        a = []
        for xx in x:
            a.append(xx)
        a = np.array(a)

        for i in xrange(len(self.weights)):
            a = np.dot(a, self.weights[i]) + self.bs[i]
            a = self.sigmoid(a)
            
        if (inLabel):
            arr = []
            for i in xrange(len(a)):
                arr.append((a[i], i))
            return max(arr)[1]
            
        return a
    
    def fit(self, features, labels):
        classNum = len(self.bs[-1])
        
        x = []
        for f in features:
            x2 = []
            for f2 in f:
                x2.append(f2)
            x.append(np.array(x2))
        y = []
        for l in labels:
            y2 = []
            for i in xrange(classNum):
                if (i == l):
                    y2.append(1.0)
                else:
                    y2.append(0.0)
            y.append(np.array(y2))
        
        for step in xrange(self.steps):
            rand = np.random.randint(len(x))
            a = [x[rand]]
            
            for i in xrange(len(self.weights)):
                a2 = np.dot(a[i], self.weights[i]) + self.bs[i]
                a2 = self.sigmoid(a2)
                a.append(a2)
               
            error = y[rand] - a[-1]
            deltas = [error * self.getPrime(a[-1])]
            
            for i in xrange(len(a) - 2, 0, -1):
                d2 = np.dot(deltas[-1], self.weights[i].T) * self.getPrime(a[i])
                deltas.append(d2)
                
            deltas.reverse()
            
            for i in xrange(len(self.weights)):
                ai = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                dweights = self.learning_rate * np.dot(ai.T, delta)
                self.weights[i] = self.weights[i] + dweights
                
                dbs = self.learning_rate * deltas[i]
                self.bs[i] = self.bs[i] + dbs
            
            if step % 1000 == 0:
                error = 0.0
                for i in xrange(len(x)):
                    error2 = y[i] - self.predict(x[i])
                    error2 = np.dot(error2, error2)
                    error = error + error2
                print 'Step:', step, 'Error:', error / len(x)
        print 'Training down!'
        print
    
    def getW(self):
        return self.weights
    
    def getB(self):
        return self.bs
    

if __name__ == '__main__':
    x = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
    
    y = [0, 0, 1, 1, 1, 1, 0, 0, 0]
    
    cf = NeuralNetwork([2, 10, 9, 2])
    cf.fit(x, y)
    for e in x:
        print (e, cf.predict(e))
