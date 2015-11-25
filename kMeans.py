import numpy as np

def kMeans(cs, d, steps = 10):
    centers = []
    for c in cs:
        centers.append(c)
    data = np.array(d)
    
    for step in xrange(steps):
        centerSize = len(centers)
        dimension = len(data[0])

        dist = []
        for i in xrange(centerSize):
            dist.append([])

        for d in data:
            for i in xrange(centerSize):
                num = 0.0
                for j in xrange(dimension):
                    num = num + pow(d[j] - centers[i][j], 2)
                dist[i].append(num)

        data2 = []
        for i in xrange(centerSize):
            data2.append([])

        for i in xrange(len(data)):
            arr = []
            for j in xrange(centerSize):
                arr.append((dist[j][i], j))
            a = min(arr)

            data2[a[1]].append(data[i])

        data2 = np.array(data2)

        centers = []
        for i in xrange(centerSize):
            centers.append([])

        for i in xrange(centerSize):
            data2[i] = np.array(data2[i])
            for j in xrange(dimension):
                centers[i].append(np.mean(data2[i][:,j]))
                
    return centers

data = [[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]]
centers = [(1.8, 2.3), (4.1, 5.4)]

centers = kMeans(centers, data)
for c in centers:
    print c
