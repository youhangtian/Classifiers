def kMeans(data, k, steps = 10):
    data = np.array(data)
    size = len(data)
    dimension = len(data[0])
    
    mins = []
    maxs = []
    for i in xrange(dimension):
        mins.append(min(data[:,i]))
        maxs.append(max(data[:,i]))
    
    centers = []
    for i in xrange(k):
        c = []
        for j in xrange(dimension):
            c.append(mins[j] + (maxs[j] - mins[j]) * np.random.random())
        centers.append(c)
        
    clusters = []
    
    for step in xrange(steps):
        dist = []
        for i in xrange(size):
            dist.append([])

        for i in xrange(size):
            for j in xrange(k):
                d = data[i] - centers[j]
                dist[i].append(np.dot(d, d))

        clusters = []
        for i in xrange(k):
            clusters.append([])

        for i in xrange(size):
            arr = []
            for j in xrange(k):
                arr.append((dist[i][j], j))
            a = min(arr)

            clusters[a[1]].append(i)

        centers = []
        for i in xrange(k):
            if (len(clusters[i]) == 0):
                c = []
                for j in xrange(dimension):
                    c.append(mins[j] + (maxs[j] - mins[j]) * np.random.random())
                centers.append(c)
            else:
                c = []
                for j in xrange(dimension):
                    c.append(np.mean(data[clusters[i], j]))
                centers.append(c)
                
    return centers, clusters

data = [[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]]
centers, clusters = kMeans(data, 2)

print centers
print clusters
