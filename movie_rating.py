import pandas as pd
import numpy as np

'''
This is a recommendation system using collaborative filtering.
In the training step, we want to get two matrices:
'userFs', which represents the user features matrix and 'itemFs', which represents the item features
The errow function we use is 'E = (targetValue - predictValue) ^ 2 + beta * (userFeatures ^ 2 + itemFeatures ^ 2)'
There are some different recommend methods in this system:
'getUserRec' will recommend items according to the predicted value by user features matrix and item features matrix
'getUserRec2' will recommend items according to similar users
'addRating' will add rating value to the data and give recommendation accroding to this new data
'newUser' will give recommendation according to the ratings by a new user
These fuctions are simitric to 'getItemRec', 'getItemRec2' and 'newItem'
'''

class CF:
    def __init__(self, k = 20, steps = 2000, alpha = 0.0002, beta = 0.02):
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        
        self.userLen = 0
        self.itemLen = 0
        self.allData = {(0, 0): 0.0}
        self.userData = [set()]
        self.itemData = [set()]
        self.userFs = 2 * np.random.random((0, 0)) - 1.0
        self.itemFs = 2 * np.random.random((0, 0)) - 1.0
        self.userDataMean = [0.0]
        self.userDataStd = [0.0]
    
    #Get the training error
    def getError(self):
        error = 0.0
        for key in self.allData.keys():
            id1 = key[0]
            id2 = key[1]
            value = np.dot(self.userFs[id1,:], self.itemFs[:,id2])                      
            dist = pow(value - self.allData[key], 2)
            error = error + dist
        return error / len(self.allData)
    
    #Train the data
    def fit(self, data, len1, len2):
        self.userLen = len1
        self.itemLen = len2
        self.allData = {}
        self.userData = []
        self.itemData = []
        self.userFs = 2 * np.random.random((len1, self.k)) - 1.0
        self.itemFs = 2 * np.random.random((self.k, len2)) - 1.0
        self.userDataMean = []
        self.userDataStd = []
        
        for i in xrange(len1):
            self.userData.append(set())
        for i in xrange(len2):
            self.itemData.append(set())
        
        for a in data:
            id1 = a[0]
            id2 = a[1]
            value = a[2]
            
            self.allData[(id1, id2)] = value
            self.userData[id1].add(id2)
            self.itemData[id2].add(id1)
            
        for step in xrange(self.steps):
            if (step % 100 == 0):
                print 'Steps:', step, ', Error:', self.getError()
            
            for a in data:
                id1 = a[0]
                id2 = a[1]
                value = a[2]
                
                e = value - np.dot(self.userFs[id1,:], self.itemFs[:,id2])
                self.userFs[id1,:] = self.userFs[id1,:] + self.alpha * (2 * e * self.itemFs[:,id2] - self.beta * self.userFs[id1,:])
                self.itemFs[:,id2] = self.itemFs[:,id2] + self.alpha * (2 * e * self.userFs[id1,:] - self.beta * self.itemFs[:,id2])
        print 'Steps:', self.steps, ', Error:', self.getError()
        
        for i in xrange(self.userLen):
            arr = []
            for j in self.userData[i]:
                arr.append(self.allData[(i, j)])
                
            if (len(arr) == 0):
                self.userDataMean.append(-1.0)
                self.userDataStd.append(-1.0)
            else:
                self.userDataMean.append(np.mean(arr))
                self.userDataStd.append(max(np.std(arr), 0.0001))
        
        print 'Training down!'
    
    #Check whether this is a valid user id
    def goodUser(self, userId):
        return (userId >= 0 and userId < self.userLen)
    
    #Check whether this is a valid item id
    def goodItem(self, itemId):
        return (itemId >= 0 and itemId < self.itemLen)
    
    #Add new rating to the data and give recommendation
    def addRating(self, userId, itemId, value, num1 = 10, num2 = 10, steps = 1000, alpha = 0.0002, beta = 0.02):
        if (not self.goodUser(userId) or not self.goodItem(itemId)):
            print 'Invalid user or item!'
            return [], []
        
        self.allData[(userId, itemId)] = value
        self.userData[userId].add(itemId)
        self.itemData[itemId].add(userId)
        
        arr = []
        for i in self.userData[userId]:
            if (i != itemId):
                arr.append((userId, i))
        for i in self.itemData[itemId]:
            if (i != userId):
                arr.append((i, itemId))
        
        np.random.shuffle(arr)
        
        userFs = self.userFs[userId,:]
        itemFs = self.itemFs[:,itemId]
        
        for step in xrange(steps):
            if (step % 10 == 0):
                id1 = userId
                id2 = itemId
                value = self.allData[(id1, id2)]
                
                e = value - np.dot(userFs, itemFs)
                userFs = userFs + alpha * (2 * e * itemFs - beta * userFs)
                itemFs = itemFs + alpha * (2 * e * userFs - beta * itemFs)
                
            else:
                (id1, id2) = arr[step % len(arr)]
                value = self.allData[(id1, id2)]
                
                if (id1 == userId):
                    e = value - np.dot(userFs, self.itemFs[:,id2])
                    userFs = userFs + alpha * (2 * e * self.itemFs[:,id2] - beta * userFs)
                else:
                    e = value - np.dot(self.userFs[id1,:], itemFs)
                    itemFs = itemFs + alpha * (2 * e * self.userFs[id1,:] - beta * itemFs)
                    
        arr1 = []
        arr2 = []
            
        for i in xrange(self.itemLen):
            if (i not in self.userData[userId]):
                value = np.dot(userFs, self.itemFs[:,i])
                arr1.append((value, i))
                
                if (len(arr1) > num1):
                    arr1.remove(min(arr1))
                
        for i in xrange(self.userLen):
            if (i not in self.itemData[itemId]):
                value = np.dot(self.userFs[i,:], itemFs)
                arr2.append((value, i))
                
                if (len(arr2) > num2):
                    arr2.remove(min(arr2))
           
        arr1.sort(reverse = True)
        arr2.sort(reverse = True)
        return arr1, arr2
    
    #Get similar users
    def similarUser(self, userId, num = 10):
        if (not self.goodUser(userId)):
            print 'No such user!'
            return []
        
        arr = []
            
        for i in xrange(self.userLen):
            if (i == userId):
                continue
            
            e = np.dot(self.userFs[userId,:], self.userFs[i,:])
            e = e / pow(np.dot(self.userFs[userId,:], self.userFs[i,:]), 0.5)
            e = e / pow(np.dot(self.userFs[i,:], self.userFs[i,:]), 0.5)
            arr.append((e, i))
            
            if (len(arr) > num):
                arr.remove(min(arr))
                
        arr.sort(reverse = True)
        return arr
    
    #Get similar items
    def similarItem(self, itemId, num = 10):
        if (not self.goodItem(itemId)):
            print 'No such item!'
            return []
        
        arr = []
            
        for i in xrange(self.itemLen):
            if (i == itemId):
                continue
                
            e = np.dot(self.itemFs[:,itemId], self.itemFs[:,i])
            e = e / pow(np.dot(self.itemFs[:,itemId], self.itemFs[:,itemId]), 0.5)
            e = e / pow(np.dot(self.itemFs[:,i], self.itemFs[:,i]), 0.5)
            arr.append((e, i))
            
            if (len(arr) > num):
                arr.remove(min(arr))
                
        arr.sort(reverse = True)
        return arr
    
    #Recommend items to this user
    def getUserRec(self, userId, num = 10):
        if (not self.goodUser(userId)):
            print 'No such user!'
            return []
        
        arr = []
        
        for i in xrange(self.itemLen):
            if (i not in self.userData[userId]):
                value = np.dot(self.userFs[userId,:], self.itemFs[:,i])
                arr.append((value, i))
                
                if (len(arr) > num):
                    arr.remove(min(arr))
                
        arr.sort(reverse = True)
        return arr
    
    #Recommend items to this user according to similar users
    def getUserRec2(self, userId, num = 10):
        if (not self.goodUser(userId)):
            print 'No such user!'
            return []
        
        similarUsers = self.similarUser(userId)
        
        table = {}
        for a in similarUsers:
            id1 = a[1]
            for id2 in self.userData[id1]:
                if (id2 in self.userData[userId]):
                    continue
                    
                value = self.allData[(id1, id2)]
                value = (value - self.userDataMean[id1]) / self.userDataStd[id1]
                
                if (id2 in table.keys()):
                    table[id2] = max(table[id2], value)
                else:
                    table[id2] = value
        arr = []
        for id2 in table.keys():
            arr.append((table[id2], id2))
        
            if (len(arr) > num):
                arr.remove(min(arr))
                
        arr.sort(reverse = True)
        return arr
    
    #Recommend similar items to a new user
    def newUser(self, data):
        arr = []
        for d in data:
            id2 = d[1]
            if (not self.goodItem(id2)):
                continue
                
            arr.append((d[1], id2))
            
            if (len(arr) > 3):
                arr.remove(min(arr))
                
        s = set()
        for a in arr:
            id2 = a[1]
            similarItems = self.similarItem(id2, num = 5)
            
            for a2 in similarItems:
                s.add(a2[1])
                
        return s
        
    #Recommend this item to users
    def getItemRec(self, itemId, num = 10):
        if (not self.goodItem(itemId)):
            print 'No such item!'
            return []
        
        arr = []
            
        for i in xrange(self.userLen):
            if (i not in self.itemData[itemId]):
                value = np.dot(self.userFs[i,:], self.itemFs[:,itemId])
                arr.append((value, i))
                
                if (len(arr) > num):
                    arr.remove(min(arr))
                
        arr.sort(reverse = True)
        return arr
    
    #Reommend this item to users according to similar items
    def getItemRec2(self, itemId, num = 10):
        if (not self.goodItem(itemId)):
            print 'No such item!'
            return []
        
        similarItems = self.similarItem(itemId)
        
        table = {}
        for a in similarItems:
            id2 = a[1]
            for id1 in self.itemData[id2]:
                if (id1 in self.itemData[itemId]):
                    continue
                    
                value = self.allData[(id1, id2)]
                value = (value - self.userDataMean[id1]) / self.userDataStd[id1]
                
                if (id1 in table.keys()):
                    table[id1] =  max(table[id1], value)
                else:
                    table[id1] = value
                
        arr = []
        for id1 in table.keys():
            arr.append((table[id1], id1))
            
            if (len(arr) > num):
                arr.remove(min(arr))
        
        arr.sort(reverse = True)
        return arr
    
    #Recommend this new item to similar users
    def newItem(self, data):
        arr = []
        for d in data:
            id1 = d[0]
            
            if (not self.goodUser(id1)):
                continue
                
            value = (d[1] - self.userDataMean[id1]) / self.userDataStd[id1]
            arr.append((value, id1))
            
            if (len(arr) > 3):
                arr.remove(min(arr))
                
        s = set()
        for a in arr:
            id1 = a[1]
            similarUsers = self.similarUser(id1, num = 5)
            
            for a2 in similarUsers:
                s.add(a2[1])
        
        return s
        
    def getUserLen(self):
        return self.userLen
    
    def getItemLen(self):
        return self.itemLen
    
    def getK(self):
        return self.k
    
    def getUserFs(self, userId):
        if (not self.goodUser(userId)):
            print 'No such user!'
            return []
        return self.userFs[userId,:]

    def getItemFs(self, itemId):
        if (not self.goodItem(itemId)):
            print 'No such item!'
            return []
        return self.itemFs[:,itemId]
    
    def getUserDataMean(self, userId):
        if (not self.goodUser(userId)):
            print 'No such user!'
        return self.userDataMean[userId]
    
    def getUserDataStd(self, userId):
        if (not self.goodUser(userId)):
            print 'No such user!'
        return self.userDataStd[userId]
    
    def getData(self, userId, itemId):
        if ((userId, itemId) not in self.allData.keys()):
            print 'No such user or item!'
            return -1.0
        return self.allData[(userId, itemId)]
    
    #Predict the rating
    def predict(self, userId, itemId):
        if (not self.goodUser(userId) or not self.goodItem(itemId)):
            print 'No such user or item!'
            return -1.0
        return np.dot(self.userFs[userId,:], self.itemFs[:,itemId])

    #Get the items rated by this user
    def getUserData(self, userId, withValue = False):
        if (not self.goodUser(userId)):
            print 'No such user!'
            return []
        
        arr = []
        for a in self.userData[userId]:
            if (withValue):
                arr.append((a, self.allData[(userId, a)]))
            else:
                arr.append(a)  
        return arr
    
    #Get the users who rated this item
    def getItemData(self, itemId, withValue = False):
        if (not self.goodItem(itemId)):
            print 'No such item!'
            return []
        
        arr = []
        for a in self.itemData[itemId]:
            if (withValue):
                arr.append((a, self.allData[(a, itemId)]))
            else:
                arr.append(a)
        return arr
    
if __name__ == '__main__':
    testSize = 200
    f = pd.read_csv('/Users/youhangtian/Downloads/movierating/smallratings.csv', header = 0)
    f = f[f['userId'] < testSize][f['movieId'] < testSize][['userId', 'movieId', 'rating']]
    print 'Data loaded!'
    print 

    data = []
    values = f.values

    maxid1 = 0
    maxid2 = 0
    for i in xrange(len(values)):
        id1 = int(values[i][0]) - 1
        id2 = int(values[i][1]) - 1
        value = values[i][2]
        data.append([id1, id2, value])
        
        maxid1 = max(maxid1, id1 + 1)
        maxid2 = max(maxid2, id2 + 1)
        
    cf = CF(steps = 500)
    cf.fit(data, maxid1, maxid2)
    print
    
    userId = 12
    itemId = 10
    
    arr1 = cf.getUserRec(userId)
    arr2 = cf.getUserRec2(userId)
    
    print 'Recommend movies for user', userId, ':'
    for a in arr1:
        print 'Movie', a[1], '\tExpected rating', a[0]
    print 'Recommend movies for user', userId, 'according to similar user:'
    for a in arr2:
        print 'Movie', a[1], '\tNormalized similar user\'s rating', a[0]
    print
    
    arr1 = cf.getItemRec(itemId)
    arr2 = cf.getItemRec2(itemId)
    
    print 'Recommend movie', itemId, 'to users:'
    for a in arr1:
        print 'User', a[1], '\tExpected rating', a[0]
    print 'Recommend movie', itemId, 'to users according to similar movie:'
    for a in arr2:
        print 'User', a[1], '\tNormalize similar movie\'s rating', a[0]
    print
