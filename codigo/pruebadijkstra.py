
from collections import defaultdict
import sys
 
 
class Heap():
 
    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []
 
    def newMinHeapNode(self, v, dist):
        minHeapNode = [v, dist]
        return minHeapNode

    def swapMinHeapNode(self, a, b):
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t
 

    def minHeapify(self, idx):
        smallest = idx
        left = 2*idx + 1
        right = 2*idx + 2
 
        if (left < self.size and
           self.array[left][1]
            < self.array[smallest][1]):
            smallest = left
 
        if (right < self.size and
           self.array[right][1]
            < self.array[smallest][1]):
            smallest = right
 

        if smallest != idx:
 

            self.pos[self.array[smallest][0]] = idx
            self.pos[self.array[idx][0]] = smallest
 
            self.swapMinHeapNode(smallest, idx)
 
            self.minHeapify(smallest)

    def extractMin(self):
 

        if self.isEmpty() == True:
            return
 

        root = self.array[0]
 

        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode
 

        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1
 

        self.size -= 1
        self.minHeapify(0)
 
        return root
 
    def isEmpty(self):
        return True if self.size == 0 else False
 
    def decreaseKey(self, v, dist):

        i = self.pos[v]
 
        self.array[i][1] = dist
 

        while (i > 0 and self.array[i][1] <
                  self.array[(i - 1) // 2][1]):
 

            self.pos[ self.array[i][0] ] = (i-1)//2
            self.pos[ self.array[(i-1)//2][0] ] = i
            self.swapMinHeapNode(i, (i - 1)//2 )
 
            i = (i - 1) // 2
 
    def isInMinHeap(self, v):
 
        if self.pos[v] < self.size:
            return True
        return False
 
 
def printArr(dist, n):
    print ("Vertex\tDistance from source")
    for i in range(n):
        print ("%d\t\t%d" % (i,dist[i]))
 
 
class Graph():
 
    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)
    def addEdge(self, src, dest, weight,oneway,riesgo):
        newNode = [dest, weight,riesgo]
        self.graph[src].insert(0, newNode)
        if(oneway==True):
            newNode = [src, weight,riesgo]
            self.graph[dest].insert(0, newNode)
    def dijkstra(self, src):
 
        V = self.V  
        dist = []  
 
        minHeap = Heap()
 
        for v in range(V):
            dist.append(1e7)
            minHeap.array.append( minHeap.
            newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)
 
        minHeap.pos[src] = src
        dist[src] = 0.0
        minHeap.decreaseKey(src, dist[src])
        minHeap.size = V
        distance=[[]]*V
        distance[src]=[src]
        while minHeap.isEmpty() == False:
            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]
            for pCrawl in self.graph[u]:

                v = pCrawl[0]
                if (minHeap.isInMinHeap(v) and
                     dist[u] != 1e7 and \
                     pCrawl[2] + dist[u] < dist[v]):
                        distance[v]=[]
                        for i in distance[u]:
                            distance[v].append(i)
                        distance[v].append(v)
                        dist[v] = pCrawl[2] + dist[u]
                        print(dist[v])
                        print(v)
                        if(v==5):
                            print(dist[5])
                        minHeap.decreaseKey(v, dist[v])
        print(distance[6])
        #printArr(dist,V)

graph = Graph(11)
graph.addEdge(0, 1, 4,True,0.2555555555555555555555555555555555555555555)
graph.addEdge(0, 7, 8,True,0.2555555555555555555555555555555555)
graph.addEdge(2, 3, 7,True,0.23333333333333333333333333)
graph.addEdge(2, 8, 2,False,0.222222222222222222222222222)
graph.addEdge(2, 5, 4,True,0.21314314111111111)
graph.addEdge(1, 2, 8,True,0.25555555555555555555555)
graph.addEdge(3, 4, 9,True,0.23141341414141)
graph.addEdge(3, 5, 14,True,0.21341341431341341341)
graph.addEdge(4, 5, 10,True,0.23141341341341341)
graph.addEdge(5, 6, 2,True,0.341341341341341341341)
graph.addEdge(1, 7, 11,True,0.244444444444444444444444444444)
graph.addEdge(6, 7, 1,True,0.2314134134134134134134141341)
graph.addEdge(6, 8, 6,True,0.21324134141341341341)
graph.addEdge(7, 8, 7,True,0.31341342342341314)
graph.addEdge(7, 9, 7,True,0.3343423423414)
graph.addEdge(4, 7, 7,True,0.312132131231231231)
graph.addEdge(9, 10, 7,True,0.9232312311231231)
graph.addEdge(4, 10, 7,True,0.3123123123121231)
graph.dijkstra(0)