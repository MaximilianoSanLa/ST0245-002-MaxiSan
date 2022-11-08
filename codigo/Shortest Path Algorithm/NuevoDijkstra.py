from operator import le
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely import wkt
import geopandas as gpd

from collections import defaultdict
 
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
 
 
class Graph:
 
    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)
 
    def addEdge(self, src, dest, weight,oneway,risk):
        newNode = [dest, weight, risk]
        self.graph[src].insert(0, newNode)
        #Add edge if
        if(oneway==True):
            newNode = [src, weight,risk]
            self.graph[dest].insert(0, newNode)
 

    def dijkstra(self, src,dest,arbol):
        V = self.V
        dist = [float('inf')]*V
        distriesgo=[float('inf')]*V
        minHeap = Heap()
        for v in range(V):
            minHeap.array.append( minHeap.newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)

        minHeap.pos[src] = src
        dist[src] = 0.0
        distriesgo[src]=0.0
        minHeap.decreaseKey(src, dist[src])

        minHeap.size = V
        distance=[[]]*V
        distance[src].append(arbol.get(str(src)))

        while minHeap.isEmpty() == False:

            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            for pCrawl in self.graph[u]:
 
                v = pCrawl[0]
                if (minHeap.isInMinHeap(v) and dist[u] != float('inf') and \
                   pCrawl[1] + dist[u] < dist[v]):
                        dist[v] = pCrawl[1] + dist[u]
                        distriesgo[v]=(pCrawl[2]+distriesgo[u])/2
                        distance[v]=[]
                        for i in distance[u]:
                            distance[v].append(i)
                        distance[v].append(arbol.get(str(v)))

                        minHeap.decreaseKey(v, dist[v])
        print("Distancia 1: "+str(dist[dest]))
        print("Riesgo 1: "+str(distriesgo[dest]))
        return distance[dest]
    def dijkstra2(self, src,dest,arbol):
        V = self.V
        dist = [float('inf')]*V
        distriesgo=[float('inf')]*V
        distdistancia=[float('inf')]*V
        minHeap = Heap()
        for v in range(V):
            minHeap.array.append( minHeap.newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)

        minHeap.pos[src] = src
        dist[src] = 0.0
        minHeap.decreaseKey(src, dist[src])
        distriesgo[src]=0.0
        distdistancia[src]=0.0
        minHeap.size = V
        distance=[[]]*V
        distance[src].append(arbol.get(str(src)))
        #distance[src].append(Origen[Origen2.index(src)])
        while minHeap.isEmpty() == False:

            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            for pCrawl in self.graph[u]:
 
                v = pCrawl[0]
                if (minHeap.isInMinHeap(v) and dist[u] != float('inf') and \
                   (((pCrawl[2]*0.99) +(pCrawl[1]*0.01))+ dist[u]) < dist[v]):
                        distance[v]=[]
                        distriesgo[v]=(distriesgo[u]+pCrawl[2])/2
                        distdistancia[v]=(distdistancia[u]+pCrawl[1])
                        for i in distance[u]:
                            distance[v].append(i)
                        distance[v].append(arbol.get(str(v)))
                        dist[v] = pCrawl[2]*0.99 + pCrawl[1]*0.01+dist[u]
                        minHeap.decreaseKey(v, dist[v])
        print(dist[dest])
        print("distancia 2: "+str(distdistancia[dest]))
        print("Riesgo 2: "+str(distriesgo[dest]))
        return distance[dest]
    def dijkstra3(self, src,dest,arbol):
        V = self.V
        dist = [float('inf')]*V
        distriesgo=[float('inf')]*V
        minHeap = Heap()
        for v in range(V):
            minHeap.array.append( minHeap.newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)

        minHeap.pos[src] = src
        dist[src] = 0.0
        distriesgo[src] = 0.0
        minHeap.decreaseKey(src, dist[src])

        minHeap.size = V
        distance=[[]]*V
        distance[src].append(arbol.get(str(src)))
        while minHeap.isEmpty() == False:

            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            for pCrawl in self.graph[u]:
 
                v = pCrawl[0]
                if (minHeap.isInMinHeap(v) and dist[u] != 1e7 and \
                   (float(pCrawl[2])+float(dist[u]))/2< float(dist[v])):
                        dist[v] = ((float(pCrawl[2])+float(dist[u])))/2
                        distance[v]=[]
                        distriesgo[v]=(distriesgo[u]+pCrawl[1])
                        for i in distance[u]:
                            distance[v].append(i)
                        distance[v].append(arbol.get(str(v)))

                        minHeap.decreaseKey(v, dist[v])
        print("Distancia 3: "+str(distriesgo[dest]))
        print("Riesgo 3: "+ str(dist[dest]))
        return distance[dest]
def graficar(target,target2,target3):

    edges = pd.read_csv('calles_de_medellin_con_acoso.csv',sep=';')
    edges['geometry'] = edges['geometry'].apply(wkt.loads)
    edges = gpd.GeoDataFrame(edges)


    area = pd.read_csv('poligono_de_medellin.csv',sep=';')
    area['geometry'] = area['geometry'].apply(wkt.loads)
    area = gpd.GeoDataFrame(area)
    fig, ax = plt.subplots(figsize=(12,8))

    area.plot(ax=ax, facecolor='black')

    edges.plot(ax=ax, linewidth=0.1)
    arrlong=[]
    arrlat=[]
    for i in target:
        arrlong.append(float(i[1:i.index(",")]))
        arrlat.append(float(i[i.index(",")+2:len(i)-1]))
    line, = plt.plot(arrlong,arrlat,linewidth=0.4, c='b')
    line.set_color('yellow')


    #///////
    arrlong=[]
    arrlat=[]
    for i in target2:
        arrlong.append(float(i[1:i.index(",")]))
        arrlat.append(float(i[i.index(",")+2:len(i)-1]))
    line2, = plt.plot(arrlong,arrlat,linewidth=0.4, c='b')
    line2.set_color('green')

    #///////
    arrlong=[]
    arrlat=[]
    for i in target3:
        arrlong.append(float(i[1:i.index(",")]))
        arrlat.append(float(i[i.index(",")+2:len(i)-1]))
    line3, = plt.plot(arrlong,arrlat,linewidth=0.4, c='b')
    line3.set_color('red')

    plt.tight_layout()
    
    plt.savefig("mapa-de-called-con-longitud.png")
df = pd.read_csv('calles_de_medellin_con_acoso.csv',sep=';')
df.drop("Unnamed: 0",axis =1,inplace=True )
s=-1
#27661
ant=None
adentro=False
origenes2=[None]*68749
origenes=[None]*68749
origenes3=[None]*68749
origenes4=[None]*68749
origenes5=[None]*68749
Oriesgo=[None]*68749
d=0
arbol=dict()
for i in df.iterrows():
    origen=i[1][1]
    nombre= i[1][0]
    destino=i[1][2]
    length=i[1][3]
    riesgo=i[1][5]
    oneway=i[1][4]
    if(arbol.get(str(origen))==None):
        s+=1
        arbol[str(origen)]=s
        arbol[str(s)]=str(origen)
    Oriesgo[d]=riesgo
    origenes2[d]=destino
    origenes[d]=origen
    origenes3[d]=s
    origenes4[d]=oneway
    origenes5[d]=length
    d+=1
origenesSupletorio=[]
for i in range(0,len(origenes2)):
    if(arbol.get(str(origenes2[i]))==None):
        arbol[str(origenes2[i])]=s
        s+=1
print(s)
#for i in range(0,len(origenes2)):
#    if origenes2[i] not in origenes:
#        s+=1
#        print(origenes2[i])
#        origenes.append(origenes2[i])
#        doll=origenes3[len(origenes3)-1]+1
#        origenes3.append(doll)
#        origenes2[i]=origenes3[len(origenes3)-1]
#        print(origenes2[i])
#    else:
#        position=origenes.index(origenes2[i])
#        origenes2[i]=origenes3[position]
head=Graph(s+1)
#27671
src=arbol.get("(-75.5608489, 6.1960587)")
destino12=arbol.get("(-75.5796302, 6.2604275)")
for i in range(0,len(origenes2)):
    head.addEdge(int(origenes3[i]),int(arbol.get(str(origenes2[i]))),float(origenes5[i]),bool(origenes4[i]),float(Oriesgo[i]))
graficar(head.dijkstra(src,destino12,arbol),head.dijkstra2(src,destino12,arbol),head.dijkstra3(src,destino12,arbol))