from operator import le
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely import wkt
import geopandas as gpd
class Heap:
 
    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []
 
    def newMinHeapNode(slef,v, dist):
        minHeapNode = [v, dist]
        return minHeapNode
    def swapMinHeapNode(self,a, b):
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t
    def minHeapify(self,idx):
        smallest = idx
        left = 2*idx + 1
        right = 2*idx + 2
 
        if (left < self.size and self.array[left][1] < self.array[smallest][1]):
            smallest = left
 
        if (right < self.size and self.array[right][1] < self.array[smallest][1]):
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
        if(dist[i]<200):
            print ("%d\t\t%d" % (i,dist[i]))

 
class Graph:
 
    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)
 
    def addEdge(self, src, dest, weight,oneway):
        newNode = [dest, weight]
        self.graph[src].insert(0, newNode)
        #Add edge if
        if(oneway==True):
            newNode = [src, weight]
            self.graph[dest].insert(0, newNode)
 

    def dijkstra(self, src,dest,Origen,Origen2,Origen3,Origen4,Origen5):
 
        V = self.V
        dist = [] 
        minHeap = Heap()

        for v in range(V):
            dist.append(1e7)
            minHeap.array.append( minHeap.newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)

        minHeap.pos[src] = src
        dist[src] = 0
        minHeap.decreaseKey(src, dist[src])

        minHeap.size = V
 
        while minHeap.isEmpty() == False:

            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            for pCrawl in self.graph[u]:
 
                v = pCrawl[0]
 
                if (minHeap.isInMinHeap(v) and dist[u] != 1e7 and \
                   pCrawl[1] + dist[u] < dist[v]):
                        dist[v] = pCrawl[1] + dist[u]
                        minHeap.decreaseKey(v, dist[v])
        printArr(dist,V)
        #a=encontrarPuntoMasCorto(Origen,Origen2,Origen3,dest,Origen4,src,dist[dest],Origen5,None,["nada"])
        #if(a!=False):
        #    for i in range(0,len(a)):
        #        print(a[i]+"->")
        #else:
        #    print("Fallo")
def graficar(src,target):
    pass

    edges = pd.read_csv('calles_de_medellin_con_acoso.csv',sep=';')
    a=edges['name'].iloc[[68736]]
    edges['geometry'] = edges['geometry'].apply(wkt.loads)
    edges = gpd.GeoDataFrame(edges)


    area = pd.read_csv('poligono_de_medellin.csv',sep=';')
    area['geometry'] = area['geometry'].apply(wkt.loads)
    area = gpd.GeoDataFrame(area)

    fig, ax = plt.subplots(figsize=(12,8))

    area.plot(ax=ax, facecolor='black')

    edges.plot(ax=ax, linewidth=0.1,edgecolor='dimgray')
    
    plt.tight_layout()
    plt.savefig("mapa-de-called-con-longitud.png")
#Separate code dont mind this ////////////////////////////////////////////////////////////////////////
def encontrarPuntoMasCorto(listaPunto,Origen,destino,destino2,both,src,maxlength,lengths,previous,arreglo,counter=0):
    a=False
    b=False
    for i in range(0,len(destino)):
        if(int(counter)>int(maxlength)):
            return False
        if(int(src)==int(destino2)):
            return arreglo
        if((int(Origen[i])==int(src) and previous!=int(destino[i]))):
            if(arreglo!=None):
                if destino[i] not in arreglo:
                    a=encontrarPuntoMasCorto(listaPunto,Origen,destino,destino2,both,destino[i],maxlength,lengths,Origen[i],arreglo.append(listaPunto[i]),int(counter)+int(lengths[i]))
                print(arreglo)
            else:
                #arreglo=[].append(listaPunto[i])
                a=encontrarPuntoMasCorto(listaPunto,Origen,destino,destino2,both,destino[i],maxlength,lengths,Origen[i],[].append(listaPunto[i]),int(counter)+int(lengths[i]))
        if((a!=False)and(a!=None)):
            return a
        if((int(destino[i])==int(src) and bool(both[i])==True and previous!=int(Origen[i]))):
            if(arreglo!=None):
                if Origen[i] not in arreglo:
                    b=encontrarPuntoMasCorto(listaPunto,Origen,destino,int(destino2),both,int(Origen[i]),maxlength,lengths,destino[i],arreglo.append(listaPunto[i]),lengths[i]+counter)

            else:

                #arreglo=[].append(listaPunto[i])
                
                b=encontrarPuntoMasCorto(listaPunto,Origen,destino,destino2,both,Origen[i],maxlength,lengths,destino[i],[].append(listaPunto[i]),lengths[i]+counter)
            if(arreglo!=None):
                print(arreglo)
        if((b!=False)and(b!=None)):
            return b
    return a or b
#more code separation ///////////////////////////////////////////////////////////////////////////////////
df = pd.read_csv('calles_de_medellin_con_acoso.csv',sep=';')
df.drop("Unnamed: 0",axis =1,inplace=True )
s=27662
c=0
ant=None
adentro=False
origenes2=[None]*68749
origenes=[None]*68749
origenes3=[None]*68749
origenes4=[None]*68749
origenes5=[None]*68749
d=0
for i in df.iterrows():
    origen=i[1][1]
    nombre= i[1][0]
    destino=i[1][2]
    length=i[1][3]
    riesgo=i[1][5]
    oneway=i[1][4]
    origenes2[d]=destino
    origenes[d]=origen
    origenes3[d]=c
    origenes4[d]=oneway
    origenes5[d]=length
    if(c!=0):
        if(str(origen)==str(ant)):
            adentro=True
    ant=origen
    if(adentro==False):
        c+=1
    d+=1
    adentro=False
print(origenes[0])
print(origenes[1])
didntbreak=False
for i in range(0,len(origenes2)):
    for j in range(0,len(origenes)):
        if(origenes2[i]==origenes[j]):
            origenes2[i]=origenes3[j]
            didntbreak=True
            break
    if(didntbreak==False):
        s+=1
        print(origenes2[i])
        origenes.append(origenes2[i])
        doll=origenes3[len(origenes3)-1]+1
        origenes3.append(doll)
        origenes2[i]=origenes3[len(origenes3)-1]
        print(origenes2[i])
    didntbreak=False
head=Graph(s)
#27671 
for i in range(0,len(origenes2)):
    head.addEdge(int(origenes3[i]),int(origenes2[i]),int(origenes5[i]),bool(origenes4[i]))
head.dijkstra(0,6,origenes,origenes3,origenes2,origenes4,origenes5)

