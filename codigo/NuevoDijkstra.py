from operator import le
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely import wkt
import geopandas as gpd
from collections import deque
import requests
import folium
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
        prev=None
        desiredPath=[Origen[Origen2.index(src)]]
        V = self.V
        dist = [] 
        minHeap = Heap()
        notdone=False
        started=False
        for v in range(V):
            dist.append(1e7)
            minHeap.array.append( minHeap.newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)

        minHeap.pos[src] = src
        dist[src] = 0
        minHeap.decreaseKey(src, dist[src])

        minHeap.size = V
        distance=[[]]*V
        #distance[src].append(Origen[Origen2.index(src)])
        while minHeap.isEmpty() == False:

            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            for pCrawl in self.graph[u]:
 
                v = pCrawl[0]
                if (minHeap.isInMinHeap(v) and dist[u] != 1e7 and \
                   pCrawl[1] + dist[u] < dist[v]):
                        dist[v] = pCrawl[1] + dist[u]
                        if(len(distance[v])!=0):
                            distance[v]=[]
                        for i in distance[u]:
                            distance[v].append(i)
                        distance[v].append(Origen[Origen2.index(v)])

                        minHeap.decreaseKey(v, dist[v])
        print(Origen[Origen2.index(dest)])
        #printArr(dist,V)
        print(distance[dest])
        graficar(distance[dest])
def graficar(target):

    edges = pd.read_csv('calles_de_medellin_con_acoso.csv',sep=';')
    edges['geometry'] = edges['geometry'].apply(wkt.loads)
    edges = gpd.GeoDataFrame(edges)


    area = pd.read_csv('poligono_de_medellin.csv',sep=';')
    area['geometry'] = area['geometry'].apply(wkt.loads)
    area = gpd.GeoDataFrame(area)
    fig, ax = plt.subplots(figsize=(12,8))

    area.plot(ax=ax, facecolor='black')

    edges.plot(ax=ax, linewidth=0.1)
    prev1=None
    prev2=None
    #for point in [points[0], points[-1]]:
    #    folium.Marker(point).add_to(fig)
    #folium.PolyLine(points, weight=5, opacity=1).add_to(fig)
    arrlong=[]
    arrlat=[]
    for i in target:
        print(i[1:i.index(",")])
        print(i[i.index(",")+2:len(i)-1])
        if(prev1==None):
            prev1=float(i[1:i.index(",")])
            prev2=float(i[i.index(",")+2:len(i)-1])
            continue
        arrlong.append(float(i[1:i.index(",")]))
        arrlat.append(float(i[i.index(",")+2:len(i)-1]))
        prev1=float(i[1:i.index(",")])
        prev2=float(i[i.index(",")+2:len(i)-1])
    plt.plot(arrlong,arrlat,linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig("mapa-de-called-con-longitud.png")
#Separate code dont mind this ///////////////////////////////////////////////////////////////////////
def create_map(response):
   # use the response
   mls = response.json()['features'][0]['geometry']['coordinates']
   points = [(i[1], i[0]) for i in mls[0]]
   m = folium.Map()
   # add marker for the start and ending points
   for point in [points[0], points[-1]]:
      folium.Marker(point).add_to(m)
   # add the lines
   folium.PolyLine(points, weight=5, opacity=1).add_to(m)
   # create optimal zoom
   df = pd.DataFrame(mls[0]).rename(columns={0:'Lon', 1:'Lat'})[['Lat', 'Lon']]
   sw = df[['Lat', 'Lon']].min().values.tolist()
   ne = df[['Lat', 'Lon']].max().values.tolist()
   m.fit_bounds([sw, ne])
   return m
#more code separation ///////////////////////////////////////////////////////////////////////////////////
def bfs(Origen,Destino,both,src,destino):
    pile=deque()
    pile.append(src)
    visits=[]
    while len(pile)>0:
        here=pile.pop()
        if here not in visits:
            visits.append(here)
            f=Origen.index(here)
            c=0
            if(here==destino):
                break
            while(Origen[f]==Origen[f+c]):
                if(f+c>len(Destino)):
                    break
                pile.append(Destino[f+c])
                if(f+c==len(Origen)):
                    break
                c+=1
            for i in range(0,len(Destino)):
                if((Destino[i]==here)and(both[i]==True)):
                    pile.append(Origen[i])
    return visits
df = pd.read_csv('calles_de_medellin_con_acoso.csv',sep=';')
df.drop("Unnamed: 0",axis =1,inplace=True )
s=0
#27661
c=0
ant=None
adentro=False
origenes2=[None]*68749
origenes=[None]*68749
origenes3=[None]*68749
origenes4=[None]*68749
origenes5=[None]*68749
d=0
arbol=dict()
for i in df.iterrows():
    origen=i[1][1]
    nombre= i[1][0]
    destino=i[1][2]
    length=i[1][3]
    riesgo=i[1][5]
    oneway=i[1][4]
    origenes2[d]=destino
    origenes[d]=origen
    origenes3[d]=s
    origenes4[d]=oneway
    origenes5[d]=length
    if(d==0):
        print(destino+"aqui")
    if(arbol.get(str(origenes[d]))==None):
        arbol[str(origenes[d])]=s
        s+=1
    if(c!=0):
        if(str(origen)==str(ant)):
            adentro=True
    ant=origen
    if(adentro==False):
        c+=1
    d+=1
    adentro=False
origenesSupletorio=[]
for i in range(0,len(origenes2)):
    if(arbol.get(str(origenes2[i]))==None):
        origenes.append(origenes2[i])
        arbol[str(origenes2[i])]=s
        print(str(origenes2[i]))
        origenesSupletorio.append(s)
        origenes3.append(int(origenes3[len(origenes3)-1]+1))
        s+=1
    else:
        origenesSupletorio.append(arbol[str(origenes2[i])])
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
head=Graph(s)
#27671
for i in range(0,len(origenes2)):
    head.addEdge(int(origenes3[i]),int(arbol.get(str(origenes2[i]))),float(origenes5[i]),bool(origenes4[i]))
print(origenes[origenes3.index(0)])
print(origenes[origenes3.index(2444)])
head.dijkstra(0,2444,origenes,origenes3,origenesSupletorio,origenes4,origenes5)