import numpy as np
import pandas as pd
from src.ResourceModel.Point import Point
from src.ResourceModel.Runway import Runway
from src.ScheduleModel.Path import Path
import networkx as nx
import math

edgepercent = 0


class MapInstance:
    def __init__(self, pointFile, edgeFile, runwayFile):
        self.pointFile = pointFile
        self.edgeFile = edgeFile
        self.runwayFile = runwayFile
        self.Points = {} 
        self.Runways = {}  
        self.G = nx.DiGraph()
        self.loadMapFromFile(pointFile, edgeFile, runwayFile)

    def getEdgeWeightType(self, pid1, pid2):
        point1 = self.Points[pid1]
        [weight, edgetype] = point1.adPoints[pid2]
        return [weight, edgetype]

    def computeAdEdges100(self): 
        for center in self.Points.keys():
            for pid1 in self.Points.keys():
                for pid2 in self.Points.keys():
                    if pid1 == pid2:
                        continue
                    if self.G.has_edge(pid1, pid2):
                        if self.isInRange(center, pid1, pid2, edgepercent):
                            cpoint = self.Points[center]
                            cpoint.addNearEdges(pid1, pid2)

    def isInRange(self, center, p1, p2, percent):
        point = self.Points[center]
        point1 = self.Points[p1]
        point2 = self.Points[p2]
        x0 = point.X
        y0 = point.Y
        x1 = point1.X
        y1 = point1.Y
        x2 = point2.X
        y2 = point2.Y
        radius = 100

        dx = x2 - x1
        dy = y2 - y1

        inp = []
        discriminant = (2 * dx * (x1 - x0) + 2 * dy * (y1 - y0)) ** 2 - 4 * (dx ** 2 + dy ** 2) * (
                    (x1 - x0) ** 2 + (y1 - y0) ** 2 - radius ** 2)

        def distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        distance_p1 = distance(x1, y1, x0, y0)
        distance_p2 = distance(x2, y2, x0, y0)
        distance_segment = distance(x1, y1, x2, y2)
        if distance_p1 <= radius and distance_p2 <= radius:
            return True
        if distance_p1 > radius and distance_p2 > radius:
            if (discriminant < 0):
                return False
            t1 = (-2 * dx * (x1 - x0) - 2 * dy * (y1 - y0) + math.sqrt(discriminant)) / (2 * (dx ** 2 + dy ** 2))
            t2 = (-2 * dx * (x1 - x0) - 2 * dy * (y1 - y0) - math.sqrt(discriminant)) / (2 * (dx ** 2 + dy ** 2))

            if 0 <= t1 <= 1:
                inp.append((x1 + t1 * dx, y1 + t1 * dy))
            if 0 <= t2 <= 1:
                inp.append((x1 + t2 * dx, y1 + t2 * dy))
            if len(inp) == 2:
                distance_in_circle = distance(inp[0][0], inp[0][1], inp[1][0], inp[1][1])
                portion = distance_in_circle / distance_segment
            else:
                return False
        elif distance_p1 <= radius and distance_p2 > radius:
            t1 = (-2 * dx * (x1 - x0) - 2 * dy * (y1 - y0) + math.sqrt(discriminant)) / (2 * (dx ** 2 + dy ** 2))
            t2 = (-2 * dx * (x1 - x0) - 2 * dy * (y1 - y0) - math.sqrt(discriminant)) / (2 * (dx ** 2 + dy ** 2))

            if 0 <= t1 <= 1:
                inp.append((x1 + t1 * dx, y1 + t1 * dy))
            if 0 <= t2 <= 1:
                inp.append((x1 + t2 * dx, y1 + t2 * dy))
            if (len(inp) == 0):
                return False
            distance_in_circle = distance(inp[0][0], inp[0][1], x1, y1)
            portion = distance_in_circle / distance_segment
        elif distance_p1 > radius and distance_p2 <= radius:
            t1 = (-2 * dx * (x1 - x0) - 2 * dy * (y1 - y0) + math.sqrt(discriminant)) / (2 * (dx ** 2 + dy ** 2))
            t2 = (-2 * dx * (x1 - x0) - 2 * dy * (y1 - y0) - math.sqrt(discriminant)) / (2 * (dx ** 2 + dy ** 2))

            if 0 <= t1 <= 1:
                inp.append((x1 + t1 * dx, y1 + t1 * dy))
            if 0 <= t2 <= 1:
                inp.append((x1 + t2 * dx, y1 + t2 * dy))
            if (len(inp) == 0):
                return False
            distance_in_circle = distance(inp[0][0], inp[0][1], x2, y2)
            portion = distance_in_circle / distance_segment
        if (portion > percent):
            return True
        return False

    def loadMapFromFile(self, pointFile, edgeFile, runwayFile):
        pointlist = pd.read_excel(pointFile)  
        ID = pointlist['id']
        ID = list(np.array(ID))
        self.G.add_nodes_from(ID)
        X = pointlist['X'] 
        Y = pointlist['Y'] 
        X = list(np.array(X))
        Y = list(np.array(Y))
        for i in range(len(ID)):
            point = Point(ID[i], X[i], Y[i])
            self.Points[ID[i]] = point
        self.computeAdEdges100()
        edgelist = pd.read_excel(edgeFile)
        Alist = edgelist['A']
        Alist = list(np.array(Alist))
        Blist = edgelist['B']
        Blist = list(np.array(Blist))
        wlist = edgelist['W']
        wlist = list(np.array(wlist))
        tlist = edgelist['T']
        tlist = list(np.array(tlist))
        edges = []
        for i in range(len(Alist)):
            aid = Alist[i]
            bid = Blist[i]
            weight = wlist[i]
            edgetype = tlist[i]
            aPoint = self.Points[aid]
            aPoint.addAdPoint(bid, weight, edgetype)
            edges.append((aid, bid, weight))
        self.G.add_weighted_edges_from(edges) 
        rwfile = open(runwayFile)
        lines = rwfile.readlines()
        lines = [item.rstrip() for item in lines]
        for item in lines:
            temp = item.split('\t')
            rid = temp[0]
            plist = temp[1]
            runway = Runway(rid)
            plist = plist.split('|')
            enterlist = plist[0].split(',')
            for item in enterlist:
                item = item.split('-')
                runway.addEnterPoints(item[0], int(item[1]))
            exitlist = plist[1].split(',')
            for pid in exitlist:
                runway.addExitPoints(pid)
            self.Runways[rid] = runway
        rwfile.close()

    def computePath(self, G, method, start, end):
        # dijkstra
        if method == 1:
            return nx.dijkstra_path(G, source=start, target=end, weight='weight')
        # bellman-ford
        if method == 2:
            return nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
        # A*
        if method == 3:
            return nx.astar_path(G, source=start, target=end, weight='weight')

    def choosePath(self, G, start, end, task, instance):
        paths = nx.shortest_simple_paths(G, source=start, target=end, weight='weight')
        n = 10 
        p = []
        for e in paths:
            if n == 0:
                break
            p.append(e)
            n -= 1
        bestPath = p[0]
        minTime = float('inf')
        for e in p:
            path = Path(task.taskid, task.tasktype)
            path.pointlist = e
            path.computeInitialTime(instance)
            runway = instance.Map.Runways[task.runwayid]
            t = path.getTotalTime(task, runway)

            if (t < minTime):
                minTime = t
                bestPath = e
        return bestPath

    def getMap(self,pointFile, edgeFile):
        df = pd.read_excel(pointFile)
        ID = df['id']
        ID = np.array(ID)
        X = df['X'] 
        Y = df['Y'] 
        X = list(np.array(X))
        Y = list(np.array(Y))
        pos = [] 
        edges = [] 
        for e1, e2 in zip(X, Y):
            pos.append((e1, e2))
        df1 = pd.read_excel(edgeFile)
        A = df1['A']
        B = df1['B']
        W = df1['W']
        W = list(np.array(W))
        A = list(np.array(A))
        B = list(np.array(B))

        for e1, e2, e3 in zip(A, B, W):
            edges.append((e1, e2, e3))

        G = nx.DiGraph()
        G.add_nodes_from(ID)
        G.add_weighted_edges_from(edges)
        return G
