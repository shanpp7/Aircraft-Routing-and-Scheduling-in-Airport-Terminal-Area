
class Point:
    def __init__(self,pid,x,y):
        self.pid = str(pid)
        self.X = x
        self.Y = y
        self.adPoints = {}
        self.adEdges100 = set()


    def addAdPoint(self,adpid,weight,edgetype):
        self.adPoints[adpid] = (weight,edgetype)

    def addNearEdges(self,pid1,pid2):
        eid = str(pid1)+","+str(pid2)
        self.adEdges100.add(eid)