
class Runway:
    def __init__(self,rid):
        self.rid = str(rid)
        self.enterPoints = {}
        self.exitPoints = []
    def addEnterPoints(self,pid,enterTime):
        self.enterPoints[pid] = enterTime

    def addExitPoints(self,pid):
        self.exitPoints.append(pid)

    def getExitTime(self,aircrafttype):
        if aircrafttype == 0:
            return 60
        else:
            return 50

    def getEnterTime(self,pid):
        return self.enterPoints[pid]

