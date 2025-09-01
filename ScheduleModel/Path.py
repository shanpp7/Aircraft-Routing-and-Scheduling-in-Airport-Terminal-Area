from src.TaskModel.aircraftInfo import *

class Path:
    def __init__(self, aid, tasktype):
        self.taskid = str(aid) 
        self.tasktype = int(tasktype)
        self.pointlist = [] 
        self.time = {} 
        self.waitingTime = {} 
        self.turningTime = {} 

    def setPath(self, pointlist):
        self.pointlist = pointlist

    def getExitPoint(self):
        return self.pointlist[-1]

    def getExitTime(self):
        exitpoint = self.pointlist[-1]
        exittime = self.time[exitpoint]
        return exittime

    def getStartTime(self):
        startpoint = self.pointlist[0]
        return self.time[startpoint]

    def getEndTime(self):
        endpoint = self.pointlist[-1]
        endtime = self.time[endpoint] + self.waitingTime[endpoint]
        return endtime

    def getTotalTime(self, task, runway):
        totaltime = self.getEndTime() - self.getStartTime()
        if task.tasktype == 0:
            enterpoint = self.pointlist[0]
            totaltime = totaltime + runway.getEnterTime(enterpoint)
        else:
            totaltime = totaltime + runway.getExitTime(task.aircrafttype)
        return totaltime

    def computeInitialTime(self, instance):
        firstpid = self.pointlist[0]
        task = instance.tasks[self.taskid]
        self.time[firstpid] = int(task.starttime)
        self.waitingTime[firstpid] = 0
        self.turningTime[firstpid] = 0
        for i in range(1, len(self.pointlist)):
            lastpid = self.pointlist[i - 1]
            curpid = self.pointlist[i]
            [weight, edgetype] = instance.Map.getEdgeWeightType(lastpid, curpid)
            runtime = weight
            self.waitingTime[curpid] = 0
            if i == len(self.pointlist) - 1:
                self.turningTime[curpid] = 0
            else:
                nexpid = self.pointlist[i + 1]
                point1 = instance.Map.Points[lastpid]
                point2 = instance.Map.Points[curpid]
                point3 = instance.Map.Points[nexpid]
                self.turningTime[curpid] = TurningTime(point1, point2, point3)
            self.time[curpid] = self.time[lastpid] + runtime + self.turningTime[curpid]

