from src.TaskModel.Task import Task
from src.ScheduleModel.PointTimetable import PointTimetable
from src.ScheduleModel.EdgeTimetable import EdgeTimetable

class Schedule:
    def __init__(self, instance):
        self.paths = {}
        self.pointTimetable = {}
        self.edgeTimetable = {} 
    def getObjective(self, instance):
        total_cost = []
        sum = 0
        for tid, path in self.paths.items():
            task = instance.tasks[tid]
            runway = instance.Map.Runways[task.runwayid]
            times = path.getTotalTime(task, runway)
            # TODO
            total_cost.append(times)
            sum += times
        return total_cost, sum

    def addPath(self, task, path):
        self.paths[task.taskid] = path
        self.updatePointTimetable(path)
        self.updateEdgeTimetable(path)

    def updatePointTimetable(self, path):
        pointlist = path.pointlist
        for pid in pointlist:
            if pid in self.pointTimetable.keys():
                ptimetable = self.pointTimetable[pid]
                ptimetable.ocuppyByPath(path)
            else:
                ptimetable = PointTimetable(pid)
                ptimetable.ocuppyByPath(path)
                self.pointTimetable[pid] = ptimetable

    def updateEdgeTimetable(self, path):
        pointlist = path.pointlist
        for i in range(len(pointlist) - 1):
            pid1 = pointlist[i]
            pid2 = pointlist[i + 1]
            edgeid = str(pid1) + "," + str(pid2)
            if edgeid in self.edgeTimetable.keys():
                etimetable = self.edgeTimetable[edgeid]
                etimetable.eOcuppyByPath(path)
            else:
                etimetable = EdgeTimetable(pid1, pid2)
                etimetable.eOcuppyByPath(path)
                self.edgeTimetable[edgeid] = etimetable
