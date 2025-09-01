from src.ResourceModel.Map import MapInstance
from src.TaskModel.Task import Task



class TestInstance:
    def __init__(self, pointFile, edgeFile, runwayFile, taskFile):
        self.tasks = {}
        self.Map = MapInstance(pointFile, edgeFile, runwayFile)
        self.loadFromTaskFile(taskFile)

    def loadFromTaskFile(self, taskFile):
        taskfile = open(taskFile)
        lines = taskfile.readlines()
        lines = [item.rstrip() for item in lines]

        lines = lines[1:]

        for item in lines:
            temp = item.split('\t')
            taskid = temp[0]
            aircrafttype = temp[1]
            tasktype = temp[2]
            standpointid = temp[3]
            runwayid = temp[4]
            starttime = temp[5]
            starttime = int(starttime)
            ruwaypoints = temp[6].split(',')
            task = Task(taskid, aircrafttype, tasktype, standpointid, runwayid, ruwaypoints, starttime)
            self.tasks[taskid] = task
        taskfile.close()
