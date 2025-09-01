class Task:
    def __init__(self, taskid, aircrafttype, tasktype, standpointid, runwayid, runwaypointlist, starttime):
        self.taskid = str(taskid)
        self.aircrafttype = int(aircrafttype) 
        self.tasktype = int(tasktype)  
        self.standpointid = str(standpointid)
        self.runwaypointlist = runwaypointlist 
        self.runwayid = str(runwayid)
        self.starttime = int(starttime)
