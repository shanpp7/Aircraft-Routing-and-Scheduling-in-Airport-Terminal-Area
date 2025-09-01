import sys

infinity = sys.float_info.max

class EdgeTimetable:
    def __init__(self, pid1, pid2):
        self.eid = str(pid1) + "," + str(pid2) 
        self.startp = str(pid1)
        self.endp = str(pid2)
        self.intervals = [] 

    def eOcuppyByPath(self, path):
        starttime1 = path.time[self.startp]
        # endtime1 = path.time[pid1]+path.waitingTime[pid1]+path.turningTime[pid1]
        # starttime2 = starttime1 = path.time[pid2]
        endtime2 = path.time[self.endp] + path.waitingTime[self.endp] + path.turningTime[self.endp]
        return self.eAddInterval(starttime1, endtime2)

    def eAddInterval(self, start, end):
        if len(self.intervals) == 0:
            self.intervals.append((start, end))
            return True
        for i in range(len(self.intervals)):
            (eachstart, eachend) = self.intervals[i]
            if self.isOverlap(start, eachstart, end, eachend):
                return False
            if start < eachstart:
                self.intervals.insert(i, (start, end))
                return True
            if i == len(self.intervals) - 1:
                self.intervals.append((start, end))
                return True

    def isOverlap(self, s1, s2, e1, e2):
        if max(s1, s2) <= min(e1, e2):
            return True
        return False
