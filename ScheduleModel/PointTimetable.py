import sys

infinity = sys.float_info.max

class PointTimetable:
    def __init__(self, pid):
        self.pid = str(pid)
        self.stack = []
        self.intervals = []

    def ocuppyByPath(self, path):
        starttime = path.time[self.pid]
        endtime = path.time[self.pid] + path.waitingTime[self.pid] + path.turningTime[self.pid]
        return self.addInterval(starttime, endtime)

    def addInterval(self, start, end):
        if len(self.intervals) == 0:
            self.intervals.append((start, end))
            self.stack.append((start, end))
            return True
        for i in range(len(self.intervals)):
            (eachstart, eachend) = self.intervals[i]
            if self.isOverlap(start, eachstart, end, eachend):
                return False
            if start < eachstart:
                self.intervals.insert(i, (start, end))
                self.stack.append((start, end))
                return True
            if i == len(self.intervals) - 1:
                self.intervals.append((start, end))
                self.stack.append((start, end))
                return True

    def isOverlap(self, s1, s2, e1, e2):
        if max(s1, s2) <= min(e1, e2):
            return True
        return False
