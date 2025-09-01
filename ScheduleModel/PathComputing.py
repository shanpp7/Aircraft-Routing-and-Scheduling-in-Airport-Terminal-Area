from src.TaskModel.aircraftInfo import *
from src.ScheduleModel.PointTimetable import PointTimetable
from src.ScheduleModel.EdgeTimetable import EdgeTimetable
import sys
from src.ScheduleModel.Path import Path
import copy
import numpy as np

infinity = sys.float_info.max

def ComputeNoConflictPathTask(pointlist, task, schedule, instance):
    path = Path(task.taskid, task.tasktype)
    path.setPath(pointlist)
    leavingtimelist = {}
    arrivingtimelist = {}
    firstpoint = path.pointlist[0]
    firstarrivinginterval = [(task.starttime, infinity)]
    firstleavinginterval = computeFeasibleFromStartLeavingTime(task.taskid, firstpoint, instance, schedule)
    arrivingtimelist[firstpoint] = firstarrivinginterval
    leavingtimelist[firstpoint] = firstleavinginterval

    for i in range(1, len(pointlist)):
        pid1 = pointlist[i - 1]
        pid2 = pointlist[i]
        leavingInteval = leavingtimelist[pid1]
        arrInterval = computeFeasibleArrivingTimeFromOnePoint2Another(pid1, pid2, leavingInteval, instance, schedule)
        arrivingtimelist[pid2] = arrInterval
        if i < len(pointlist) - 1:
            pid3 = pointlist[i + 1]
            leavinginterval = computeFeasibleDepartTime(pid1, pid2, pid3, arrInterval, instance)
            leavingtimelist[pid2] = leavinginterval
        else:  
            leavingtimelist[pid2] = arrInterval
    for i in range(len(pointlist) - 1, -1, -1):
        pid = pointlist[i]
        point = instance.Map.Points[pid]
        if i == len(pointlist) - 1:
            (spstart, spend) = arrivingtimelist[pid][0]
            path.time[pid] = spstart
            path.turningTime[pid] = 0
            path.waitingTime[pid] = 0
        else:
            nextpid = pointlist[i + 1]
            nextpoint = instance.Map.Points[nextpid]
            nextarrivingtime = path.time[nextpid]
            (lenn, edgetype) = point.adPoints[nextpid]
            currentleavingtime = nextarrivingtime - lenn
            if i != 0:
                prepid = pointlist[i - 1]
                prepoint = instance.Map.Points[prepid]
                turningtime = TurningTime(prepoint, point, nextpoint)
                path.turningTime[pid] = turningtime
                path.time[pid] = currentleavingtime - turningtime
                path.waitingTime[pid] = 0
            else: 
                path.time[pid] = task.starttime
                path.turningTime[pid] = 0 
                path.waitingTime[pid] = currentleavingtime - task.starttime
    return path

def ComputeNoConflictPathTaskDRL(pointlist, task, schedule, instance):
    path = Path(task.taskid, task.tasktype)
    path.setPath(pointlist)
    leavingtimelist = {}
    arrivingtimelist = {}
    firstpoint = path.pointlist[0]
    randominterval = np.random.randint(200)
    firstarrivinginterval = [(task.starttime + randominterval, infinity)]
    firstleavinginterval = computeFeasibleFromStartLeavingTime(task.taskid, firstpoint, instance, schedule)
    arrivingtimelist[firstpoint] = firstarrivinginterval
    leavingtimelist[firstpoint] = firstleavinginterval

    for i in range(1, len(pointlist)):
        pid1 = pointlist[i - 1]
        pid2 = pointlist[i]
        leavingInteval = leavingtimelist[pid1]
        arrInterval = computeFeasibleArrivingTimeFromOnePoint2Another(pid1, pid2, leavingInteval, instance, schedule)
        arrivingtimelist[pid2] = arrInterval
        if i < len(pointlist) - 1:
            pid3 = pointlist[i + 1]
            leavinginterval = computeFeasibleDepartTime(pid1, pid2, pid3, arrInterval, instance)
            leavingtimelist[pid2] = leavinginterval
        else: 
            leavingtimelist[pid2] = arrInterval
    for i in range(len(pointlist) - 1, -1, -1):
        pid = pointlist[i]
        point = instance.Map.Points[pid]
        if i == len(pointlist) - 1:
            (spstart, spend) = arrivingtimelist[pid][0]
            path.time[pid] = spstart
            path.turningTime[pid] = 0
            path.waitingTime[pid] = 0
        else:
            nextpid = pointlist[i + 1]
            nextpoint = instance.Map.Points[nextpid]
            nextarrivingtime = path.time[nextpid]
            (lenn, edgetype) = point.adPoints[nextpid]
            currentleavingtime = nextarrivingtime - lenn
            if i != 0:
                prepid = pointlist[i - 1]
                prepoint = instance.Map.Points[prepid]
                turningtime = TurningTime(prepoint, point, nextpoint)
                path.turningTime[pid] = turningtime
                path.time[pid] = currentleavingtime - turningtime
                path.waitingTime[pid] = 0
            else:
                path.time[pid] = task.starttime
                path.turningTime[pid] = 0 
                path.waitingTime[pid] = currentleavingtime - task.starttime
    return path


def computeFeasibleArrivingTimeFromOnePoint2Another(pid1, pid2, leavingInteval, instance, schedule):
    point1 = instance.Map.Points[pid1]
    (length, edgetype) = point1.adPoints[pid2]
    runtime = length
    intervalwithruntime = addRuntime(leavingInteval, runtime)
    if pid2 in schedule.pointTimetable:
        timetableofP2 = schedule.pointTimetable[pid2]
    else:
        timetableofP2 = None
    restrictedIntervalsofP2 = getRestrictedZone(pid1, timetableofP2, instance)
    feasiblearrivingintervals = intervalsMinus(intervalwithruntime, restrictedIntervalsofP2)
    point2 = instance.Map.Points[pid2]
    for eid in point2.adEdges100:
        if eid in schedule.edgeTimetable:
            etimetable = schedule.edgeTimetable[eid]
            feasiblearrivingintervals = intervalsMinus(feasiblearrivingintervals, etimetable.intervals)
    return feasiblearrivingintervals


def computeFeasibleDepartTime(pid1, pid2, pid3, arrivingInterval, instance):
    point1 = instance.Map.Points[pid1]
    point2 = instance.Map.Points[pid2]
    point3 = instance.Map.Points[pid3]
    if TurningTime(point1, point2, point3) == turningTime: 
        return removeTuringTime(arrivingInterval)
    return copy.deepcopy(arrivingInterval)


def computeFeasibleFromStartLeavingTime(tid, pid, instance, schedule):
    task = instance.tasks[tid]
    starttime = task.starttime
    feasiblearrivingintervals = [(starttime, infinity)]
    departime = [(starttime, infinity)]
    point = instance.Map.Points[pid]
    for eid in point.adEdges100: 
        if eid in schedule.edgeTimetable:
            etimetable = schedule.edgeTimetable[eid]
            feasiblearrivingintervals = intervalsMinus(departime, etimetable.intervals)
    return feasiblearrivingintervals


def getRestrictedZone(frompid, timetable, instance):
    restrictedIntervals = []
    if timetable is None:
        return restrictedIntervals
    frompoint = instance.Map.Points[frompid]
    (weight, edgetype) = frompoint.adPoints[timetable.pid]
    restrictedTime = getRunningTime(int(edgetype), saveDistance)  # saveDisatance=100
    for (start, end) in timetable.intervals:
        restrictedIntervals.append((start - restrictedTime, end + restrictedTime))
    return restrictedIntervals


def getRunningTime(edgetype, length):
    runtime = 0
    if edgetype == 0:
        runtime = int(length * 36 / 900)
    elif edgetype == 1 or edgetype == 2:
        runtime = int(length * 36 / 500)
    elif edgetype == 3:
        runtime = int(length * 36 / 270)
    else:
        runtime = int(length * 36 / 50)
    return runtime


def addRuntime(intervals, runtime):
    result = []
    for (start, end) in intervals:
        if end == infinity:
            result.append((int(start) + int(runtime), int(end)))
        else:
            result.append((int(start) + int(runtime), int(end) + int(runtime)))
    return result


def removeTuringTime(intervals):
    result = []
    for (start, end) in intervals:
        if end - start > turningTime:
            result.append((start + turningTime, end))
    return result


def intervalsMinus(intervals1, intervals2):  # intervals1-intervals2
    result = []
    for (s1, e1) in intervals1:
        start1 = s1
        end1 = e1
        for (start2, end2) in intervals2:
            if isOverlap(start1, end1, start2, end2):
                # case1
                if start1 >= start2 and end1 <= end2:
                    start1 = -1
                    end1 = -1
                    break
                # case2
                elif start1 < start2 and end1 > end2:
                    result.append((start1, start2))
                    start1 = end2
                    end1 = end1
                # case3
                elif start1 < start2 and end1 <= end2:
                    result.append((start1, start2))
                    start1 = -1
                    end1 = -1
                    break
                # case4 
                elif start1 >= start2 and end1 > end2:
                    start1 = end2
                    end1 = end1
            else: 
                if end1 <= start2:
                    # result.append((start1,end1))
                    break
        if start1 != -1 and end1 != -1:
            result.append((start1, end1))
    return result


def isOverlap(start1, end1, start2, end2):
    if max(start1, start2) > min(end1, end2):
        return False
    return True


INF = float('inf')

def floyd_warshall(instance, start, end):
    map = instance.Map
    G = map.G
    plist = list(map.Points.keys())
    V = len(plist)
    dist = [[INF] * V for _ in range(V)]
    edge = [[-1] * V for _ in range(V)]
    next_node = [[-1] * V for _ in range(V)]

    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            p1 = plist[i]
            p2 = plist[j]
            if G.has_edge(p1, p2):
                point1 = map.Points[p1]
                [weight, edgetype] = point1.adPoints[p2]
                dist[i][j] = weight
                edge[i][j] = weight
                next_node[i][j] = j



    for i in range(V):
        dist[i][i] = 0
        next_node[i][i] = i
    for k in range(V):
        print(k)
        print(plist[k])
        for i in range(V):
            if i == k:
                continue
            if edge[i][k] == -1:
                continue
            for j in range(V):
                if j == i or j == k:
                    continue
                if edge[k][j] == -1 or edge[i][j] == -1:
                    continue
                point1 = map.Points[plist[i]]
                point2 = map.Points[plist[k]]
                point3 = map.Points[plist[j]]
                if dist[i][j] > dist[i][k] + dist[k][j]:  # + TurningTime(point1, point2, point3):
                    dist[i][j] = dist[i][k] + dist[k][j]  # + TurningTime(point1, point2, point3)
                    next_node[i][j] = k
    u = plist.index(start)
    v = plist.index(end)

    path = [start]
    while u != v:
        u = next_node[u][v]
        print(u)
        path.append(plist[u])
        print(path)
    return dist[plist.index(start)][plist.index(end)], path


def ComputeNoConflictPath(pointlist, task, schedule, instance):
    path = Path(task.taskid, task.tasktype)
    path.setPath(pointlist)
    leavingtimelist = {}
    arrivingtimelist = {}
    standpoint = pointlist[0]
    firstarrivinginterval = [(task.starttime, infinity)]
    firstleavinginterval = computeFeasibleFromStartLeavingTime(task.taskid, standpoint, instance, schedule)
    arrivingtimelist[standpoint] = firstarrivinginterval
    leavingtimelist[standpoint] = firstleavinginterval
    path.time[standpoint] = task.starttime 
    path.turningTime[standpoint] = 0  
    path.waitingTime[standpoint] = firstleavinginterval[0][0] - task.starttime

    for i in range(1, len(pointlist)):
        pid1 = pointlist[i - 1]
        pid2 = pointlist[i]
        leavingInteval = leavingtimelist[pid1]
        arrInterval = computeFeasibleArrivingTimeFromOnePoint2Another(pid1, pid2, leavingInteval, instance, schedule)

        arrivingtimelist[pid2] = arrInterval
        if i < len(pointlist) - 1:
            pid3 = pointlist[i + 1]
            leavinginterval = computeFeasibleDepartTime(pid1, pid2, pid3, arrInterval, instance)
            leavingtimelist[pid2] = leavinginterval

            prepoint = instance.Map.Points[pid1]
            point = instance.Map.Points[pid2]
            nextpoint = instance.Map.Points[pid3]
            turningtime1 = TurningTime(prepoint, point, nextpoint)

            path.time[pid2] = arrInterval[0][0]
            path.turningTime[pid2] = turningtime1


            path.waitingTime[pid2] = leavinginterval[0][0] - arrInterval[0][0]

        else:  
            leavingtimelist[pid2] = arrInterval
            path.time[pid2] = arrInterval[0][0]
            path.turningTime[pid2] = 0  
            path.waitingTime[pid2] = 0
    return path

def ComputeNoConflictPathDRL(pointlist, task, schedule, instance):
    path = Path(task.taskid, task.tasktype)
    path.setPath(pointlist)
    leavingtimelist = {}
    arrivingtimelist = {}
    standpoint = pointlist[0]
    randominterval = np.random.randint(50)
    firstarrivinginterval = [(task.starttime, infinity)]
    firstleavinginterval = computeFeasibleFromStartLeavingTime(task.taskid, standpoint, instance, schedule)
    arrivingtimelist[standpoint] = firstarrivinginterval
    leavingtimelist[standpoint] = firstleavinginterval
    path.time[standpoint] = task.starttime 
    path.turningTime[standpoint] = 0  
    path.waitingTime[standpoint] = firstleavinginterval[0][0] - task.starttime + randominterval

    for i in range(1, len(pointlist)):
        pid1 = pointlist[i - 1]
        pid2 = pointlist[i]
        leavingInteval = leavingtimelist[pid1]
        arrInterval = computeFeasibleArrivingTimeFromOnePoint2Another(pid1, pid2, leavingInteval, instance, schedule)

        arrivingtimelist[pid2] = arrInterval
        if i < len(pointlist) - 1:
            pid3 = pointlist[i + 1]
            leavinginterval = computeFeasibleDepartTime(pid1, pid2, pid3, arrInterval, instance)
            leavingtimelist[pid2] = leavinginterval

            prepoint = instance.Map.Points[pid1]
            point = instance.Map.Points[pid2]
            nextpoint = instance.Map.Points[pid3]
            turningtime1 = TurningTime(prepoint, point, nextpoint)

            path.time[pid2] = arrInterval[0][0]
            path.turningTime[pid2] = turningtime1


            path.waitingTime[pid2] = leavinginterval[0][0] - arrInterval[0][0]

        else:  
            leavingtimelist[pid2] = arrInterval
            path.time[pid2] = arrInterval[0][0]
            path.turningTime[pid2] = 0
            path.waitingTime[pid2] = 0
    return path

def computeArrivingWaitingTime(instance,schedule):
    runwaypointQueuest0 = []
    runwaypointQueuest1 = []
    arrivetime = []
    waitingtime = []
    paths = schedule.paths
    tasks = instance.tasks
    for tid, task in tasks.items():
        if task.tasktype == 1:
            path = paths[tid]
            if path.pointlist[-1] in r15_33:
                arrivetime.append((path.time[path.pointlist[-1]], tid, 0))
            else:
                arrivetime.append((path.time[path.pointlist[-1]], tid, 1))
    arrivetimed = sorted(arrivetime, key=lambda x: x[0], reverse=False)
    for time,tid,type in arrivetimed:
        if type == 0:
            runwaypointQueuest0.append((tid,time))
        else:
            runwaypointQueuest1.append((tid,time))
    if len(runwaypointQueuest0) != 0:
        waitingtime.append((runwaypointQueuest0[0][0], 0))
        for i in range(1, len(runwaypointQueuest0)):
            preatype = tasks[runwaypointQueuest0[i - 1][0]].aircrafttype
            curatype = tasks[runwaypointQueuest0[i][0]].aircrafttype
            predistance = timeDistanceOnRunWay(int(preatype), int(curatype))
            if runwaypointQueuest0[i - 1][1] + predistance > runwaypointQueuest0[i][1]:
                waitingtime.append((runwaypointQueuest0[i][0],
                                    runwaypointQueuest0[i - 1][1] + predistance - runwaypointQueuest0[i][1]))
            else:
                waitingtime.append((runwaypointQueuest0[i][0], 0))
    if len(runwaypointQueuest1) != 0:

        waitingtime.append((runwaypointQueuest1[0][0], 0))
        for i in range(1, len(runwaypointQueuest1)):
            preatype = tasks[runwaypointQueuest1[i - 1][0]].aircrafttype
            curatype = tasks[runwaypointQueuest1[i][0]].aircrafttype
            predistance = timeDistanceOnRunWay(int(preatype), int(curatype))
            if runwaypointQueuest1[i - 1][1] + predistance > runwaypointQueuest1[i][1]:
                waitingtime.append((runwaypointQueuest1[i][0], runwaypointQueuest1[i - 1][1] + predistance - runwaypointQueuest1[i][1]))
            else:
                waitingtime.append((runwaypointQueuest1[i][0], 0))
    for tid, time in waitingtime:
        path = paths[tid]
        path.waitingTime[path.pointlist[-1]] = time
    return schedule
