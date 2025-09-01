import numpy as np
import pandas as pd
import datetime
import random
import time
import os

# The camera positions are divided into multiple zones and allocated according to each zone
SP = {}
data = [] # Used for storing test data
def loadStandPointFile(pointFile):
    pointlist = pd.read_excel(pointFile)
    ID = pointlist['id']
    ID = list(np.array(ID))
    Num = pointlist['num']
    Num = list(np.array(Num))
    data = ID
    for id, num in zip(ID, Num):
        SP[id] = num

def fun1(Tin, Toff, i):
    map = 'MAP1'
    tid = 1 
    aircrafttype = [1, 2] 
    atypepossible = [0.9, 0.1]
    tasktype = [0, 1]
    ttypepossible = [0.5, 0.5]  # The probabilities of randomly generated arrivals and departures each account for half
    inBlockNum = 150 #[50, 100, 150]
    offBlockNum = 150 # [50, 100, 150]
    queueLen = [5, 10, 15]  # queue size
    qpossible = [0.3, 0.4, 0.3]

    timeinterval = 2 * 60 * 60 # [2 * 60 * 60, 4 * 60 * 60, 6 * 60 * 60]  # 2h、4h、6h
    d = datetime.datetime.strptime('2024/01/01 9:0:0', '%Y/%m/%d %H:%M:%S')
    starttime = time.mktime(d.utctimetuple())  # The departure moment is converted into a timestamp

    inNum = Tin  # The number of inbound flights
    offNum = Toff  # The number of departing flights
    for k in range(inNum + offNum):
        tidstr = 'T' + str(tid)
        tid += 1
        aType = random.choices(aircrafttype, atypepossible)[0]
        standpoint = random.choices(list(SP.keys()))[0]
        curnum = SP[standpoint]
        runwaypoint = ''
        rid = '' # The runway id is determined by the runway entrance
        if int(tType) == 1:
            # Departure Mission
			# Allocate runway entrances based on the aircraft position area code
            if int(curnum) < 7:
                runwaypoint = random.choices(['P_565_1', 'P_1297_1'])[0]
                if runwaypoint == 'P_565_1':
                    rid = '15'
                else:
                    rid = '33'
            elif int(curnum) < 13:
                runwaypoint = random.choices(['P_572_1', 'P_1375_1'])[0]
                if runwaypoint == 'P_572_1':
                    rid = '15'
                else:
                    rid = '33'
            else:
                runwaypoint = 'P_1297_1'
                rid = '33'
        else:
            # Port Entry Mission
			# Allocate runway exits based on aircraft models and aircraft location area codes
            if int(curnum) < 7:
                if int(aType) == 1:
                    runwaypoint = random.choices(['P_1163', 'P_1541'])[0]
                    if runwaypoint == 'P_1163':
                        rid = '33'
                    else:
                        rid = '15'
                else:
                    runwaypoint = random.choices(['P_994', 'P_1544'])[0]
                    if runwaypoint == 'P_994':
                        rid = '33'
                    else:
                        rid = '15'
            elif int(curnum) < 13:
                if int(aType) == 1:
                    runwaypoint = random.choices(['P_912', 'P_1532'])[0]
                    if runwaypoint == 'P_912':
                        rid = '33'
                    else:
                        rid = '15'
                else:
                    runwaypoint = random.choices(['P_998', 'P_1227'])[0]
                    if runwaypoint == 'P_998':
                        rid = '33'
                    else:
                        rid = '15'
            else:
                if int(aType) == 1:
                    runwaypoint = 'P_1541'
                else:
                    runwaypoint = 'P_1544'
                rid = '15'

        # Randomly generate task time intervals
        taskinterval = random.randint(0, timeinterval)



        nextstarttime = starttime + taskinterval
        data.append((tidstr, aType, tType, standpoint, rid, int(nextstarttime), runwaypoint))
    dataed = sorted(data, key=lambda x : x[5])
    f = pd.DataFrame(dataed, columns=['aid', 'aircrafttype', 'tasktype', 'standpointid', 'runwayid', 'startime', 'runwaypoint'])
    f.to_csv('testdata_300_' + str(i) + '.txt', sep='\t', index=False)


currentpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pointFile = currentpath + '\data\standpoint.xlsx'
loadStandPointFile(pointFile)
Tin =150
Toff = 150
for i in range(10):
    fun1(Tin, Toff, i + 1)






