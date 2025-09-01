Airport Ground Traffic Scheduling Optimization

Project Introduction

This is an airport ground traffic scheduling system based on reinforcement learning and multiple optimization algorithms. The system aims to solve the aircraft taxiing path planning problem within airports, optimizing aircraft movement paths between runways, stands, and taxiways through intelligent algorithms to reduce taxiing time and improve airport operational efficiency.

Data Link: https://ieee-dataport.org//documents/30-day-real-instances-aircraft-routing-and-scheduling

Project Structure

```
Aircraft Routing and Scheduling in Airport Terminal Area/
├── Algorithms/           # Algorithm modules
│   ├── ppo.py           # DRL
│   ├── solutionGeneration.py  # Multiple optimization algorithm implementations
│   └── testGen1.py      # Test case generator
├── ResourceModel/        # Resource model
│   ├── Instance.py      # Instance management
│   ├── Map.py           # Map
│   ├── Point.py         # Points
│   └── Runway.py        # Runway
├── ScheduleModel/        # Scheduling model
│   ├── EdgeTimetable.py # Edge timetable
│   ├── Path.py          # Path model
│   ├── PathComputing.py # Path computing
│   ├── PointTimetable.py # Point timetable
│   └── Schedule.py      # Schedule management
├── TaskModel/           # Task model
│   ├── aircraftInfo.py  # Aircraft information
│   └── Task.py          # Task definition
├── data/                # Data files
│   ├── biDirection.xlsx # Bidirectional data
│   ├── edges16.xlsx     # Edge data
│   ├── points.xlsx      # Point data
│   ├── runway.txt       # Runway information
│   ├── standpoint.xlsx  # Stand data
│   ├── new_airport/     # New airport data
│   │   ├── edges.xlsx   # Edge data
│   │   ├── points.xlsx  # Point data
│   │   ├── runway.txt   # Runway information
│   │   └── standpoint.xlsx # Stand data
│   └── tasks/           # Task data
│       └── *.txt        # Task files grouped by time (248 files)
└── README.md            # Project documentation
```

## Core Module Description

### 1. Algorithm Module (Algorithms/)

#### Solution Generation (`solutionGeneration.py`, `ppo.py`)
Implements multiple classic optimization algorithms:
- **Greedy Algorithm (ILS)**
- **Simulated Annealing**
- **Genetic Algorithm**
- **Particle Swarm Optimization (PSO)**
- **DRL**
- **Q-Learning based Particle Swarm Algorithm**

### 2. Resource Model (ResourceModel/)

#### Instance Management (`Instance.py`)
- Manages airport instance data
- Loads map, task, and other configuration information
- Provides unified instance access interface

#### Map Model (`Map.py`)
- Airport map data structure
- Contains geographic information such as points, edges, runways, etc.
- Provides path calculation functionality

#### Point and Runway Models
- `Point.py`: Airport point definitions
- `Runway.py`: Runway information management

### 3. Scheduling Model (ScheduleModel/)

#### Schedule Management (`Schedule.py`)
- Core scheduling logic
- Manages path allocation and timetables
- Calculates objective function values

#### Path Computing (`PathComputing.py`)
- Calculates optimal taxiing paths
- Handles path conflict detection
- Optimizes waiting times

#### Timetable Management
- `PointTimetable.py`: Point time occupancy management
- `EdgeTimetable.py`: Edge time occupancy management

### 4. Task Model (TaskModel/)

#### Task Definition (`Task.py`)
- Aircraft task data structure
- Contains task ID, aircraft type, task type, start/end points, and other information

#### Aircraft Information (`aircraftInfo.py`)
- Aircraft type and performance parameters

## Data Format

### Task Data Format
```
aid	aircrafttype	tasktype	standpointid	runwayid	startime	P
T11537	1	0	P_265	16	1640966437	P_505
```
- `aid`: Task ID
- `aircrafttype`: Aircraft type (1/2)
- `tasktype`: Task type (0: Takeoff, 1: Landing)
- `standpointid`: Stand ID
- `runwayid`: Runway ID
- `startime`: Start timestamp
- `P`: Runway point

### Runway Data Format
```
15	P_433-40,P_1541-50,P_1532-50,P_1544-60,P_1227-60|P_565_1,P_572_1
```
- Runway ID: 15, 33, 16, 34
- Taxiway points and speed limits
- Runway entry points

## Main Features

1. **Path Planning**: Calculate optimal taxiing paths for each aircraft
2. **Conflict Detection**: Detect and resolve path conflicts
3. **Time Optimization**: Minimize total taxiing time
4. **Multi-Algorithm Support**: Provide multiple optimization algorithm options





