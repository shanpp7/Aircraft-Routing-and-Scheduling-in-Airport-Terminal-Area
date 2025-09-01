from src.ScheduleModel.Schedule import Schedule
from src.ScheduleModel.PathComputing import *
from src.ResourceModel.Map import *
from src.ResourceModel.Instance import *
import os
import pandas as pd
import time
import copy
import random
import datetime
from datetime import datetime


PATHS = {} # store the shortest path for each task planning tid:pointlist
def PathPlanning(instance):
    tasks = instance.tasks
    for tid in tasks.keys():
        task = tasks[tid]
        if task.tasktype == 0:
            startpid = task.runwaypointlist[0]
            endpid = task.standpointid
        else:
            startpid = task.standpointid
            endpid = runwayPointSelection(task.runwayid, startpid)
        #pointlist = instance.Map.computePath(instance.Map.G, 3, startpid, endpid)
        pointlist = instance.Map.choosePath(instance.Map.G, startpid, endpid, task, instance)
        PATHS[tid] = pointlist

# Resolve conflicts-B
def Arrange(instance, sequence):
    schedule = Schedule(instance)
    tasks = instance.tasks
    for tid in sequence:
        task = tasks[tid]
        pointlist = PATHS[tid]
        path = ComputeNoConflictPathTask(pointlist, task, schedule, instance)
        schedule.addPath(task, path)
    schedule = computeArrivingWaitingTime(instance,schedule)
    return schedule


def printInfo(instance, schedule):
    # print the coasting time information
    total = []
    totaltime = []
    sum=0
    tasks = instance.tasks
    for tid in tasks:
        task = tasks[tid]
        path = schedule.paths[tid]
        runway = instance.Map.Runways[task.runwayid]
        t = path.getTotalTime(task, runway)
        sum += t
        totaltime.append(t)

        # output specific information
        total.append(path.pointlist)
        t1 = []
        for t in path.time.values():
            tt = time.localtime(t)
            t1.append(time.strftime("%Y-%m-%d %H:%M:%S", tt))

        t2 = list(path.turningTime.values())
        t3 = list(path.waitingTime.values())
        total.append(t1)
        total.append(t3)
        total.append(t2)
    return (sum,total, totaltime)


# Sequential rollout worry-free-C
def ArrangeWaiting(instance, sequence):
    schedule = Schedule(instance)
    tasks = instance.tasks
    for tid in sequence:
        task = tasks[tid]
        pointlist = PATHS[tid]
        #path = ComputeNoConflictPathTask(pointlist, task, schedule, instance)
        path = ComputeNoConflictPath(pointlist, task, schedule, instance)
        schedule.addPath(task, path)
    schedule = computeArrivingWaitingTime(instance,schedule)
    return schedule


def runwayPointSelection(runwayid, runwaypointlist):
    runwayid = int(runwayid)
    R15_1_0 = ['P_565_1', 'P_563_1']
    R15_1_1 = ['P_572_1', 'P_575_1']
    R33_1_0 = ['P_1297_1', 'P_1299']
    R33_1_1 = ['P_1375_1']
    R16_1 = ['P_5_1', 'P_11']
    R34_1 = ['P_1110_1', 'P_1118']
    if (runwayid == 15):
        if (runwaypointlist in runwayType0):
            return R15_1_0[0]
        else:
            return R15_1_1[0]
    if (runwayid == 33):
        if (runwaypointlist in runwayType0):
            return R33_1_0[0]
        else:
            return R33_1_1[0]
    if (runwayid == 16):
        return R16_1[0]
    if (runwayid == 34):
        return R34_1[0]


# Local search algorithm
def greedy(instance):
    num_iterations = 50
    tasksid = list(instance.tasks.keys())
    iterations_without_improvement = 0
    num_tasks = len(tasksid)
    task_to_index = {task: index for index, task in
                     enumerate(tasksid)}  # the tid of each task corresponds to a subscript, similar to{'T11532': 0, 'T11533': 1, 'T1': 2, 'T11534': 3,……}
    current_order = list(range(num_tasks))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,……]
    random.shuffle(current_order)
    current_order1 = [tasksid[i] for i in current_order]  # ['T11532', 'T11533', 'T1', 'T11534', 'T11535', 'T11536']
    schedule = ArrangeWaiting(instance, current_order1)
    _, sum = schedule.getObjective(instance)
    current_energy = sum  # used for comparison to obtain the minimum value

    best_order = current_order.copy()
    best_energy = current_energy

    for iteration in range(num_iterations):
        print("iteration", iteration)
        start_i = time.time()
        better_order = current_order.copy()
        better_energy = current_energy
        i = random.randint(0, num_tasks - 1)
        for j in range(0, num_tasks):
            new_order = current_order.copy()
            if j != i:
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_order1 = [tasksid[i] for i in new_order]
                schedule = ArrangeWaiting(instance, new_order1)
                _, sum = schedule.getObjective(instance)
                new_energy = sum
                if new_energy < better_energy:
                    better_energy = new_energy
                    better_order = new_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
        # calculate the energy variation
        energy_change = better_energy - best_energy

        # if the new permutation is better or accepts a worse solution with a certain probability, update the current permutation
        # if energy_change < 0 or random.random() < math.exp(-energy_change / temperature):
        if energy_change < 0:
            print("better_order", better_order)
            print("better_energy", better_energy)
            current_order = better_order.copy()
            current_energy = better_energy
            best_order = current_order.copy()
            best_energy = current_energy
        else:
            break
        end_i = time.time()
        processing_time = round(end_i - start_i, 2)  # calculate the processing time for each iteration
        print(processing_time)

    best_order1 = [tasksid[i] for i in best_order]  # convert the subscript to a string
    print("best_order", best_order)
    print("best_order1", best_order1)
    print("best_energy", best_energy)
    return best_order1


# Simulated annealing algorithm
def simulatedAnnealing(instance):
    initial_temperature = 100
    cooling_rate = 0.995
    num_iterations = 50
    iterations_without_improvement = 0
    tasksid = list(instance.tasks.keys())
    num_tasks = len(tasksid)
    task_to_index = {task: index for index, task in
                     enumerate(tasksid)}
    current_order = list(range(num_tasks))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,……]
    current_order1 = [tasksid[i] for i in current_order]  # ['T11532', 'T11533', 'T1', 'T11534', 'T11535', 'T11536']
    schedule = ArrangeWaiting(instance, current_order1)
    _, sum = schedule.getObjective(instance)

    current_energy = sum

    best_order = current_order.copy()
    best_energy = current_energy

    temperature = initial_temperature
    listenergy = []
    for iteration in range(num_iterations):
        print("iteration", iteration)
        better_order = current_order.copy()
        better_energy = current_energy
        i = random.randint(0, num_tasks - 1)
        for j in range(0, num_tasks):
            new_order = current_order.copy()
            if j != i:
                new_order[i], new_order[j] = new_order[j], new_order[i]

                new_order1 = [tasksid[i] for i in new_order]

                schedule = ArrangeWaiting(instance, new_order1)
                _, sum = schedule.getObjective(instance)
                new_energy = sum
                if new_energy < better_energy:
                    better_energy = new_energy
                    better_order = new_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]

        energy_change = better_energy - best_energy

        if energy_change < 0:
            # if energy_change < 0:
            # listorder.append(better_order)
            print("better_order", better_order)
            print("better_energy", better_energy)
            listenergy.append(better_energy)
            current_order = better_order.copy()
            current_energy = better_energy
            iterations_without_improvement = 0
            best_order = current_order.copy()
            best_energy = current_energy

        else:
            iterations_without_improvement += 1
            if iterations_without_improvement >= 5:
                break

        temperature *= cooling_rate
    best_order1 = [tasksid[i] for i in best_order]
    print("best_order", best_order)
    print("best_order1", best_order1)
    print("best_energy", best_energy)
    return best_order1


class Gena_TSP(object):
    def __init__(self, map, size_pop=50, cross_prob=0.9, pmuta_prob=0.01, select_prob=0.8):
        self.size_pop = size_pop  # the number of groups is 100
        self.cross_prob = cross_prob  # cross probability is 0.9
        self.pmuta_prob = pmuta_prob  # mutation probability is 0.01
        self.select_prob = select_prob  # selection probability is 0.8
        self.tasksid = []
        self.num = 0  # the number of tasks corresponds to the length of the chromosome
        self.task_to_index = {}
        self.maxgen = 50  # 3 * self.num Maximum number of iterations
        self.initialize_positions(map)

        # Determine the number of choices for the offspring by selecting the probability
        self.select_num = max(math.floor(self.size_pop * self.select_prob + 0.5), 2)

        self.chrom = np.array(['T11532'] * self.size_pop * self.num).reshape(self.size_pop, self.num)  # 父
        self.sub_sel = np.array(['T11532'] * self.select_num * self.num).reshape(self.select_num, self.num)  # 子

        # Store the total time of each chromosome in the population, and the fitness corresponding to a single chromosome is its reciprocal
        self.fitness = np.zeros(self.size_pop)

    def initialize_positions(self, instance):
        self.tasksid = list(instance.tasks.keys())
        self.num = len(self.tasksid)
        self.task_to_index = {task: index for index, task in enumerate(self.tasksid)}

        # Randomly generate the initialization group function
    def rand_chrom(self,instance):
        rand_ch = list(range(self.num))
        for i in range(self.size_pop):
            random.shuffle(rand_ch)  # Disrupt chromosome coding
            rand_ch1 = [self.tasksid[i] for i in rand_ch]
            self.chrom[i, :] = rand_ch1 
            schedule = ArrangeWaiting(instance, rand_ch1)
            _, sum = schedule.getObjective(instance)
            self.fitness[i] = sum

    # The selection of offspring is based on the selection probability and the corresponding fitness function, and a random traversal selection method is adopted
    def select_sub(self):
        fit = 1. / (self.fitness) 
        cumsum_fit = np.cumsum(fit)  # Cumulative sum  a = np.array([1,2,3]) b = np.cumsum(a) b=1 3 6
        pick = cumsum_fit[-1] / self.select_num * (np.random.rand() + np.array(
            range(self.select_num))) 
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
                i += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]

    # Crossover: Perform crossover operations on the offspring individuals based on probability
    def cross_sub(self):
        if self.select_num % 2 == 0:
            num = range(0, self.select_num, 2)
        else:
            num = range(0, self.select_num - 1, 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    def intercross(self, ind_a, ind_b):  # ind_a，ind_b Paternal chromosome
        ind_a = ind_a.tolist()
        ind_b = ind_b.tolist()
        r1 = np.random.randint(self.num)  # Randomly generate an integer within num
        r2 = np.random.randint(self.num)
        while r2 == r1:  # 如果r1==r2
            r2 = np.random.randint(self.num)
        left, right = min(r1, r2), max(r1, r2)
        ind_a1 = ind_a.copy()
        ind_b1 = ind_b.copy()
        fragment1 = ind_a1[left:right + 1
        fragment2 = ind_b1[left:right + 1]

        ind_a[left:right + 1] = ind_b1[left:right + 1]
        ind_b[left:right + 1] = ind_a1[left:right + 1]
        for i in ind_a[:left]:
            while i in fragment2:
                index = ind_a[:left].index(i)
                i = fragment1[fragment2.index(i)]
                ind_a[index] = i
        for i in ind_a[right + 1:]:
            while i in fragment2:
                index = ind_a[right + 1:].index(i) + right + 1
                i = fragment1[fragment2.index(i)]
                ind_a[index] = i
        for i in ind_b[:left]:
            while i in fragment1:
                index = ind_b[:left].index(i)
                i = fragment2[fragment1.index(i)]
                ind_b[index] = i
        for i in ind_b[right + 1:]:
            while i in fragment1:
                index = ind_b[right + 1:].index(i) + right + 1
                i = fragment2[fragment1.index(i)]
                ind_b[index] = i
        ind_a = np.array(ind_a)
        ind_b = np.array(ind_b)
        return ind_a, ind_b

    # Under the control of the mutation probability, the mutation module randomly exchanges the positions of two points on a single chromosome
    def mutation_sub(self):
        for i in range(self.select_num):  # Traverse each selected offspring
            if np.random.rand() <= self.cross_prob:  # If the random number is less than the mutation probability
                r1 = np.random.randint(self.num) 
                r2 = np.random.randint(self.num)
                while r2 == r1:
                    r2 = np.random.randint(self.num) 
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]  # Randomly swap the positions of two points

    # Evolutionary reversal randomly selects two positions r1:r2 on the selected chromosome and reverses the elements of r1:r2 to r2:r1. If the fitness after the reversal is higher, the original chromosome is replaced; otherwise, it remains unchanged
    def reverse_sub(self, instance):
        for i in range(self.select_num): 
            r1 = np.random.randint(self.num) 
            r2 = np.random.randint(self.num)
            while r2 == r1: 
                r2 = np.random.randint(self.num)  
            left, right = min(r1, r2), max(r1, r2) 
            sel = self.sub_sel[i, :].copy()  
            self.sub_sel[i, left:right + 1] = sel[left:right + 1][::-1]  # Flip the (r1:r2) fragment in the chromosome to (r2:r1)
            # If the total time after flipping is greater than that of the original chromosome, it remains unchanged
            schedule = ArrangeWaiting(instance, sel)
            _, sum1 = schedule.getObjective(instance)
            schedule = ArrangeWaiting(instance, self.sub_sel[i, :])
            _, sum2 = schedule.getObjective(instance)

            if sum1 < sum2:
                self.sub_sel[i, :] = sel

    # When the offspring insert into the parent, a new population of the same size is obtained
    def reins(self):
        index = np.argsort(self.fitness)[::-1]
        self.chrom[index[:self.select_num], :] = self.sub_sel


def main(instance):
    Path_short = Gena_TSP(instance)  # Generate a genetic algorithm class
    Path_short.rand_chrom(instance)  # Initialize the parent class

    best_order = Path_short.chrom[0, :]
    best_time = Path_short.fitness[0]
    best_iteration = 0

    # Cyclic iterative genetic process
    for iteration in range(Path_short.maxgen):
        if iteration - best_iteration > 10:
            break
        print("iteration ", iteration + 1)
        Path_short.select_sub()
        Path_short.cross_sub() 
        Path_short.mutation_sub()
        Path_short.reverse_sub(instance)
        Path_short.reins()

        # Recalculate the time of the new group
        for j in range(Path_short.size_pop):

            schedule = ArrangeWaiting(instance, Path_short.chrom[j, :])
            _, sum = schedule.getObjective(instance)
            Path_short.fitness[j] = sum

        # Display the optimal path of the current group
        index = Path_short.fitness.argmin()
        if Path_short.fitness[index] < best_time:
            best_order = Path_short.chrom[index, :]
            best_time = Path_short.fitness[index]
            best_iteration = iteration
        print(best_time)

    print('best_time:' + str(best_time))
    print("best_order:", best_order)

    return best_order

class AssignmentQLearningAgent:
    # num_tasks indicate the number of particles, num_workers indicates the action to be executed
    def __init__(self, num_tasks, num_workers, learning_rate=0.1, discount_factor=0.1, exploration_rate=0.5):
        self.num_tasks = num_tasks
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        #self.q_table = np.zeros((num_tasks, num_workers)) # Initialize to 0
        self.q_table = np.ones((num_tasks, num_workers))  # Initialize to 1

    def choose_action(self, state):
        q_values = self.q_table[state, :]
        # softmax
        scaled_q = q_values - np.max(q_values)
        exp_q = np.exp(scaled_q / self.exploration_rate)
        probabilities = exp_q / np.sum(exp_q)
         # Choose your moves in roulette
        return np.random.choice(self.num_workers, p=probabilities)

    def update_q_table(self, state, action, next_state, reward):
        self.q_table[state, action] += self.learning_rate * (reward - self.q_table[state, action])


def main_PSO(instance, nums, iterion):
    particles = DParticles(instance, nums, iterion)
    # iterative optimization
    # Record the optimal value and the task sequence
    plist = [] # Record the most significant sequence generated in each iteration
    for j in range(particles.max_iterations):
        print(particles.global_best_value)
        for n in range(particles.num_particles):
            p1 = copy.deepcopy(particles.personal_best_position[n])
            p2 = copy.deepcopy(particles.personal_best_position[n])
            p3 = copy.deepcopy(particles.personal_best_position[n])
            newlist1 = particles.take_action1(p1, particles.global_best_position)
            particles.update_position(n, newlist1, instance)
            newlist2 = particles.take_action2(p2)
            particles.update_position(n, newlist2, instance)
            newlist3 = particles.take_action3(p3)
            particles.update_position(n, newlist3, instance)
        id = []
        for e in particles.global_best_position:
            id.append(particles.task_to_index[e])
        schedule = ArrangeWaiting(instance, id)
        (_, totaltime) = printInfo(instance, schedule)
        plist.append(totaltime)
    return plist

class DParticles:
    def __init__(self, instance, num_p, num_i, func):
        self.num_particles = num_p # Number of particles
        self.max_iterations = num_i # Maximum number of iterations
        self.tasksid = []
        self.num_tasks = 0  # Number of tasks
        self.task_to_index = {}

        self.positions = []  # The position of the particle
        self.values = []  # The fitness value corresponding to the particle position
        self.global_best_position = []
        self.global_best_value = float('inf')
        self.personal_best_position = []
        self.personal_best_value = []
        self.initialize_positions(instance,func)

    # Randomly generated particle positions (including initial positions) 
	# Given the number of particles and the number of tasks
    def initialize_positions(self, instance, func):
        self.tasksid = list(instance.tasks.keys())
        self.num_tasks = len(self.tasksid)
        self.task_to_index = {index: task for index, task in enumerate(self.tasksid)}
        for i in range(self.num_particles):
            position = list(range(self.num_tasks))
            random.shuffle(position)
            self.positions.append(position)
        for p in self.positions:
            id = []
            for e in p:
                id.append(self.task_to_index[e])
            schedule = func(instance, id)
            _, sum = schedule.getObjective(instance)
            self.values.append(sum)

        # Update the global position and extreme values
        for p, v in zip(self.positions, self.values):
            if v < self.global_best_value:
                self.global_best_value = v
                self.global_best_position = p
        # self.personal_best_position, self.personal_best_value = self.computingEliteSolution(instance)
        self.personal_best_position = copy.deepcopy(self.positions)
        self.personal_best_value = self.values.copy()


    def checklist(self, tasklist):
        indexlist = list(range(len(tasklist)))
        for e in tasklist:
            if e not in indexlist:
                break
            else:
                indexlist.remove(int(e))
        if len(indexlist) == 0:
            return True
        else:
            return False

    # Update the global extremum of the particle
    def update_global(self, value, position):
        self.global_best_value = value
        self.global_best_position = position

    # Update the individual extremum of the particle
    def update_personal(self, value, index, position):
        self.personal_best_value[index] = value
        self.personal_best_position[index] = position

    def take_action1(self, tasklist, bestposition):
        # Cross with the current global optimal position
		# Randomly swap two task indices
        indexlist = list(range(len(tasklist)))
        m = random.randint(0, len(tasklist) - 1)
        n = random.randint(0, len(tasklist) - 1)
        while (m == n):
            n = random.randint(0, len(tasklist) - 1)
        left, right = min(m, n), max(m, n)
        atasklist = bestposition[left:right]
        for e in atasklist:
            indexlist.remove(e)
        btasklist = tasklist[left:right]
        setC = set(btasklist) - set(atasklist)
        for i in range(0, left):
            if tasklist[i] in indexlist:
                indexlist.remove(tasklist[i])
            else:
                num = random.choice(list(setC))
                tasklist[i] = num
                setC.remove(num)
        for i in range(right, len(tasklist)):
            if tasklist[i] in indexlist:
                indexlist.remove(tasklist[i])
            else:
                num = random.choice(list(setC))
                tasklist[i] = num
                setC.remove(num)
        alllist = tasklist[0:left]
        alllist.extend(atasklist)
        alllist.extend(tasklist[right:])
        return alllist

    def take_action2(self, tasklist):
        # Randomly reverse the sequence of two positions partially
        m = random.randint(0, len(tasklist) - 1)
        n = random.randint(0, len(tasklist) - 1)
        while (m == n):
            n = random.randint(0, len(tasklist) - 1)
        left, right = min(m, n), max(m, n)
        # reverse_sublist(alllist, left, right)
        alllist = tasklist[left:right]
        alllist.reverse()
        btasklist = tasklist[0:left]
        btasklist.extend(alllist)
        btasklist.extend(tasklist[right:])
        return btasklist

    def take_action3(self, tasklist):
        # Randomly reverse the partial learning of two position sequences
        m = random.randint(0, len(tasklist) - 1)
        n = random.randint(0, len(tasklist) - 1)
        while (m == n):
            n = random.randint(0, len(tasklist) - 1)
        left, right = min(m, n), max(m, n)
        btaskliat = []
        for i in range(left, right):
            num = 0 + len(tasklist) - 1 - tasklist[i]
            tasklist[i] = num
            btaskliat.append(tasklist[i])

        for i in range(0, left):
            if tasklist[i] in btaskliat:
                btaskliat.remove(tasklist[i])
                num = 0 + len(tasklist) - 1 - tasklist[i]
                tasklist[i] = num

        for i in range(right, len(tasklist)):
            if tasklist[i] in btaskliat:
                btaskliat.remove(tasklist[i])
                num = 0 + len(tasklist) - 1 - tasklist[i]
                tasklist[i] = num

        alllist = tasklist[0:left]
        alllist.extend(tasklist[left:right])
        alllist.extend(tasklist[right:])
        return alllist


def main_DPSO(instance, nums, iterion, func,iinstance):
    particles = DParticles(instance, nums, iterion, func)
    records = []
    # iterative optimization
    for j in range(particles.max_iterations):
        # solve the distribution problem
        solve_assignment_problem(particles.num_particles, 3, 50, particles, func, instance)

        if j == particles.max_iterations - 1:
            id = [particles.task_to_index[e] for e in particles.global_best_position]
            schedule = func(instance, id)
            _, final_sum = schedule.getObjective(instance)
        records.append({
            "(10,5,-1)":(10,5,-1),
            "instance":iinstance,
            "iteration": j,
            "best_obj": particles.global_best_value
        })

    final_sequence = [particles.task_to_index[e] for e in particles.global_best_position]
    return particles.global_best_value, final_sequence,records



def solve_assignment_problem(num_tasks, num_workers, num_episodes, particles, fun, instance):
    #result = solve_assignment_problem(particles.num_particles, 3, 100, particles, func, instance)
    agent = AssignmentQLearningAgent(num_tasks, num_workers)
    # 1       # 2, 5, 8
    lr1 = 10
    lr2 = 2
    lr3 = -1

    tlist = []
    rlist1 = []
    rlist2 = []
    rlist3 = []
    rtoal1 = 0
    rtoal2 = 0
    rtoal3 = 0
    actionnum = -1
    personvaluelist = []

    # train agent
    count = 0
    for episode in range(num_episodes):
        state = np.random.randint(num_tasks)  # randomly select particles
        #print("state是",state)
        visited = [state]
        unvisited = list(range(num_tasks))
        unvisited.remove(state)
        while unvisited:
            action = agent.choose_action(state)
            p = copy.deepcopy(particles.personal_best_position[state])
            if action == 0:
                newlist = particles.take_action1(p, particles.global_best_position)
                actionnum = 0
            elif action == 1:
                newlist = particles.take_action2(p)
                actionnum = 1
            else:
                newlist = particles.take_action3(p)
                actionnum = 2
            # check for errors in the sequence
            if not particles.checklist(newlist):
                # for i in range(num_workers):
                #     agent.q_table[state][i] = 0
                next_state = np.random.choice(unvisited)
                state = next_state
                continue
            next_state = np.random.choice(unvisited)
            id = []
            for e in newlist:
                id.append(particles.task_to_index[e])
            schedule = fun(instance,id)
            _, sum = schedule.getObjective(instance)

            # v1
            # if sum < particles.personal_best_value[state]:
            #     print(sum)
            #     particles.update_personal(sum, state, newlist)
            #     if sum < particles.global_best_value:
            #         particles.update_global(sum, newlist)
            #     for i in range(num_workers):
            #         agent.q_table[state][i] = 0
            #     continue

            # v2
            reward = 0
            if sum < particles.global_best_value:
                reward = (particles.global_best_value - sum) / sum
                agent.q_table[state][action] += lr1 * reward
                particles.global_best_value = sum
                particles.global_best_position = newlist
                particles.personal_best_position[state] = newlist
                particles.personal_best_value[state] = sum
            elif sum < particles.personal_best_value[state]:
                #print(count, sum)
                count += 1
                reward = (particles.personal_best_value[state] - sum) / sum
                agent.q_table[state][action] += lr2 * reward
                particles.personal_best_position[state] = newlist
                particles.personal_best_value[state] = sum
            else:
                reward = (sum - particles.personal_best_value[state]) / particles.personal_best_value[state]
                agent.q_table[state][action] += lr3 * reward

            personvaluelist.append(particles.global_best_value)

            if actionnum == 0:
                rtoal1 += reward
            elif actionnum == 1:
                rtoal2 += reward
            else:
                rtoal3 += reward
            rlist1.append(rtoal1)
            rlist2.append(rtoal2)
            rlist3.append(rtoal3)

            state = next_state
            visited.append(state)
            unvisited.remove(state)

    assignment = [np.argmax(agent.q_table[task, :]) for task in range(num_tasks)]
    for i in range(len(assignment)):
        p = copy.deepcopy(particles.personal_best_position[i])
        if assignment[i] == 0:
            newlist = particles.take_action1(p, particles.global_best_position)
        elif assignment[i] == 1:
            newlist = particles.take_action2(p)
        else:
            newlist = particles.take_action3(p)

        if not particles.checklist(newlist):
            continue

        id = []
        for e in newlist:
            id.append(particles.task_to_index[e])
        schedule = fun(instance, id)
        _, sum = schedule.getObjective(instance)
        if sum < particles.personal_best_value[i]:
            #print(sum)
            particles.personal_best_position[i] = newlist
            particles.personal_best_value[i] = sum
        if sum < particles.global_best_value:
            #print(sum)
            particles.global_best_value = sum
            particles.global_best_position = newlist
    

def main1():
    # obtain the upper-level directory of the current script
    currentpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    pointFile = os.path.join(currentpath,  'data', 'points.xlsx')
    edgeFile = os.path.join(currentpath, 'data', 'edges16.xlsx')
    runwayFile = os.path.join(currentpath, 'data', 'runway.txt')

    task_folder = os.path.join(currentpath, 'Algorithms', 'txtfile')
    task_files = [f for f in os.listdir(task_folder) if f.endswith('.txt')]

    # parameters of particle swarm optimization algorithm
    nums = 25
    iterion = 2
    iinstance= 1
    all_records = []

    for task_filename in task_files:
        taskFile = os.path.join(task_folder, task_filename)

        # initialize the test instance
        instance = TestInstance(pointFile, edgeFile, runwayFile, taskFile)
        PathPlanning(instance)


        best_obj, best_seq,records = main_DPSO(instance, nums, iterion, ArrangeWaiting,iinstance)
        all_records.extend(records)
        iinstance+=1

    df_all = pd.DataFrame(all_records)
    output_path = os.path.join(currentpath, f"L__{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    df_all.to_excel(output_path, index=False)

if __name__ == "__main__":
    main1()
