import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from src.Algorithms.solutionGeneration import *

class TaxiSequenceSchedulingEnv(gym.Env):
    def __init__(self, instance, max_steps):
        super().__init__()
        self.num_tasks = len(list(instance.tasks.keys()))
        self.max_steps = max_steps
        self.current_step = 0
        self.min_cost = float('inf')
        self.tasksid = list(instance.tasks.keys())
        self.instance = instance
        # self.action_pairs = [(i, j) for i in range(num_tasks) for j in range(i+1, num_tasks)]
        self.task_to_index = {index: task for index, task in enumerate(self.tasksid)}  # store the correspondence between the task id and the index
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=self.num_tasks-1, shape=(self.num_tasks,), dtype=np.int32)
        self.sequence = []
        self.best_sequence = []

    def reset(self):
        self.sequence = np.random.permutation(self.num_tasks)
        self.sequence = list(self.sequence)
        self.current_step = 0
        self.min_cost = float('inf')
        return self.sequence.copy()

    def take_action1(self):
        # randomly swap two task indices
        i, j = random.sample(range(len(self.sequence)), 2)
        # swap the elements at these two positions
        self.sequence[i], self.sequence[j] = self.sequence[j], self.sequence[i]

    def take_action2(self):
        # randomly reverse the sequence of two positions partially
        tasklist = self.sequence.copy()
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
        self.sequence = btasklist.copy()


    def take_action3(self):
        tasklist = self.sequence.copy()
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
        self.sequence = alllist.copy()

    def step(self, action):
        if action == 0:
            self.take_action1()
        elif action == 1:
            self.take_action2()
        else:
            self.take_action3()
        reward = -self.evaluate_cost(self.sequence)
        if -reward < self.min_cost:
            self.best_sequence = self.sequence
            self.min_cost = -reward
        self.current_step += 1
        done = self.current_step >= self.max_steps # multi-step optimization
        return self.sequence.copy(), reward, done, {}

    def evaluate_cost(self, seq):
        # simplified version: The total sliding time is the sum of the task position and can be replaced with real simulation evaluation
        id = []
        for e in seq:
            id.append(self.task_to_index[e])
        schedule = Arrange(self.instance, id)
        (_, totaltime) = printInfo(self.instance, schedule)
        sum = 0
        for n in totaltime:
            sum += n
        print(sum)
        return sum

# instantiation environment
# test a single task file
currentpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pointFile = currentpath + '\data\points.xlsx'
edgeFile = currentpath + '\data\edges16.xlsx'
runwayFile = currentpath + '\data\\runway.txt'
taskFile = currentpath + '\data\\testdata_300.txt'

instance = TestInstance(pointFile, edgeFile, runwayFile, taskFile)
print(list(instance.tasks.keys()))
PathPlanning(instance)
maxSteps = 10
env = TaxiSequenceSchedulingEnv(instance, maxSteps)

# create a PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=0,
    learning_rate=1e-3,
    n_steps=64,
    batch_size=32,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.9,
)

# train model
model.learn(total_timesteps=100)

# test model
obs = env.reset()
done = False
print("Initial sequence:", obs)
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
print("Optimal sequence:", env.best_sequence)
print("Optimal sequence target value:", env.min_cost)
