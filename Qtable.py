#Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

from torch import nn
from tqdm import tqdm
import numpy as np
import random
import torch 

Q_tap = [[0,0,0,-100,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,100]]

Q_deg = [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]]

lr = 0.7
ga = 0.8

class Q_learn:
    def algoritma(self, state, action):
        if self.Q(state) == 0:
            new_deg = self.Q(action) + lr * (Q_tap[action[0]][action[1]] + ga * np.max(self.Qmax(action)) - self.Q(action)) + Q_tap[state[0]][state[1]]
            new_deg = round(new_deg, 1)
            Q_deg[state[0]][state[1]] = new_deg
            
    def Qmax(self, action):
        Q_list = []
        legal = self.legal(action)
        for i in range(len(legal)):
            Q_list.append(self.Q([legal[i][0],legal[i][1]]))
          
        return Q_list
        
    def legal(self, action):
        legalmove = []
        if action[0] != 0:
            legalmove.append([action[0] - 1, action[1]])
            
        if action[0] != 3:
            legalmove.append([action[0] + 1, action[1]])
            
        if action[1] != 0:
            legalmove.append([action[0], action[1] - 1])
            
        if action[1] != 4:
            legalmove.append([action[0], action[1] + 1])
     
        return legalmove

    def Q(self, state):
        deger = Q_deg[state[0]][state[1]]
        
        return deger
    
    def moveQ(self, laststate, state):
        Q_list = []
        legal = self.legal(state)
        if legal.count(laststate) != 0:
            legal.remove(laststate)
        for i in range(len(legal)):
            Q_list.append(self.Q([legal[i][0],legal[i][1]]))
        act = np.argmax(Q_list)
        act = legal[act]
        
        return act
    
    def deepQ(self, laststate, state):
        move1list = []
        legal1 = self.legal(state)
        if legal1.count(laststate) != 0:
            legal1.remove(laststate)
        for i in range(len(legal1)):
            Q_list = []
            legal2 = self.legal(legal1[i])
            if legal2.count(state) != 0:
                legal2.remove(state)
            for x in range(len(legal2)):
                Q_list.append(self.Q([legal2[x][0],legal2[x][1]]))
            move1list.append(np.max(Q_list))
        act = np.argmax(move1list)
        act = legal1[act]
        
        return act
   
class DDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.createModel()

    def createModel(self):      
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 4)

    def forward(self, state): 
        x = self.layer1(state) 
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.tanh(x)

        return x
    
class Agent():
    def __init__(self):
        self.model = DDQN()
        
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_f = nn.MSELoss()
        
    def backpropagate(self, state, index):
        self.opt.zero_grad()
        state = torch.tensor(state, dtype=torch.float)
        output = self.model(state)
        target = output.clone().detach()
        target[index] = 1

        loss = self.loss_f(output, target)
        loss.backward()
        self.opt.step()
 
         
alg = Q_learn()

state = [0, 0]
action = [0, 0]
for _ in range(20):
    for episode in tqdm(range(150), unit='episodes'):
        legalmove = alg.legal(state)
        act = random.choice(legalmove)
        action = [act[0], act[1]]

        alg.algoritma(state, action)
    
        state = [act[0], act[1]]
    
    laststate = [0, 0]
    state = [0, 0]
    done = False
    while not done:
        act = alg.moveQ(laststate, state)
        laststate = [state[0], state[1]]
        state = [act[0], act[1]]

        print(Q_deg)

        if Q_tap[act[0]][act[1]] == 100:
            done = True
    
"""   
agent = Agent()
Q_ta = [0,10,2,-100]
Q_ta = torch.tensor(Q_ta, dtype=torch.float)
agent.backpropagate(Q_ta, 2)
"""
[[-6.3, -21.0, -70.0, -121.0, -106.3],
 [-1.9, -6.3, -21.0, -106.3, 0],
 [-0.6, -1.9, 0.0, 0, 0], 
 [-0.2, 0.0, 0.0, 0.0, 0]]
[[-9.0, -30.0, -100.0, -100.0, -100.0],
 [16.1, -9.0, -30.0, -70.0, -0.2], 
 [13.4, 28.7, -9.0, -2.7, -0.8], 
 [20.1, 8.6, 56.0, 16.8, 100.0]]