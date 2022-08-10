import sos1 
import ayarlar as ayarlar
import random
import numpy as np
import torch
from torch import nn
import random
import copy

class DDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.createModel()

    def createModel(self):      
        self.layer1 = nn.Linear(36, 72)
        self.layer2 = nn.Linear(72, 72)
        self.layer3 = nn.Linear(72, 72)

    def forward(self, state): 
        x = self.layer1(state) 
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.tanh(x)

        return x

class Agent():
    def __init__(self, env, load = False):
        self.env = env
        
        self.model = DDQN()
        self.target_model = DDQN()
        
        if load:
            self.model.load_state_dict(torch.load(ayarlar.MODEL_PATH))
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.eval()
        
        self.opt = torch.optim.SGD(self.model.parameters(), lr=ayarlar.LR)
        self.loss_f = nn.MSELoss()
        
        self.target_update = 0
        
    def backpropagate(self, state, move_index, target_value):
        self.opt.zero_grad()
        state = torch.tensor(state, dtype=torch.float)
        output = self.model(state)
        target = output.clone().detach()
        
        target[move_index] = target_value

        loss = self.loss_f(output, target)
        loss.backward()
        self.opt.step()
    
    def SOSelP1(self, outputs, state):
        q_move = []
        
        print("state_p1",state)
        for q, Mask in zip(outputs, self.env.getLegalMoves(state)):               
            if Mask == 1:                    
                q_move.append(q[0])
                q_move.append(q[1])
        
        next_q = max(q_move)
        
        for n in range(len(outputs)):
            for m in range(2):
                if outputs[n][m] == next_q:
                    nV = n
                    mV = m
       
        return nV, mV
    
    def SOSelP2(self, outputs, state):
        q_move = []
        
        print("state_p2",state)
        for q, Mask in zip(outputs, self.env.getLegalMoves(state)):               
            if Mask == 1:                    
                q_move.append(q[0])
                q_move.append(q[1])
        
        next_q = min(q_move)
        
        for n in range(len(outputs)):
            for m in range(2):
                if outputs[n][m] == next_q:
                    nV = n
                    mV = m
            
        return nV, mV
        
    def get_qs(self, state, model):
        outputs = []
        with torch.no_grad():
            output = model(state)
            for i in range(0, 72, 2):
                toapp = [output[i], output[i + 1]]
                outputs.append(toapp)

        return outputs

    def moveP1(self, p1history, p1done, p2done):    
        lastscor = self.env.score[1]                 
        toAppend = [copy.deepcopy(self.env.board)]
        action, SO = self.makeMove(self.env.board, (1,0))
        self.env.move(action, SO)
        toAppend.append([action, SO])
        toAppend.append(self.env.score[1])
    
        p1history.append(toAppend)
        score = self.env.score[1]
        
        if score == lastscor:
            p1done = False
            p2done = True
            
        elif score != lastscor:
            p1done = True
            p2done = False
        
        return p1done, p2done, score
    
    def moveP2(self, p2history, p1done, p2done):
        #                              P2 OYUNU MİNİMİZE ETMEYE ÇALIŞTIĞI İÇİN REWARDI - YAPICAZ
        lastscor = self.env.score[2] 
        toAppend = [copy.deepcopy(self.env.board)]
        action, SO = self.makeMove(self.env.board, (0,1))
        self.env.move(action, SO)
        toAppend.append([action, SO])
        toAppend.append(self.env.score[2])
    
        p2history.append(toAppend)
        score = self.env.score[2]
        
        if score == lastscor:
            p2done = False
            p1done = True
            
        elif score != lastscor:
            p2done = True
            p1done = False
       
        return p1done, p2done, score

    def makeMove(self, state, player):               
        if np.random.random() > 0:

            state = torch.tensor(state, dtype=torch.float)
            
            if player == (0,1):
                n, m = self.SOSelP2(self.get_qs(state, self.model), state)
            if player == (1,0):
                n, m = self.SOSelP1(self.get_qs(state, self.model), state)
            
            action = n
            if m == 0:
                moveSO = -1
            if m == 1:
                moveSO = 1
                
        else:
            Move = [1 ,-1]
            actionList = []
            for move in range(len(self.env.getLegalMoves(state))):
                if self.env.getLegalMoves(state)[move] == 1: 
                    actionList.append(move)
            action = random.choice(actionList)
            moveSO = random.choice(Move)
            
        return action, moveSO
    
    def train(self, p1his, p2his):
       
        nextState, action, reward = p1his.pop()
        
        if action[1] == -1:
            move_index = 2 * action[0]
        if action[1] == 1:
            move_index = 2 * action[0] + 1           
        self.backpropagate(nextState, move_index, reward)
        
        for _ in range(len(p1his)):
            curret_state, action, revard = p1his.pop()
            
            nextState = torch.tensor(nextState, dtype=torch.float)
            outputs = self.get_qs(nextState, self.target_model)
            q_move = []
            for q, Mask in zip(outputs, self.env.getLegalMoves(nextState)):               
                if Mask == 1:                    
                    q_move.append(q[0])
                    q_move.append(q[1]) 
            next_q = max(q_move)
            
            new_q = reward + next_q *ayarlar.DISCOUNT
            
            if action[1] == -1:
                move_index = 2 * action[0]
            if action[1] == 1:
                move_index = 2 * action[0] + 1   
            self.backpropagate(curret_state, move_index, new_q)
            
            nextState = curret_state
            
            
        nextState, action, reward = p2his.pop()
        
        if action[1] == -1:
            move_index = 2 * action[0]
        if action[1] == 1:
            move_index = 2 * action[0] + 1           
        self.backpropagate(nextState, move_index, reward)
        
        for _ in range(len(p2his)):
            curret_state, action, revard = p2his.pop()
         
            nextState = torch.tensor(nextState, dtype=torch.float)
            outputs = self.get_qs(nextState, self.target_model)
            q_move = []
            for q, Mask in zip(outputs, self.env.getLegalMoves(nextState)):               
                if Mask == 1:                    
                    q_move.append(q[0])
                    q_move.append(q[1])        
            next_q = min(q_move)
            
            new_q = reward + next_q *ayarlar.DISCOUNT
            
            if action[1] == -1:
                move_index = 2 * action[0]
            if action[1] == 1:
                move_index = 2 * action[0] + 1   
            self.backpropagate(curret_state, move_index, new_q)
            
            nextState = curret_state
                  
        self.target_update += 1
        if self.target_update > ayarlar.UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update = 0