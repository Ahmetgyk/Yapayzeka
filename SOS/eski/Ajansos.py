import torch
from torch import nn
import numpy as np
import ayarlar
import random

class DDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.createModel()
        
    def createModel(self):
        self.layer1 = nn.Linear(36, 72)
        self.layer2 = nn.Linear(72, 72)
        self.layer3 = nn.Linear(72, 36)
        
    def forward(self, state):
        x1 = self.layer1(state)
        xr1 = torch.relu(x1)
        x2 = self.layer2(xr1)
        xr2 = torch.relu(x2)
        x3 = self.layer3(xr2)
        out = torch.relu(x3)      
        return out
    
class Agent():
    def __int__(self, env, player, load=False):
        self.player = player
        self.env = env 
        
        self.model = DDQN()
        self.target_model = DDQN()
        
        if load:
            self.model.load_state_dict(torch.load(ayarlar.MODEL_PATH))
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.eval()
        
        self.opt = torch.optim.SGD(self.model.parameters(), lr=ayarlar.LR)
        self.lossF = nn.MSELoss()
        
        self.target_update = 0
        
    def backpropgate(self, state, move_index, target_value):
        self.opt.zero_grad()
        output = self.model(self.convert_to_tensor(state))
        target = output.clone().detach()
        
        target[move_index] = target_value
        
        loss = self.lossF(output, target)
        loss.backward()
        self.opt.step()
        
    def get_qs(self, state, model):
        inputs = self.convert_to_tensor(state)
        with torch.no_grad():
            outputs = model(inputs)
            
        return outputs
    
    def convert_to_tensor(self, state):
        toTensor = []
        for row in state:
            for square in row:
                if square == (0,0):
                    toTensor.append(0)
                elif square == (0,1):
                    toTensor.append(1)
                else:
                    toTensor.append(-1)
                    
        return torch.tensor(toTensor, dtype=torch.float)
    
    def p0Move(self, state):
        if np.random.random() > ayarlar.EPSILON:
            q_move = []
            mask = self.env.getLegalMoves(self.env.board)
            
            for q, Mask in zip(self.get_qs(state, self.model), mask):               
                if Mask == 1:                    
                    q_move.append(q)
                else:
                    q_move.append(-999)
            action = np.argmax(q_move)  


        else:
            Move = [1 ,-1]
            actionList = []
            legalMoves = self.env.getLegalMoves(self.env.board)
            
            for move in range(len(legalMoves)):       
                actionList.append(ayarlar.ACTION_LIST[move])
            action = random.choice(actionList)
            moveSO = random.choice(Move)
        return action, moveSO
    
    def p1Move(self, state):
        if np.random.random() > ayarlar.EPSILON:
            q_move = []
            mask = self.env.getLegalMoves(self.env.board)
            
            for q, Mask in zip(self.get_qs(state, self.model), mask):               
                if Mask == 1:                    
                    q_move.append(q)
                else:
                    q_move.append(999)
            action = np.argmax(q_move)  


        else:
            Move = [1 ,-1]
            actionList = []
            legalMoves = self.env.getLegalMoves(self.env.board)
            
            for move in range(len(legalMoves)):       
                actionList.append(ayarlar.ACTION_LIST[move])
            action = random.choice(actionList)
            moveSO = random.choice(Move)
        return action, moveSO
    
    def train(self, p0History, p1History):
        nextstate, action, reward = p0History.pop()
        self.backpropgate(nextstate, action, reward)
        
        for _ in range(len(p0History)):
            current_state, action, reward = p0History.pop()
            next_qs = self.get_qs(nextstate, self.target_model)
            
            q_move = []
            mask = self.env.getLegalMoves(self.env.board)
            
            for q, Mask in zip(next_qs, mask):               
                if Mask == 1:                    
                    q_move.append(q.item())
            next_qs = max(q_move)
            new_q = reward + next_qs * ayarlar.DISCOUNT
            
            self.backpropgate(current_state, action, new_q)
            next_qs = current_state
            
        nextstate, action, reward = p1History.pop()
        self.backpropgate(nextstate, action, reward)
        
        for _ in range(len(p1History)):
            current_state, action, reward = p1History.pop()
            next_qs = self.get_qs(nextstate, self.target_model)
            
            q_move = []
            mask = self.env.getLegalMoves(self.env.board)
            
            for q, Mask in zip(next_qs, mask):               
                if Mask == 1:                    
                    q_move.append(q.item())
            next_qs = min(q_move)
            new_q = reward + next_qs * ayarlar.DISCOUNT
            
            self.backpropgate(current_state, action, new_q)
            next_qs = current_state
            
        self.target_update += 1
        if self.target_update > ayarlar.UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update = 0