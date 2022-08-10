import torch
import torch.nn as nn
import numpy as np
import random
import constants

input_size = 9
hidden_size = 36
output_size = 9

class DDQN(nn.Module):
    def __init__(self):
        super().__init__()
        CreateModel(input_size, hidden_size, output_size)
        
    def CreateModel(self, input_size, hidden_size, output_size):
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        x1 = self.layer1(state)
        xr1 = torch.relu(x1)
        x2 = self.layer2(xr1)
        xr2 = torch.relu(x2)
        x3 = self.layer3(xr2)
        x = torch.tanh(x3)
        return x
    
class Agent():
    def __init__(self, env, player, load = False):
        self.player = player
        self.env = env
        
        self.model = DDQN()
        self.target_model = DDQN()
        
        if load:
            self.model.load_state_dict(torch.load(constants.MODEL_PATH))
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.eval()
        
        self.opt = torch.optim.SGD(self.model.parameters(), lr = constants.LR)
        self.lossF = nn.MSELoss()
        
        self.target_update = 0
        
    def backpropagate(self, state, move_index, target_value):
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
        return torch.tensor(toTensor, dtype = torch.float)
    
    def OMove(self, state):
        if np.random.random() > constants.epsilon:
            mask = self.env.getMask()
            q_values = []
            for q, maskValue in zip(self.get_qs(state, self.model), mask):
                if maskValue == 1:
                    q_values.append(q)
                else:
                    q_values.append(-999)
            action = np.argmax(q_values)
            
        else:
            legalMove = self.env.getLegalMoves(self.env.board)
            actionList = []
            for move in legalMove:
                actionList.append(constants.ACTION_LIST.index((move)))
            action = random.choice(actionList)
        return action
    
    def OMove(self, state):
        if np.random.random() > constants.epsilon:
            mask = self.env.getMask()
            q_values = []
            for q, maskValue in zip(self.get_qs(state, self.model), mask):
                if maskValue == 1:
                    q_values.append(q)
                else:
                    q_values.append(999)
            action = np.argmax(q_values)
            
        else:
            legalMove = self.env.getLegalMoves(self.env.board)
            actionList = []
            for move in legalMove:
                actionList.append(constants.ACTION_LIST.index((move)))
            action = random.choice(actionList)
        return action
    
    def train(self, xMoveHistory, oMoveHistory):
        nextState, action, reward = oMoveHistory.pop()
        self.backpropagate(nextState, action, reward)
        
        for _ in range(len(oMoveHistory)):
            current_state, action, reward = oMoveHistory.pop()
            next_qs = self.get_qs(nextState, self.target_model)
            mask = self.env.getMask(nextState)
            qs_to_select = []
            for q, maskValue in zip(next_qs, mask):
                if maskValue == 1:
                    qs_to_select.append(q.item())
            next_q = min(qs_to_select)
            new_q = reward + next_q * constants.DISCOUNT
            self.backpropagate(current_state, action, new_q)
            nextState = current_state
            
            
        nextState, action, reward = xMoveHistory.pop()
        self.backpropagate(nextState, action, reward)
        
        for _ in range(len(xMoveHistory)):
            current_state, action, reward = xMoveHistory.pop()
            next_qs = self.get_qs(nextState, self.target_model)
            mask = self.env.getMask(nextState)
            qs_to_select = []
            for q, maskValue in zip(next_qs, mask):
                if maskValue == 1:
                    qs_to_select.append(q.item())
            next_q = min(qs_to_select)
            mew_q = reward + next_q * constants.DISCOUNT
            self.backpropagate(current_state, action, new_q)
            nextState = current_state
            
        self.target_update += 1
        if self.target_update > constants.UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update = 0
        