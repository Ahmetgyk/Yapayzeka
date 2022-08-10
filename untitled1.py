import os
import constants
import numpy as np
import copy 
import random
import torch
from collections import deque
import time
from tqdm import tqdm
from tictactoe import TicTacToeEnv
from torchAgent import Agent
from torch.utils.tensorboard import SummaryWriter
from visualizer import Visualizer

write = SummaryWriter(flush_secs=5, log_dir =  f'logs/{constants.MODEL_NAME}_{int(time.time())}')

visul = Visualizer()
env = TicTacToeEnv()
visul.env = env
agent = Agent(env, (0,1), constants.LOAD_MODEL)

suma = 0
draw = 0
winX = 0
winO = 0
for episode in tqdm(range(1, constants.EPISODES + 1), ascii=True, unit="episodes"):
    env.start()
    xMoveHistories = []
    oMoveHistories = []
    
    done = False
    while not done:
        toAppend = [copy.deepcopy(env.board)]
        action = agent.makeMoveX(env.board)
        
        new_state, reward, done = env.move(action)
        reward = -reward
        toAppend.append(action)
        toAppend.append(reward)
        xMoveHistories.append(toAppend)
        
        if episode % constants.SHOW_EVERY == 0 and constants.IS_VISUALIZER_ON:
            visul.show()
            
        if not done:
            toAppend = [copy.deepcopy(new_state)]
            action2 = agent.makeMoveO(new_state)
            _, reward, done = env.move(action2)
            toAppend.append(action2)
            toAppend.append(reward)
            oMoveHistories.append(toAppend)
            
            if done:
                state,action, _, = xMoveHistories.pop()
                xMoveHistories.append([state, action, reward])
                suma += -1
                winO += 1
            
            elif done and reward == 0:
                draw += 1
            
            else:
                state, action, _, = oMoveHistories.pop()
                oMoveHistories.append([state, action, reward])
                suma += 1
                winX += 1 
               
            if episode % constants.SHOW_EVERY == 0 and constants.IS_VISUALIZER_ON:
                visul.show()
                
        agent.train(xMoveHistories, oMoveHistories)
        
        if not episode % constants.AGGREGATE_STATS_EVERY or episode == 1:
            write.add_scalar("sum", suma, episode)
            write.add_scalar("episilon", constants.epsilon, episode)
            write.add_scalar("draw", draw, episode)
            write.add_scalar("winO", winO, episode)
            write.add_scalar("winX", winX, episode)
            
            if not constants.IS_TEST:
                torch.save(agent.model.state_dict(), f'models/{constants.MODEL_NAME}_{winX}_{int(time.time())}.model')
                draw = 0
                suma = 0
                winO = 0
                winX = 0
                
            if constants.episilon > constants.MIN_EPSILON:
                constants.epsilon *= constants.EPSILON_DECAY
                constants.epsilon = max(constants.MIN_EPSILON, constants.epsilon)