import os
import ayarlar
import numpy as np
import copy
import random
import torch
from collections import deque
import time
from tqdm import tqdm
from sos1 import SosEnv
from Ajansos import Agent
from torch.utils.tensorboard import SummaryWriter
from visualizersos import Visualizer

writer = SummaryWriter(flush_secs=5, log_dir=f'logs/{ayarlar.MODEL_NAME}_{int(time.time())}')

visu = Visualizer()
env = SosEnv()
visu.env = env
agent = Agent(env, (0,1), ayarlar.LOAD_MODEL)

suma = 0
draw = 0
winP0 = 0
winP1 = 0

for episode in tqdm(range(1, ayarlar.EPISODES + 1), ascii=True, unit='episodes'):
    env.start()
    p0History = [] 
    p1History = []
    
    done = False
    while not done:
        toAppend = [copy.deepcopy(env.board)]
        action = agent.p1Move(env.board)
        
        new_state, reward, done = env.move(action)
        reward = -reward
        toAppend.append(action)
        toAppend.append(reward)
        p1History.append(toAppend)
        
        if episode % ayarlar.SHOW_EVERY == 0 and ayarlar.IS_VISUALIZER_ON:
            visu.show()
            
        if done:
            toAppend = [copy.deepcopy(new_state)]
            action2 = agent.p1Move(new_state)
        
            _, reward, done = env.move(action2)
            toAppend.append(action2)
            toAppend.append(reward)
            p0History.append(toAppend)
            
            if done:
                state, action, _, = p1History.pop()
                p1History.append([state, action, reward])
                suma += -1
                winP0 += 1
                
        elif done and reward == 0:
            draw += 1 

        else:            
            state, action, _, = p0History.pop()
            p0History.append([state, action, reward])
            suma += 1
            winP1 += 1


        if episode % ayarlar.SHOW_EVERY == 0 and ayarlar.IS_VISUALIZER_ON:
            visu.show()

    agent.train(p1History, p0History)

    if not episode % ayarlar.AGGREGATE_STATS_EVERY or episode == 1:
        writer.add_scalar('sum', suma, episode)
        writer.add_scalar('epsilon', ayarlar.epsilon, episode)
        writer.add_scalar('draw', draw, episode)
        writer.add_scalar('winP0', winP0, episode)
        writer.add_scalar('winP1', winP1, episode)
       
        if not ayarlar.IS_TEST:
            torch.save(agent.model.state_dict(), f'models/{ayarlar.MODEL_NAME}_{winP1}_{int(time.time())}.model')
        draw = 0
        suma = 0
        winP0 = 0
        winP1 = 0
        
    if ayarlar.epsilon > ayarlar.MIN_EPSILON:
        ayarlar.epsilon *= ayarlar.EPSILON_DECAY
        ayarlar.epsilon = max(ayarlar.MIN_EPSILON, ayarlar.epsilon)