import os
import ayarlar
import numpy as np
import copy
import random
import torch
import time
from collections import deque
from tqdm import tqdm
from sos1 import SosEnv
from Ajan import Agent
from visualizersos import Visualizer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(flush_secs=5, log_dir = f'logs/{ayarlar.MODEL_NAME}_{int(time.time())}')

visu = Visualizer()
env = SosEnv()
visu.env = env
agent = Agent(env, ayarlar.LOAD_MODEL)


suma = 0
draw = 0
winP1 = 0
winP2 = 0
for episode in tqdm(range(1, ayarlar.EPISODES + 1), ascii=True, unit='episodes'):

    env.start()
    p1history = []
    p2history = []

    p1history.clear()
    p2history.clear()

    done = False
    p1done = True
    p2done = True
    a = 0
    pl = 0

    while not done:
        if env.getLegalMoves(env.board).count(1) != 0:
            if a < 1:
                p1done, p2done, p1score = agent.moveP1(p1history, p1done, p2done)
                p1done, p2done, p2score = agent.moveP2(p2history, p1done, p2done)
                pl = 2
       
            elif pl == 1:
                while p2done:

                    p1done, p2done, p2score = agent.moveP2(p2history, p1done, p2done)
                    pl=2
                    
                    if env.getLegalMoves(env.board).count(1) == 0: 
                        
                        if p1score == p2score:
                            state, action, _, = p1history.pop()
                            p1history.append([state, action, 0])
                        
                            draw += 1
                            
                        else:
                            state, action, _, = p1history.pop()
                            p1history.append([state, action, -1])
                        
                            suma += -1
                            winP2 += 1
                            
                        done = True
                        p2done = False
                                    
            elif pl == 2:
                while p1done:

                    p1done, p2done, p1score = agent.moveP1(p1history, p1done, p2done)
                    pl=1 
               
                    if env.getLegalMoves(env.board).count(1) == 0:  
                        
                        if p1score == p2score:
                            state, action, _, = p1history.pop()
                            p1history.append([state, action, 0])
                        
                            draw += 1
                            
                        else:
                            state, action, _, = p2history.pop()
                            p2history.append([state, action, 1])
                        
                            suma += 1
                            winP1 += 1
                        
                        done = True
                        p1done = False
            
        a += 1


    agent.train(p1history, p2history)

    if not episode % ayarlar.AGGREGATE_STATS_EVERY or episode == 1:

        writer.add_scalar('sum', suma, episode)
        writer.add_scalar('epsilon', ayarlar.EPSILON, episode)
        writer.add_scalar('draw', draw, episode)
        writer.add_scalar('winP1', winP1, episode)
        writer.add_scalar('winP2', winP2, episode)

        if not ayarlar.IS_TEST:
            torch.save(agent.model.state_dict(), f'{ayarlar.MODEL_NAME}_{winP1}_{int(time.time())}')
        draw = 0
        suma = 0
        winP1 = 0
        winP2 = 0

    if ayarlar.EPSILON > ayarlar.MIN_EPSILON:
        ayarlar.EPSILON *= ayarlar.EPSILON_DECAY
        ayarlar.EPSILON = max(ayarlar.MIN_EPSILON, ayarlar.EPSILON)