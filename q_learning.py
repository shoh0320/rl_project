import gym
import numpy as np
import random
import os
import time

ALPHA=0.4
GAMMA=0.9
EPSILON=0.1
TIMES = 10000000

env = gym.make("Taxi-v3")

def action_select(state, q_function, mode="s"):
    actions = q_function[state]
    rand_prob = random.random()
    action = 0
    
    max_val = np.max(actions)
    # max인 인덱스 저장
    max_list = [idx for idx, i in enumerate(actions) if i==max_val]
    # greedy 또는 큐러닝의 경우 가장 큰 큐함수값의 idx
    if rand_prob >= EPSILON or mode=="q":
        action = random.choice(max_list)
    # non-greedy
    else:
        actions_idx= np.delete([0,1,2,3, 4, 5], max_list)
        if len(actions_idx)==0:
            action = 0
        else:
            action = random.choice(actions_idx)
        
    return action
    
q_function = [[0, 0, 0, 0, 0, 0] for i in range(env.observation_space.n)]
observation = env.reset()

for t in range(TIMES):
    action = action_select(observation, q_function)
    current_q = q_function[observation][action]

    next_observation, reward, done, info = env.step(action)
    
    next_action = action_select(next_observation, q_function, "q")   
    next_q = q_function[next_observation][next_action]
    q_function[observation][action] = current_q + ALPHA*(reward + GAMMA*next_q - current_q)
    
    if done:
        observation = env.reset()
        pass
        
    observation = next_observation

print(np.array(q_function, dtype="U"))

observation = env.reset()
done = False

while not done:
    action = action_select(observation, q_function, "q")
    observation, _, done, _ = env.step(action)
    
    os.system('cls')
    env.render()
    time.sleep(0.7)
