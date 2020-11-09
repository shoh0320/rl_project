import gym
import numpy as np
import random

EPSILON=0.1
TIMES=10000000
ALPHA=0.1
GAMMA=0.9

env = gym.make("FrozenLake-v0")

def action_select(state, q_function):
    actions = q_function[state]
    rand_prob = random.random()
    action = 0
    
    max_val = np.max(actions)
    # max인 인덱스 저장
    max_list = [idx for idx, i in enumerate(actions) if i==max_val]
    # greedy
    if rand_prob >= EPSILON:
        action = random.choice(max_list)
    # non-greedy
    else:
        actions_idx= np.delete([0,1,2,3], max_list)
        if len(actions_idx)==0:
            action = 0
        else:
            action = random.choice(actions_idx)
        
    return action
    q_function = [[0, 0, 0, 0] for i in range(env.observation_space.n)]

observation = env.reset()

for t in range(TIMES):
    action = action_select(observation, q_function)
    current_q = q_function[observation][action]

    next_observation, reward, done, info = env.step(action)
    
    next_action = action_select(next_observation, q_function)   
    next_q = q_function[next_observation][next_action]
    q_function[observation][action] = current_q + ALPHA*(reward + GAMMA*next_q - current_q)
    
    if done:
        observation = env.reset()
        
    observation = next_observation
        
print(np.array(q_function, dtype="U").reshape(4,4,4))

def printprint(states):
    states = np.round(states, 9)
    print("      ",states[0][3],"                 ", states[1][3],"                 ",  states[2][3],"                 ", states[3][3])
    print(states[0][0],"   ", states[0][2], " ",
          states[1][0],"   ", states[1][2], " ",
          states[2][0],"   ", states[2][2], " ",
          states[3][0],"   ", states[3][2])
    print("      ",states[0][1],"                  ", states[1][1],"                 ", states[2][1],"                 ", states[3][1])
    print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    
printprint(q_function[0:4])
printprint(q_function[4:8])
printprint(q_function[8:12])
printprint(q_function[12:16])
