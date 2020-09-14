import gym
from gym.wrappers import Monitor

env = gym.make('CartPole-v1')
observation = env.reset()

for i in range(100):
    env.render()
    # 알고리즘1:
    # 막대기가 오른쪽으로 기울어져 있다면, 오른쪽으로 힘을 가하고
    # 그렇지 않다면, 왼쪽으로 힘을 가하기.
    if observation[2] > 0:
        action = 1
    else: action = 0

    observation, reward, done, info = env.step(action)
    print(observation, done)
    if done:
        print(i+1)
        break
env.close()
