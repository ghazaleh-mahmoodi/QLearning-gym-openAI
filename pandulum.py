
import numpy as array
import math
import gym
from gym import wrappers
import time

iteration_max = 7000
test_max = 1000

min_learning_rate = 0.03
eps = 0.2

def obs_to_state(obs):
    """ Maps an observation to state """
    state_cos_theta = int(array.digitize(obs[0], Sample_cos_theta))
    state_sin_theta = int(array.digitize(obs[1], Sample_sin_theta))
    state_theta_dot = int(array.digitize(obs[2], Sample_theta_dot))
    return (state_cos_theta, state_sin_theta, state_theta_dot)

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(0)
    array.random.seed(0)
    print ('----- Start Learning -----')
    
    Sample_cos_theta = array.around(array.arange(env.observation_space.low[0], env.observation_space.high[0], 0.1), 1)[1:]
    Sample_sin_theta = Sample_cos_theta
    Sample_theta_dot = array.around(array.arange(env.observation_space.low[2], env.observation_space.high[2], 1), 0)[1:]

    Sample_out = array.around(array.arange(-2, 2.2, 0.2), 1)

    q_state_table = array.zeros((len(Sample_cos_theta) + 1, len(Sample_cos_theta) + 1, len(Sample_cos_theta) + 1, len(Sample_out)))
    
    scores = []
    for i in range(iteration_max):
        obs = env.reset()
        new_state = obs_to_state(obs)
        total_reward = 0

        alpha = max(min_learning_rate, 1.0 * (0.85 ** (i//100)))
        
        for j in range(test_max):
            current_state = new_state

            #select action : random or using q_state(best action from current state)

            if array.random.random() < eps:
                action_index = array.random.randint(len(Sample_out))
            
            else:
                action_index = array.argmax(q_state_table[current_state])
            
            # map index to action value
            
            action = Sample_out[action_index]  
            obs, reward, done, _ = env.step([action])
            total_reward += reward

            new_state = obs_to_state(obs)
            # q_state_table[current_state][action_index] += alpha * (reward +  array.max(q_state_table[new_state]) - q_state_table[current_state][action_index])
            q_state_table[current_state][action_index] = q_state_table[current_state][action_index] *(1 - alpha) + alpha * (reward + array.max(q_state_table[new_state]))
            
            if done:
                break
        
        if i % 100 == 0:
            print('Iteration %d -- alpha = %f Total reward = %d.' %(i, alpha, total_reward))    

    scores.append(total_reward)
    
    frames = []
    obs = env.reset()
    while True:
        frames.append(env.render(mode = 'rgb_array'))
        state = obs_to_state(obs)
        action_idx = array.argmax(q_state_table[state])
        obs, reward, done, _ = env.step([Sample_out[action_idx]])  # conversion index to value
        if done:
            print "done"
            time.sleep(1)
            break

    env.close()