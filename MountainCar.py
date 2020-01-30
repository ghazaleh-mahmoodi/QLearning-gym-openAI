import numpy as array

import gym
from gym import wrappers

n_states = 40
iteration_max = 5000

min_learning_rate = 0.03
test_max = 1000
eps = 0.01

def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    array.random.seed(0)
    print ('----- Start Learning -----')
    
    q_state_table = array.zeros((n_states, n_states, 3))
    
    for i in range(iteration_max):
        
        obs = env.reset()
        # alpha: learning rate is decreased at each step
        alpha = max(min_learning_rate, 1.0 * (0.85 ** (i//100)))
        
        for j in range(test_max):
            
            a, b = obs_to_state(env, obs)
            
            # select action : random or using q_state(best action from current state)

            if array.random.uniform(0, 1) < eps:
                action = array.random.choice(env.action_space.n)

            else:
                logits = q_state_table[a][b]
                logits_exp = array.exp(logits)
                probs = logits_exp / array.sum(logits_exp)
                action = array.random.choice(env.action_space.n, p=probs)
            
            obs, reward, done, _ = env.step(action)

            # update q table
            a_, b_ = obs_to_state(env, obs)
            q_state_table[a][b][action] = (1 - alpha) * q_state_table[a][b][action] + alpha * (reward +  array.max(q_state_table[a_][b_]))
            
            if done:
                break
        
        if i % 100 == 0:
            print('Iteration %d -- alpha = %f ' %(i, alpha,))
    
    solution_policy = array.argmax(q_state_table, axis=2)

    frames = []
    obs = env.reset()
    while True:
        frames.append(env.render(mode = 'rgb_array'))
        a, b = obs_to_state(env,obs)
        action = solution_policy[a][b]
        obs, reward, done, _ = env.step(action)  
        if done:
            print "done"
            break

    env.close()