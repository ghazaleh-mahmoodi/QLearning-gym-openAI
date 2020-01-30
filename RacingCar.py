import os

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adamax

import cv2

import gym
from gym import wrappers


# sns.set_style("ticks")
# sns.despine()

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    # plt.plot(running_avg)
    # plt.title("Running Average")
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.show()

env = gym.make("CarRacing-v0")
env = wrappers.Monitor(env, "train_1", force=True, mode='training')
# env = wrappers.Monitor(env, "larger_image", resume=True, mode='training')
env = wrappers.Monitor(env, "larger_image", resume=True, mode='evaluation')    




def transform(s):
    # We will crop the digits in the lower right corner, as they yield little 
    # information to our agent, as well as grayscale the frames.
    bottom_black_bar = s[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_b2 = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation=cv2.INTER_NEAREST)
    
    # We will crop the sides of the screen, so we have an 84x84 frame, and grayscale them:
    upper_field = s[:84, 6:90]
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation=cv2.INTER_NEAREST)
    upper_field_bw = upper_field_bw.astype('float')/255
    
    # The car occupies a very small space, we do the same preprocessing:
    car_field = s[66:78, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
    car_field_t = [car_field_bw[:, 3].mean()/255, 
                   car_field_bw[:, 4].mean()/255,
                   car_field_bw[:, 5].mean()/255, 
                   car_field_bw[:, 6].mean()/255]
    
    return bottom_black_bar_bw, upper_field_bw, car_field_t




# This function uses the bottom black bar of the screen and extracts the
# steering setting, speed, and gyroscope data.

def compute_steering_speed_gyro_abs(a):
    right_steering = a[6, 36:46].mean()/255
    left_steering = a[6, 26:36].mean()/255
    steering = (right_steering - left_steering + 1.0)/2
    
    left_gyro = a[6, 46:60].mean()/255
    right_gyro = a[6, 60:76].mean()/255
    gyro = (right_gyro - left_gyro + 1.0)/2
    
    speed = a[:, 0][:-2].mean()/255
    abs1 = a[:, 6][:-2].mean()/255
    abs2 = a[:, 8][:-2].mean()/255
    abs3 = a[:, 10][:-2].mean()/255
    abs4 = a[:, 12][:-2].mean()/255
    
    return [steering, speed, gyro, abs1, abs2, abs3, abs4]


vector_size = 10*10 + 7 + 4

def create_nn():
    if os.path.exists('race-car_larger.h5'):
        return load_model('race-car_larger.h5')
    
    model = Sequential()
    model.add(Dense(512, init='lecun_uniform', input_shape=(vector_size,)))  # 7x7+3 or 14x14+3
    model.add(Activation('relu'))
    
    model.add(Dense(11, init='lecun_uniform'))
    model.add(Activation('linear'))  # linear output so we can have a range of real-valued opts.
    
    model.compile(loss='mse', optimizer=Adamax())  # lr=0.001
    model.summary()
    
    return model


class Model:
    def __init__(self, env):
        self.env = env
        self.model = create_nn()  # One FFNN for all actions
        
    def predict(self, s):
        return self.model.predict(s.reshape(-1, vector_size), verbose=0)[0]
    
    def update(self, s, G):
        self.model.fit(s.reshape(-1, vector_size), 
                       np.array(G).reshape(-1, 11), 
                       nb_epoch=1, 
                       verbose=0)
        
    def sample_action(self, s, eps):
        qval = self.predict(s)
        if np.random.random() < eps:
            return random.randint(0,10), qval
        else:
            return np.argmax(qval), qval  




def convert_argmax_qval_to_env_action(output_value):
    # We reduce the action space to 
    
    gas = 0.0
    brake = 0.0
    steering = 0.0
    
    # Output value ranges from 0 to 10:
    
    if output_value <= 8:
        # Steering, brake, and gas are zero
        output_value -= 4
        steering = float(output_value)/4
    elif output_value >=9 and output_value <=9:
        output_value -= 8
        gas = float(output_value)/3  # 33% of gas
    elif output_value >= 10 and output_value <= 10:
        output_value -= 9
        brake = float(output_value)/2  # 50% of brake
    else:
        print("Error")  #Why?
        
    white = np.ones((round(gas * 100), 10))
    black = np.zeros((round(100 - gas * 100), 10))
    gas_display = np.concatenate((black, white)) * 255
    
    white = np.ones((round(brake * 100), 10))
    black = np.zeros((round(100 - brake * 100), 10))
    brake_display = np.concatenate((black, white)) * 255
    
    control_display = np.concatenate((brake_display, gas_display), axis=1)
    
    cv2.imshow('controls', control_display)
    cv2.waitKey(1)
    
    return [steering, gas, brake]




def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    full_reward_received = False
    totalreward = 0
    iters = 0
    while not done:
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(),
                               b.reshape(1, -1).flatten(),c), axis=0)  # 3+7*7 size vector, scaled in range 0-1
        # print ("state" , state )                      
        argmax_qval, qval = model.sample_action(state, eps)
        prev_state = state
        action = convert_argmax_qval_to_env_action(argmax_qval)
        observation, reward, done, info = env.step(action)
        
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(),
                               b.reshape(1,-1).flatten(), c), axis=0)
        
        # Update the model, standard Q-Learning TD(0)
        next_qval = model.predict(state)
        G = reward + gamma*np.max(next_qval)
        y = qval[:]
        y[argmax_qval] = G
        model.update(prev_state, y)
        totalreward += reward
        iters += 1
        
        if iters > 1600:
            print("This episode is stuck.")
            break
            
    return totalreward, iters


model = Model(env)
gamma = 0.99

N = 10000
totalrewards = np.empty(N)
costs = np.empty(N)

for n in range(N):
    eps = 0.5/np.sqrt(n+1+900)
    totalreward, iters = play_one(env, model, eps, gamma)
    totalrewards[n] = totalreward
    print("Episode: ", n, 
          ", iters: ", iters, 
          ", total reward: ", totalreward, 
          ", epsilon: ", eps, 
          ", average reward (of last 100): ", totalrewards[max(0,n-100):(n+1)].mean()
         )
    # We save the model every 10 episodes:
    if n%10 == 0:
        model.model.save('race-car_larger.h5')
        
print("Average reward for the last 100 episodes: ", totalrewards[-100:].mean())
print("Total steps: ", totalrewards.sum())

# plt.plot(totalrewards)
# plt.title("Rewards")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.show()

# plot_running_avg(totalrewards)

