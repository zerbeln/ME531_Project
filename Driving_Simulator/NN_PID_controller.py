import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from driving_sim import RobotCar
import matplotlib.pyplot as plt


LR = 1e-3
goal_steps = 500
initial_games = 30000

y_start_range = [-2.0, 2.0]
theta_start_range = [-45, 45]
score_requirement = 150



def pid_controller(rc, k_p, k_d, k_i, speed=1.0):
    """
        Helper function that gets next action with PID controller
    """
    prev_cte = rc.y
    sum_err = 0.0  # cross-track error sum (integral term)
    sum_err += rc.y
    d_error = rc.y - prev_cte  # Derivative of cross-track error
    prev_cte = rc.y
    steer = -(k_p*rc.y) - (k_d*d_error) - (k_i*sum_err)

    return steer


def initial_population(noise):
    """
        Generate training data
    """
    training_data = [] # [observations, actions]
    scores = [] # all scores:
    accepted_scores = [] # scores that met our threshold:
    
    for _ in range(initial_games):
        
        # Generate new problem in starting region
        rc = RobotCar()
        y_start = random.uniform(y_start_range[0], y_start_range[1])
        theta_start = random.uniform(theta_start_range[0], theta_start_range[1])
        rc.set_robot(0.0, y_start, theta_start)
        if noise:
            rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)
        else:
            rc.set_noise(0.0, 0.0, 0.0)

        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):
            # get PID controller recommended action
            action = pid_controller(rc, 0.2, 3.0, 0.004)
            # get random adujstment to action in the range -1 to +1 degree
            adjustment = random.uniform(-(0.5/180)*np.pi, (0.5/180)*np.pi)
            steer = action + adjustment
            speed = 1.0
        
            # take action and observe the results
            rc.move(steer, speed)
            observation = [rc.y, rc.orientation] #don't include x
            
            dist =(rc.y**2)**0.5 
   
            if (dist < 0.5):
                reward = 1
            else:
                reward = 0
        
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, steer])
            prev_observation = observation
            score+=reward

        # IF our score is higher than our threshold, we save
        # every move we made
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0], [data[1]]])

        scores.append(score)
    
    # to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    # some stats 
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 1, activation='linear') 
    network = regression(network, optimizer='adam', learning_rate=LR, loss='mean_square', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning2')
    return model

if __name__ == '__main__':

    training_data = initial_population(True) # Change to false to run without noise
    #training_data = np.load('saved.npy', allow_pickle=True)
    model = train_model(training_data)
    rc = RobotCar()
    rc.set_robot(0.0, 1, 30)
    rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi) # Change to (0.0, 0.0, 0.0) to run without noise
    x_trajectory = []
    y_trajectory = []
    prev_obs = [rc.y, rc.orientation]
    while rc.x<199:
        x_trajectory.append(rc.x)
        y_trajectory.append(rc.y)

        action = model.predict(np.array(prev_obs).reshape(-1,len(prev_obs),1))[0]
        speed = 1.0
                    
        rc.move(action, speed)
        new_observation = [rc.y, rc.orientation]
        prev_obs = new_observation

    print(x_trajectory)
    x_trajectory = np.array(x_trajectory)
    y_trajectory = np.array(y_trajectory)
    np.save("IC1PID_x_noise.npy", x_trajectory) # remove _noise to run without noise
    np.save("IC1PID_y_noise.npy", y_trajectory) # remove _noise to run without noise

    rc = RobotCar()
    rc.set_robot(0.0, -1, -30)
    rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi) # Change to (0.0, 0.0, 0.0) to run without noise
    x_trajectory = []
    y_trajectory = []
    prev_obs = [rc.y, rc.orientation]
    while rc.x<199:

        x_trajectory.append(rc.x)
        y_trajectory.append(rc.y)

        action = model.predict(np.array(prev_obs).reshape(-1,len(prev_obs),1))[0]
        speed = 1.0
                    
        rc.move(action, speed)
        new_observation = [rc.y, rc.orientation]
        prev_obs = new_observation

    print(x_trajectory)
    x_trajectory = np.array(x_trajectory)
    y_trajectory = np.array(y_trajectory)
    np.save("IC2PID_x_noise.npy", x_trajectory) # remove _noise to run without noise
    np.save("IC2PID_y_noise.npy", y_trajectory) # remove _noise to run without noise





