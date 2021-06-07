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
score_requirement = 20
initial_games = 30000

                
def initial_population(noise):
    
    training_data = [] # [OBS, MOVES]
    scores = [] # all scores
    accepted_scores = [] # scores that met our threshold:

    for _ in range(initial_games):

        # Generate new problem in starting region
        rc = RobotCar()
        y_start = random.uniform(-2.0, 2.0)
        theta_start = random.uniform(-45, 45)
        rc.set_robot(0.0, y_start, theta_start)
        if noise:
            rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)
        else:
            rc.set_noise(0.0, 0.0, 0.0)

        score = 0
        game_memory = []
        prev_observation = []
        prev_x = rc.x
        
        for _ in range(goal_steps):
            
            # choose random action 
            action = random.uniform(-np.pi/4.0, np.pi/4.0)
            speed = 1.0
        
            # take action and observe the results
            rc.move(action, speed)
            observation = [rc.y, rc.orientation] #rcx

            
            dist =(rc.y**2)**0.5 
            #done = dist > 30
            done = prev_x > rc.x
            prev_x = rc.x
            if (dist < 0.5):
                reward = 1
            else:
                reward = 0
        
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # saving our training data
                training_data.append([data[0], [data[1]]]) 

        # save overall scores
        scores.append(score)
    
    # to reference later
    training_data_save = np.array(training_data)
    np.save('saved2.npy',training_data_save)
    
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

    network = fully_connected(network, 1, activation='linear') #changed from 2, softmax
    network = regression(network, optimizer='adam', learning_rate=LR, loss='mean_square', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


if __name__ == '__main__':

    training_data = initial_population()
    #training_data = np.load('saved2.npy', allow_pickle=True)
    model = train_model(training_data)
    scores = []
    choices = []
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    for title in ["IC1", "IC2", "IC1 Noisy", "IC2 Noisy"]:
        score = 0
        game_memory = []
        prev_obs = []

        x_trajectory = []
        y_trajectory = []
        rc = RobotCar()
        if title == "IC1":
            rc.set_robot(0.0, 1, 30)
            rc.set_noise(0.0, 0.0, 0.0)
            color = "g"
        elif title == "IC2":
            rc.set_robot(0.0, -1, -30)
            rc.set_noise(0.0, 0.0, 0.0)
            color = "blue"
        elif title =="IC1 Noisy":
            rc.set_robot(0.0, 1, 30)
            rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)
            color="purple"
        else:
            rc.set_robot(0.0, -1, -30)
            rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)
            color="red"
        for _ in range(200):

            x_trajectory.append(rc.x)
            y_trajectory.append(rc.y)
    #         env.render()

            if len(prev_obs)==0:
                action = random.uniform(-np.pi/4.0, np.pi/4.0)
            else:
                action = model.predict(np.array(prev_obs).reshape(-1,len(prev_obs),1))[0]
            speed = 1.0
            rc.move(action, speed)
            choices.append(action)
                    
            rc.move(action, speed)
            new_observation = [rc.y, rc.orientation]
                
            dist =(rc.y**2)**0.5 
            done = dist > 25
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            if (dist < 0.5):
                reward = 1
            else:
                reward = 0
            score+=reward
            #if done: break
        
        n = len(x_trajectory)
        print(n)
        ax1.plot(x_trajectory, y_trajectory, color, label=title)
    ax1.plot(range(-300,300), np.zeros(600), 'black', label='reference')
    plt.title("NN Driving controller")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()

   