import gym
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
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 10
initial_games = 10000

def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # TODO: currently always sets robot to the same position
        rc = RobotCar()
        y_start = random.uniform(-2.0, 2.0)
        theta_start = random.uniform(-45, 45)
        rc.set_robot(0.0, y_start, theta_start)
        rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)

        x_trajectory = []
        y_trajectory = []

        # this is each frame, up to 200...but we wont make it that far.
        for t in range(300):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            #env.render()
            
            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            #action = env.action_space.sample()
            speed = 1.0
            steer = random.uniform(-np.pi/4.0, np.pi/4.0)
            
            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            rc.move(steer, speed)
            x_trajectory.append(rc.x)
            y_trajectory.append(rc.y)
            #observation, reward, done, info = env.step(action)
            #print(observation)
            print(steer)
            #if done:
            #    break
        return x_trajectory, y_trajectory
                
def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        rc = RobotCar()
        y_start = random.uniform(-2.0, 2.0)
        theta_start = random.uniform(-45, 45)
        rc.set_robot(0.0, y_start, theta_start)
        rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)

        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            #action = random.randrange(0,2)
            action = random.uniform(-np.pi/4.0, np.pi/4.0)
            speed = 1.0
            # do it!
            rc.move(action, speed)
            #print(action)
            observation = [rc.x, rc.y, rc.orientation]
            
            dist =(rc.y**2)**0.5 
            done = dist > 30
            if (dist < 0.5):
                reward = 1
            else:
                reward = 0
            #observation, reward, done, info = env.step(action)
            #print(reward)
            
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break
        print(score)

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here. 
        # all we're doing is reinforcing the score, we're not trying 
        # to influence the machine in any way as to HOW that score is 
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                # needs to be continuous
                #if data[1] == 1:
                #    output = [0,1]
                #elif data[1] == 0:
                #    output = [1,0]
                    
                # saving our training data
                training_data.append([data[0], [data[1]]]) #changed from ([data[0], output])

        # reset env to play again
        #env.reset()
        # save overall scores
        scores.append(score)
    
    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    # some stats here, to further illustrate the neural network magic!
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
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model



training_data = initial_population()
#training_data = np.load('saved.npy', allow_pickle=True)
model = train_model(training_data)
scores = []
choices = []
#for each_game in range(10):
fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
for title in ["IC1", "IC2", "IC1 Noisy", "IC2 Noisy"]:
    score = 0
    game_memory = []
    prev_obs = []

    x_trajectory = []
    y_trajectory = []
    #     env.reset()
    rc = RobotCar()
    #y_start = random.uniform(-2.0, 2.0)
    #theta_start = random.uniform(-45, 45)
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
            #action = random.randrange(0,2)
        else:
            #action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            action = model.predict(np.array(prev_obs).reshape(-1,len(prev_obs),1))[0]
        speed = 1.0
        print(action)
        # do it!
        rc.move(action, speed)
        
        choices.append(action)
                
        #new_observation, reward, done, info = env.step(action)
        rc.move(action, speed)
        new_observation = [rc.x, rc.y, rc.orientation]
            
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

    #scores.append(score)

# print('Average Score:',sum(scores)/len(scores))
# print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
# print(score_requirement)


# test of random game
# x_traj, y_traj = some_random_games_first()
# fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
# n = len(x_traj)
# print(n)
# ax1.plot(x_traj, y_traj, 'g', label='Random Controller')
# ax1.plot(x_traj, np.zeros(n), 'r', label='reference')
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.legend()
# plt.show()

#ic1 = [0.0, 1.0, 30.0]
#ic2 = [0.0, -1.0, -30.0]
#[X, Y, Theta]