from driving_sim import RobotCar
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np

import NN_driving_controller as NN



def pid_controller(initial_pos, noise, k_p, k_d, k_i, n_epochs=200, speed=1.0):
    rc = RobotCar()
    rc.set_robot(initial_pos[0], initial_pos[1], initial_pos[2])
    rc.set_noise(noise[0], noise[1], noise[2])

    x_trajectory = []
    y_trajectory = []

    prev_cte = rc.y
    sum_err = 0.0  # cross-track error sum (integral term)
    for i in range(n_epochs):
        sum_err += rc.y
        d_error = rc.y - prev_cte  # Derivative of cross-track error
        prev_cte = rc.y
        steer = -(k_p*rc.y) - (k_d*d_error) - (k_i*sum_err)
        rc.move(steer, speed)
        x_trajectory.append(rc.x)
        y_trajectory.append(rc.y)

    return x_trajectory, y_trajectory


if __name__ == '__main__':

    print("Run NN_PID_controller.py first")

    IC1 = [0.0, 1, 30]
    IC2 = [0, -1, -30]
    noise = [(1/180)*np.pi, 0.0, (10/180)*np.pi] # change to [0.0, 0.0, 0.0] to run without noise

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    l0 = ax1.plot(range(200), np.zeros(200), color='black')
    
    #PID
    x_traj, y_traj = pid_controller(IC1, noise, 0.2, 3.0, 0.004)
    l1 = ax1.plot(x_traj, y_traj, color='g')
    x_traj, y_traj = pid_controller(IC2, noise, 0.2, 3.0, 0.004)
    ax1.plot(x_traj, y_traj,  "--", color="g")
    
    # NN Random
    training_data = NN.initial_population(noise=True) # change to noise = False to run without noise
    model = NN.train_model(training_data)

    
    # IC1
    x_traj = []
    y_traj = []
    rc = RobotCar()
    rc.set_robot(IC1[0], IC1[1], IC1[2])
    rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)
    prev_obs = [rc.y, rc.orientation]
    for _ in range(200):
            x_traj.append(rc.x)
            y_traj.append(rc.y)
            action = model.predict(np.array(prev_obs).reshape(-1,len(prev_obs),1))[0]
            speed = 1.0
            rc.move(action, speed)
            new_observation = [rc.y, rc.orientation] 
            if rc.x < 0:
                break
    l2 = ax1.plot(x_traj, y_traj, color='r')
    # IC2 
    x_traj = []
    y_traj = []
    rc = RobotCar()
    rc.set_robot(IC2[0], IC2[1], IC2[2])
    rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)
    prev_obs = [rc.y, rc.orientation]
    for _ in range(200):
            x_traj.append(rc.x)
            y_traj.append(rc.y)
            action = model.predict(np.array(prev_obs).reshape(-1,len(prev_obs),1))[0]
            speed = 1.0
            rc.move(action, speed)
            new_observation = [rc.y, rc.orientation] #rcx?
            if rc.x < 0:
                break
    ax1.plot(x_traj, y_traj, "--", color='r')

    # NN PID
    # Data prodcued by and loaded from NN_PID_controller.py to avoid issues with having multiple TFLearn sessions
    x_traj = np.load("IC1PID_x_noise.npy", allow_pickle=True)  # Remove _noise to get data without noise
    y_traj = np.load("IC1PID_y_noise.npy", allow_pickle=True)
    l3 = ax1.plot(x_traj, y_traj, color='blue')
    x_traj = np.load("IC2PID_x_noise.npy", allow_pickle=True)
    y_traj = np.load("IC2PID_y_noise.npy", allow_pickle=True)
    ax1.plot(x_traj, y_traj, "--", color='blue')
    
    ax1.legend([l0[0],l1[0], l2[0], l3[0]], ["Reference", "PID", "Random NN", "PID NN"], title="Controller", loc="center right")
    l4, =ax1.plot([0], [0], color="black")
    l5, =ax1.plot([0], [0], "--",color="black")
    leg = Legend(ax1, [l4, l5], ["(0, 1, 30)", "(0, -1, -30)"], title="Initial Position", loc="lower right") 
    ax1.add_artist(leg)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Vehicle Path with Classical and Learning-Based Controllers")
    plt.show()
