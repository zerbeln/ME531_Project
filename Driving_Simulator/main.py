from driving_sim import RobotCar
import matplotlib.pyplot as plt
import numpy as np


def pid_controller(k_p, k_d, k_i, n_epochs=200, speed=1.0):
    rc = RobotCar()
    rc.set_robot(0.0, 1.0, 30.0)
    rc.set_noise((1/180)*np.pi, 0.0, (10/180)*np.pi)

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x_traj, y_traj = pid_controller(0.2, 3.0, 0.004)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    n = len(x_traj)
    print(n)
    ax1.plot(x_traj, y_traj, 'g', label='PID controller')
    ax1.plot(x_traj, np.zeros(n), 'r', label='reference')
    plt.legend()
    plt.show()
