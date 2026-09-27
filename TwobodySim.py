import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Define fixed values in the system

m1 = 1
m2 = 1
G = 1

#Define the initial conditions
x1, y1 = -1, 0
x2, y2 =  0, 0.5

vx1, vy1 = 0, -0.5
vx2, vy2 = 0,  0.5

#Define the state
state = np.array(
    [
        [x1, y1], 
        [x2, y2], 
        [vx1, vy1], 
        [vx2, vy2]
    ]
        )

#Define the derivative functions

def derivatives(t, state):
    r1 = state[0]
    r2 = state[1]

    v1 = state[2]
    v2 = state[3]

    a1  = G * m2 * (r2 - r1) / np.linalg.norm(r2 - r1)**3
    a2  = -G * m1 * (r2 - r1) / np.linalg.norm(r2 - r1)**3

    return np.array([
        v1, 
        v2,
        a1,
        a2 
    ])
#Define the Runge-Kutta Fourth order function
def RK4(t, state_n, h):
    t = np.arange(0, t + h, h)
    state_ = [state_n]
    for i in range(len(t) - 1):
        k1 = derivatives(t[i], state_[i])
        k2 = derivatives(t[i] + h / 2, state_[i] + h * k1 / 2)
        k3 = derivatives(t[i] + h  / 2, state_[i] + h * k2 /2)
        k4 = derivatives(t[i] + h , state_[i] + h * k3)

        n_state = state_[i] + h * (k1 + 2 * k2 + 2* k3 + k4) /6
        state_.append(n_state)
    return state_

#Extract the position vectors

result = RK4(100, state, 0.01)
x1_values = [s[0][0] for s in result]
x2_values = [s[1][0] for s in result]
y1_values = [s[0][1] for s in result]
y2_values = [s[1][1] for s in result]

#Animating the motion
#Create a subplot
fig, ax = plt.subplots(figsize = (8, 8))

#Set axis limits
ax.set_xlim(
    min(min(x1_values), min(x2_values)) - 1,
    max(max(x1_values), max(x2_values) + 1
        )
)

ax.set_ylim(
    min(min(y1_values), min(y2_values)) -1,
    max(max(y1_values), max(y2_values)) + 1  
)

#Define the masses
mass1, = ax.plot(
    [],
    [],
    "o",
    color = "tab:blue",
    markersize = 10
)

mass2, = ax.plot(
    [],
    [],
    "o",
    color = "tab:red",
    markersize= 10
)

#Add trails
trail1, = ax.plot(
    [],
    [],
    "--",
    color="tab:blue",
    alpha= 0.5
)

trail2, = ax.plot(
    [],
    [],
    "--",
    color="tab:red",
    alpha = 0.5
)

#Inclue the title and axis labels
ax.set_title(
    "Two Body Problem",
    fontsize = 16,
    fontweight = "bold"
)

ax.set_xlabel(
    "x Position",
    fontsize = 12
)
ax.set_ylabel(
    "y Position", 
    fontsize = 12
)

#Create animation update function

def update(i):
    mass1.set_data(
        [x1_values[i]],
        [y1_values[i]]
    )

    mass2.set_data(
        [x2_values[i]],
        [y2_values[i]]
    )

    trail1.set_data(
        [x1_values[:i + 1]],
        [y1_values[:i + 1]]
    )

    trail2.set_data(
        [x2_values[:i+1]],
        [y2_values[:i+1]]
    )


    return [
        mass1,
        mass2,
        trail1,
        trail2
    ]

animation = FuncAnimation(
    fig,
    update,
    frames=len(result),
    interval = 10,
    blit = True
)

plt.show()