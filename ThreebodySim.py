import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

#Define constants
m1 = 1.
m2 = 1.
m3 = 1.
G = 1

#Define the initial conditions
#Feel free to play around with the initial conditions
x1, y1 = 1.0,  0.0
x2, y2 = -0.5,  0.866025
x3, y3 = -0.5, -0.866025

vx1, vy1 =  0.0,  0.5
vx2, vy2 = -0.433, -0.25
vx3, vy3 =  0.433, -0.25

#Define the state of the system (Three positions and three celocities each with x and y componenet)
state = np.array([
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [vx1, vy1],
    [vx2, vy2],
    [vx3, vy3]
])

#Define the derivatives function

def derivatives(t, state):
    r1, r2, r3 = state[0], state[1], state[2]
    v1, v2, v3 = state[3], state[4], state[5]

    #Write out the second order ODE for each of the particle

    a1 = -G * m2 * (r1 - r2) / np.linalg.norm(r1 - r2)**3 - (G * m3 * (r1 - r3) / np.linalg.norm(r1 - r3)**3)
    a2 = -G * m1 * (r2 -r1) / np.linalg.norm(r2 -r1)**3 - (G * m3 * (r2 - r3) / np.linalg.norm(r2 - r3)**3)
    a3  = - G * m1 * (r3 - r1) / np.linalg.norm(r3 - r1)**3 - ( G * m2 * (r3 -r2) / np.linalg.norm(r3-r2)**3)

    return np.array([
        v1, v2, v3,
        a1, a2, a3
    ])

#Implement the fourth order Runge-Kutta method
def RK4(t, state_n, h):
    t = np.arange(0, t+h, h)
    new_state = [state_n]

    for i in range(len(t) - 1):
        k1 = derivatives(t[i], new_state[i])
        k2 = derivatives(t[i] + h/2, new_state[i] + h * k1 / 2)
        k3 = derivatives(t[i] + h/2, new_state[i] + h * k2 /2)
        k4 = derivatives(t[i] + h, new_state[i] + h * k3)

        rk = new_state[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        new_state.append(rk)

    return np.array(new_state)

#Now, we extrract the positions

result = RK4(30, state, 0.01)

x1 = np.array([s[0][0] for s in result])
x2 = np.array([s[1][0] for s in result])
x3 = np.array([s[2][0] for s in result])

y1 = np.array([p[0][1] for p in result])
y2 = np.array([p[1][1] for p in result])
y3 = np.array([p[2][1] for p in result])

#Let's animate
#Create a subplot
fig, ax = plt.subplots(figsize= (8,8))

#create masses
mass1, = ax.plot(
    [],
    [],
    "o",
    color="tab:blue",
    markersize=10
)

mass2, = ax.plot(
    [],
    [],
    "o",
    color="tab:red",
    markersize = 10
)

mass3, = ax.plot(
    [],
    [],
    "o",
    color="tab:green",
    markersize=10
)

#INclude trails

trail1, = ax.plot(
    [],
    [],
    "--",
    color="tab:blue",
    alpha = 0.5
)

trail2,  = ax.plot(
    [],
    [],
    "--",
    color="tab:red",
    alpha = 0.5
)

trail3, = ax.plot(
    [],
    [],
    "--",
    color="tab:green",
    alpha = 0.5
)

#Set limits
xmin = min(x1.min(), x2.min(), x3.min())
xmax = max(x1.max(), x2.max(), x3.max())

ymin = min(y1.min(), y2.min(), y3.min())
ymax = max(y1.max(), y2.max(), y3.max())

center_x = (xmin + xmax) / 2
center_y = (ymin + ymax) / 2

span = max(xmax - xmin, ymax - ymin)

ax.set_xlim(center_x - span/2 - 1, center_x + span/2 + 1)
ax.set_ylim(center_y - span/2 - 1, center_y + span/2 + 1)

ax.set_aspect("equal")

#Give title and labels
ax.set_title(
    "Three Body Problem",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "x position",
    fontsize= 12
)

ax.set_ylabel(
    "y position",
    fontsize=12
)

#Now update frame by frame

def update(frame):
    mass1.set_data(
        [x1[frame]],
        [y1[frame]]
    )

    mass2.set_data(
        [x2[frame]],
        [y2[frame]]
    )

    mass3.set_data(
        [x3[frame]],
        [y3[frame]]
    )

    trail1.set_data(
        [x1[:frame+1]],
        [y1[:frame+1]]
    )

    trail2.set_data(
        [x2[:frame+1]],
        [y2[:frame+1]]
    )

    trail3.set_data(
        [x3[:frame+1]],
        [y3[:frame +1]]
    )

    return [
        mass1,
        mass2,
        mass3,
        trail1,
        trail2,
        trail3
    ]

animation = FuncAnimation(
    fig,
    update,
    frames=len(result),
    interval=10,
    blit = True
)

plt.show()

#Energy Drift plot

#Extract the velocities

vx1 = np.array([s[3][0] for s in result])
vx2 = np.array([s[4][0] for s in result])
vx3 = np.array([s[5][0] for s in result])

vy1 = np.array([p[3][1] for p in result])
vy2 = np.array([p[4][1] for p in result])
vy3 = np.array([p[5][1] for p in result])

#Compute kinetic energies

ke1 = 0.5 * m1 * (vx1**2 + vy1**2)
ke2 =  0.5 * m2 * (vx2**2 + vy2**2)
ke3 = 0.5 * m3 * (vx3**2 + vy3**2)

total_ke = ke1 + ke2 + ke3

#Compute potential energies

r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
r13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
r23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)

pe1 = -G * m1 * m2 / r12
pe2 = -G * m3 * m2 / r23
pe3 = -G * m1 * m3 / r13

total_pe = pe1 + pe2 + pe3

total_energy = total_ke + total_pe

energy_drift = total_energy - total_energy[0] / abs(total_energy[0])

plt.figure(figsize=(8, 5))

plt.plot(np.arange(len(result)) * 0.01, energy_drift)

plt.xlabel("Time")
plt.ylabel("Relative energy drift")
plt.title("Energy drift in RK4 of the three-body system")
plt.grid(True, alpha=0.3)

plt.show()


#Special intial conditions
# Figure 8 orbit: 
# x1, y1 = -0.97000436,0.24308753
# x2, y2 = 0.97000436, -0.24308753
# x3, y3 = 0., 0.

# vx1, vy1 = 0.466203685, 0.43236573
# vx2, vy2 = 0.466203685,  0.43236573
# vx3, vy3 = -0.93240737, -0.86473146

#Symmetric triangle:dd
# x1, y1 = 1.0,  0.0
# x2, y2 = -0.5,  0.866025
# x3, y3 = -0.5, -0.866025

# vx1, vy1 = 0., 0.
# vx2, vy2 = 0., 0.
# vx3, vy3 = 0., 0.

#Approximately circular arrangement
# x1, y1 = 1.0,  0.0
# x2, y2 = -0.5,  0.866025
# x3, y3 = -0.5, -0.866025

# vx1, vy1 =  0.0,  0.5
# vx2, vy2 = -0.433, -0.25
# vx3, vy3 =  0.433, -0.25

#Asymmetric condition:
# x1, y1 = -3.0, 0.0
# x2, y2 =  0.0, 0.2
# x3, y3 = 1.0, 3.0

# vx1, vy1 = 0.0,  0.3
# vx2, vy2 = 0.2, -0.1
# vx3, vy3 = -0.2, 0.0

#Linear bodies
# x1, y1 = -2.0, 0.0
# x2, y2 =  0.0, 0.0
# x3, y3 = 2.0, 0.0

# vx1, vy1 = 0, 0
# vx2, vy2 = 0, 0
# vx3, vy3 = 0, 0

