#Edison Sun
#5/30/26
import sympy as sym

t = sym.Symbol('t')

mc = 1   # cart mass
m1 = 1   # mass of first link
m2 = 1   # mass of second link
L1 = 1   # length of first link
L2 = 1   # length of second link

g = 9.8
F = sym.Symbol('F')  # force applied to cart

# Generalized coordinates: cart position, link angles are defined from vertical
xc = sym.Function('x')(t)
t1 = sym.Function(r'\theta_1')(t)
t2 = sym.Function(r'\theta_2')(t)

#define configuration vector
q = sym.Matrix([xc, t1, t2])

q_dot = q.diff(t)
q_ddot = q.diff(t).diff(t)

# Positions of pendulum masses (θ=0 is UP / inverted)
x1 = xc + L1*sym.sin(q[1])
y1 = L1*sym.cos(q[1])

x2 = x1 + L2*sym.sin(q[2])
y2 = y1 + L2*sym.cos(q[2])

# Velocities
xc_dot = xc.diff(t)
x1_dot = x1.diff(t)
y1_dot = y1.diff(t)
x2_dot = x2.diff(t)
y2_dot = y2.diff(t)

# Kinetic and potential energy
K = 0.5*mc*xc_dot**2 + 0.5*m1*(x1_dot**2+y1_dot**2) + 0.5*m2*(x2_dot**2+y2_dot**2)
U = m1*g*y1 + m2*g*y2

#Lagrangian
L_lag = K - U

# Euler-Lagrange equations


#maybe replace with jacobian func later
dLdq = []
for m in q:
  dLdq.append(L_lag.diff(m))
dLdq = sym.simplify(sym.Matrix(dLdq))

dLdq_dot = []
for m in q_dot:
  dLdq_dot.append(L_lag.diff(m))
dLdq_dot = sym.simplify(sym.Matrix(dLdq_dot))

ddt_dLdq_dot = dLdq_dot.diff(t)

#lhs of el
Q = sym.Matrix([F, 0, 0])

el = sym.Eq(ddt_dLdq_dot - dLdq, Q)

sol2 = sym.solve(el, q_ddot, dict=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from nmpc import nMPC

# Keep F as a lambdify argument so controller can supply it
xc_ddot_expr  = sol2[0][q_ddot[0]]
th1_ddot_expr = sol2[0][q_ddot[1]]
th2_ddot_expr = sol2[0][q_ddot[2]]

# Lambdify with F as 7th argument
args = [q[0], q[1], q[2], q_dot[0], q_dot[1], q_dot[2], F]
xc_ddot_f  = sym.lambdify(args, xc_ddot_expr, 'numpy')
th1_ddot_f = sym.lambdify(args, th1_ddot_expr, 'numpy')
th2_ddot_f = sym.lambdify(args, th2_ddot_expr, 'numpy')

def dyn(s, f_val):
  return np.array([s[3], s[4], s[5],
          xc_ddot_f(*s, f_val), th1_ddot_f(*s, f_val), th2_ddot_f(*s, f_val)])

# Initial conditions: start with larger perturbation (NMPC handles this)
state0 = np.array([0.0, 0.5, 0.4, 0.0, 0.0, 0.0])

T = 10.0
dt = 0.005
t_eval = np.arange(0, T, dt)

# NMPC parameters
N_horizon = 30           # prediction horizon steps
dt_mpc = 0.02            # coarser timestep for MPC prediction
Q_mpc = np.diag([1, 100, 100, 1, 10, 10])
R_mpc = np.array([[0.01]])
Q_f   = np.diag([1, 100, 100, 1, 10, 10]) * 5  # heavier terminal cost
F_MAX = 200.0

controller = nMPC(dyn, N_horizon, dt_mpc, Q_mpc, R_mpc, Q_f, F_MAX)

# Solve NMPC every mpc_every simulation steps (for speed)
mpc_every = max(1, int(dt_mpc / dt))

print("running runge-kutta 4 with NMPC control")
states = [state0.copy()]
forces = []
s = state0.copy()
f_val = 0.0

for i in range(len(t_eval) - 1):
    # Re-solve NMPC periodically
    if i % mpc_every == 0:
        f_val = controller.compute(s)
        if (i % (mpc_every * 10)) == 0:
            print(f"  t={t_eval[i]:.2f}s")
    forces.append(f_val)
    # RK4 step
    k1 = dyn(s, f_val)
    k2 = dyn(s + 0.5*dt*k1, f_val)
    k3 = dyn(s + 0.5*dt*k2, f_val)
    k4 = dyn(s + dt*k3, f_val)
    s = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    if np.any(np.isnan(s)) or np.any(np.abs(s) > 1e6):
        print(f"Simulation diverged at t={t_eval[i+1]:.3f}s, stopping.")
        break
    states.append(s.copy())

# Trim t_eval to match states
t_eval = t_eval[:len(states)]

states = np.array(states)
forces = np.array(forces)

# Compute positions for animation
xc_vals  = states[:, 0]
th1_vals = states[:, 1]
th2_vals = states[:, 2]

x1_vals = xc_vals + L1*np.sin(th1_vals)
y1_vals = L1*np.cos(th1_vals)
x2_vals = x1_vals + L2*np.sin(th2_vals)
y2_vals = y1_vals + L2*np.cos(th2_vals)

# Diagnostic plots
fig_diag, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(t_eval, np.degrees(th1_vals), label='θ₁')
axes[0].set_ylabel('θ₁ (deg)')
axes[0].axhline(0, color='gray', ls='--', lw=0.5)
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(t_eval, np.degrees(th2_vals), label='θ₂', color='green')
axes[1].set_ylabel('θ₂ (deg)')
axes[1].axhline(0, color='gray', ls='--', lw=0.5)
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(t_eval, xc_vals, label='x_cart', color='steelblue')
axes[2].set_ylabel('Cart x (m)')
axes[2].axhline(0, color='gray', ls='--', lw=0.5)
axes[2].legend(); axes[2].grid(True, alpha=0.3)

axes[3].plot(t_eval[:-1], forces, label='F', color='red')
axes[3].set_ylabel('Force (N)')
axes[3].set_xlabel('Time (s)')
axes[3].legend(); axes[3].grid(True, alpha=0.3)

fig_diag.suptitle('NMPC Control Diagnostics')
fig_diag.tight_layout()

# Animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
ax.set_ylim(-3, 2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Cart + Double Pendulum')
ax.grid(True, alpha=0.3)

cart_w, cart_h = 0.4, 0.2
cart_patch = patches.Rectangle((-cart_w/2, -cart_h/2), cart_w, cart_h,
              fc='steelblue', ec='black', lw=1.5)
ax.add_patch(cart_patch)
ax.axhline(0, color='gray', lw=0.5)

line1, = ax.plot([], [], 'o-', color='darkred', lw=3, markersize=8)
line2, = ax.plot([], [], 'o-', color='darkgreen', lw=3, markersize=8)
trail, = ax.plot([], [], '-', color='orange', lw=0.8, alpha=0.5)
trail_x, trail_y = [], []

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    trail.set_data([], [])
    return cart_patch, line1, line2, trail

def animate(i):
    cx = xc_vals[i]
    if np.isnan(cx) or np.isinf(cx):
        return cart_patch, line1, line2, trail
    cart_patch.set_x(cx - cart_w/2)
    cart_patch.set_y(-cart_h/2)
    line1.set_data([cx, x1_vals[i]], [0, y1_vals[i]])
    line2.set_data([x1_vals[i], x2_vals[i]], [y1_vals[i], y2_vals[i]])
    trail_x.append(x2_vals[i])
    trail_y.append(y2_vals[i])
    trail.set_data(trail_x, trail_y)
    ax.set_xlim(cx - 4, cx + 4)
    return cart_patch, line1, line2, trail

anim_step = max(1, int(0.02 / dt))
ani = animation.FuncAnimation(fig, animate, frames=range(0, len(t_eval), anim_step),
                              init_func=init, blit=False, interval=20)
plt.show()
