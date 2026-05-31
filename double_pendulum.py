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

# Positions of pendulum masses
x1 = xc + L1*sym.sin(q[1])
y1 = -L1*sym.cos(q[1])

x2 = x1 + L2*sym.sin(q[2])
y2 = y1 - L2*sym.cos(q[2])

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
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# Extract accelerations and substitute F=0 (free swing)
xc_ddot_expr  = sol2[0][q_ddot[0]].subs(F, 0)
th1_ddot_expr = sol2[0][q_ddot[1]].subs(F, 0)
th2_ddot_expr = sol2[0][q_ddot[2]].subs(F, 0)

# Lambdify
args = [q[0], q[1], q[2], q_dot[0], q_dot[1], q_dot[2]]
xc_ddot_f  = sym.lambdify(args, xc_ddot_expr, 'numpy')
th1_ddot_f = sym.lambdify(args, th1_ddot_expr, 'numpy')
th2_ddot_f = sym.lambdify(args, th2_ddot_expr, 'numpy')

def dyn(t_val, s):
  return [s[3], s[4], s[5],
          xc_ddot_f(*s), th1_ddot_f(*s), th2_ddot_f(*s)]

# Initial conditions
state0 = [0.0, np.pi/4, np.pi/6, 0.0, 0.0, 0.0]

T = 10.0
dt = 0.02
t_eval = np.arange(0, T, dt)

print("Integrating equations of motion...")
sol = solve_ivp(dyn, (0, T), state0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)

# Compute positions for animation
xc_vals  = sol.y[0]
th1_vals = sol.y[1]
th2_vals = sol.y[2]

x1_vals = xc_vals + L1*np.sin(th1_vals)
y1_vals = -L1*np.cos(th1_vals)
x2_vals = x1_vals + L2*np.sin(th2_vals)
y2_vals = y1_vals - L2*np.cos(th2_vals)

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
    cart_patch.set_x(cx - cart_w/2)
    cart_patch.set_y(-cart_h/2)
    line1.set_data([cx, x1_vals[i]], [0, y1_vals[i]])
    line2.set_data([x1_vals[i], x2_vals[i]], [y1_vals[i], y2_vals[i]])
    trail_x.append(x2_vals[i])
    trail_y.append(y2_vals[i])
    trail.set_data(trail_x, trail_y)
    ax.set_xlim(cx - 4, cx + 4)
    return cart_patch, line1, line2, trail

ani = animation.FuncAnimation(fig, animate, frames=len(t_eval),
                              init_func=init, blit=False, interval=dt*1000)
plt.show()
