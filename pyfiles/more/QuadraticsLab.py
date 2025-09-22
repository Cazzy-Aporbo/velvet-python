import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Settings
NUM_GRAPHS = 3
PASTEL_COLORS = ['#AEC6CF', '#FFB347', '#77DD77', '#FFD1DC']
X_RANGE = np.linspace(-10, 10, 500)
INITIAL_COEFFS = [[1, 0, 0], [0.5, -2, 1], [-1, 2, -1]]

# Create figure and subplots
fig, axes = plt.subplots(1, NUM_GRAPHS, figsize=(5*NUM_GRAPHS, 5))
plt.subplots_adjust(bottom=0.35)

lines = []
vertices = []
roots = []
symmetry_lines = []

for i in range(NUM_GRAPHS):
    a, b, c = INITIAL_COEFFS[i]
    y = a*X_RANGE**2 + b*X_RANGE + c
    ax = axes[i] if NUM_GRAPHS > 1 else axes
    ax.set_facecolor('#FDF6F0')
    line, = ax.plot(X_RANGE, y, color=PASTEL_COLORS[i], lw=2, label=f'y={a}x²+{b}x+{c}')
    vertex_dot, = ax.plot([], [], 'ro', markersize=8, label='Vertex')
    root_dot, = ax.plot([], [], 'go', markersize=6, label='Roots')
    symmetry_line = ax.axvline(0, color='purple', linestyle='--', lw=1, alpha=0.7, label='Axis of Symmetry')
    ax.set_ylim(-20, 20)
    ax.set_title(f'Graph {i+1}')
    ax.grid(True)
    ax.legend()
    lines.append(line)
    vertices.append(vertex_dot)
    roots.append(root_dot)
    symmetry_lines.append(symmetry_line)

sliders = []

for i in range(NUM_GRAPHS):
    axcolor = 'lightgray'
    ax_a = plt.axes([0.1, 0.25-0.05*i, 0.2, 0.03], facecolor=axcolor)
    ax_b = plt.axes([0.35, 0.25-0.05*i, 0.2, 0.03], facecolor=axcolor)
    ax_c = plt.axes([0.6, 0.25-0.05*i, 0.2, 0.03], facecolor=axcolor)
    slider_a = Slider(ax_a, f'a{i+1}', -5, 5, valinit=INITIAL_COEFFS[i][0])
    slider_b = Slider(ax_b, f'b{i+1}', -10, 10, valinit=INITIAL_COEFFS[i][1])
    slider_c = Slider(ax_c, f'c{i+1}', -10, 10, valinit=INITIAL_COEFFS[i][2])
    sliders.append([slider_a, slider_b, slider_c])

def update(val):
    for i in range(NUM_GRAPHS):
        a = sliders[i][0].val
        b = sliders[i][1].val
        c = sliders[i][2].val
        y = a*X_RANGE**2 + b*X_RANGE + c
        lines[i].set_ydata(y)
        xv = -b/(2*a) if a != 0 else 0
        yv = a*xv**2 + b*xv + c
        vertices[i].set_data([xv], [yv])
        symmetry_lines[i].set_xdata([xv, xv])
        disc = b**2 - 4*a*c
        if disc >= 0:
            r1 = (-b + np.sqrt(disc))/(2*a)
            r2 = (-b - np.sqrt(disc))/(2*a)
            roots[i].set_data([r1, r2], [0, 0])
        else:
            roots[i].set_data([], [])
        title = f'Graph {i+1} Vertex=({xv:.2f},{yv:.2f}) Roots='
        title += f'{r1:.2f},{r2:.2f}' if disc >= 0 else 'Complex'
        axes[i].set_title(title)
    fig.canvas.draw_idle()

for s in sliders:
    for slider in s:
        slider.on_changed(update)

resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color='lightgray', hovercolor='0.975')

def reset(event):
    for s in sliders:
        for slider in s:
            slider.reset()
button.on_clicked(reset)

update(None)
plt.show()