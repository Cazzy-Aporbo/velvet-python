"""
HYPERDIMENSIONAL ATTRACTOR DYNAMICS ENGINE 2025
Advanced attractor visualization with flowing gradients and particle dynamics
Features: Strange attractors, dimensional folding, particle flows, field interactions
"""

import colorsys
import random
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


# Generate massive color gradient palette
def generate_advanced_gradient(n_colors=256):
    """Generate sophisticated multi-dimensional color gradient"""
    colors = []
    for i in range(n_colors):
        # Multiple gradient waves for complexity
        t = i / n_colors

        # Primary wave (rainbow)
        h1 = t
        # Secondary wave (shifted)
        h2 = (t + 0.33) % 1.0
        # Tertiary wave (reverse)
        h3 = (1 - t * 0.5) % 1.0

        # Combine waves with varying saturation and value
        h = (h1 * 0.5 + h2 * 0.3 + h3 * 0.2) % 1.0
        s = 0.7 + 0.3 * np.sin(t * np.pi * 4)
        v = 0.6 + 0.4 * np.cos(t * np.pi * 3)

        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)

    return colors

# Create ultra-vibrant color maps
GRADIENT_COLORS = generate_advanced_gradient(256)

class AttractorSystem:
    """Advanced attractor system with multiple interacting wells"""

    def __init__(self):
        self.attractors = []
        self.particles = []
        self.field_resolution = 100
        self.time = 0

        # Initialize attractor field
        self.x = np.linspace(-5, 5, self.field_resolution)
        self.y = np.linspace(-5, 5, self.field_resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Multiple attractor types
        self.attractor_types = ['lorenz', 'rossler', 'henon', 'duffing', 'vanderpol']

        # Initialize attractors
        self._initialize_attractors()

        # Initialize particles
        self._initialize_particles()

        # Field values
        self.potential_field = np.zeros((self.field_resolution, self.field_resolution))
        self.velocity_field_x = np.zeros_like(self.potential_field)
        self.velocity_field_y = np.zeros_like(self.potential_field)
        self.color_field = np.zeros((*self.potential_field.shape, 3))

    def _initialize_attractors(self):
        """Create multiple dynamic attractors"""
        n_attractors = 7
        for i in range(n_attractors):
            angle = 2 * np.pi * i / n_attractors
            radius = 2.5 + np.random.random() * 1

            attractor = {
                'position': np.array([radius * np.cos(angle), radius * np.sin(angle)]),
                'strength': np.random.uniform(0.5, 2.0),
                'frequency': np.random.uniform(0.1, 0.5),
                'phase': np.random.random() * 2 * np.pi,
                'type': random.choice(self.attractor_types),
                'color_index': int(i * 256 / n_attractors),
                'rotation': 0,
                'depth': np.random.uniform(-1, 1)
            }
            self.attractors.append(attractor)

    def _initialize_particles(self):
        """Create particle swarm"""
        n_particles = 500
        for _ in range(n_particles):
            particle = {
                'position': np.random.uniform(-4, 4, 2),
                'velocity': np.random.randn(2) * 0.1,
                'color_index': np.random.randint(0, 256),
                'trail': deque(maxlen=20),
                'age': 0,
                'energy': 1.0
            }
            self.particles.append(particle)

    def calculate_potential(self, x, y, t):
        """Calculate complex potential field"""
        potential = 0

        for attractor in self.attractors:
            # Dynamic attractor position
            pos = attractor['position'] + 0.5 * np.array([
                np.sin(t * attractor['frequency'] + attractor['phase']),
                np.cos(t * attractor['frequency'] * 1.3 + attractor['phase'])
            ])

            # Distance to attractor
            dx = x - pos[0]
            dy = y - pos[1]
            r = np.sqrt(dx**2 + dy**2 + 0.1)

            # Different attractor types create different potentials
            if attractor['type'] == 'lorenz':
                # Lorenz-like spiral
                potential += attractor['strength'] * np.sin(r * 2) / (r + 0.5)
                potential += 0.5 * np.cos(np.arctan2(dy, dx) * 3 + t) / (r + 1)

            elif attractor['type'] == 'rossler':
                # Rossler-like folding
                potential += attractor['strength'] * (np.sin(r * 3) * np.cos(r)) / (r + 0.3)

            elif attractor['type'] == 'henon':
                # Henon-like sharp wells
                potential += attractor['strength'] * np.exp(-r**2 / 2) * np.cos(r * 5)

            elif attractor['type'] == 'duffing':
                # Duffing double well
                potential += attractor['strength'] * (r**2 - 2) * np.exp(-r/3)

            else:  # vanderpol
                # Van der Pol oscillation
                potential += attractor['strength'] * np.sin(r * 4 + t) * np.exp(-r/4)

            # Add rotating component
            potential += 0.2 * np.sin(np.arctan2(dy, dx) * 2 + t * 0.5)

        # Add global waves
        potential += 0.3 * np.sin(x * 0.5 + t) * np.cos(y * 0.5 - t)
        potential += 0.2 * np.sin(np.sqrt(x**2 + y**2) * 2 - t * 2)

        return potential

    def update_fields(self, t):
        """Update all field values"""
        self.time = t

        # Calculate potential at each grid point
        for i in range(self.field_resolution):
            for j in range(self.field_resolution):
                self.potential_field[i, j] = self.calculate_potential(
                    self.X[i, j], self.Y[i, j], t
                )

        # Smooth the field
        self.potential_field = gaussian_filter(self.potential_field, sigma=0.5)

        # Calculate gradients (velocity field)
        gy, gx = np.gradient(self.potential_field)
        self.velocity_field_x = -gx
        self.velocity_field_y = -gy

        # Calculate color field based on multiple factors
        for i in range(self.field_resolution):
            for j in range(self.field_resolution):
                # Base color from potential
                potential_norm = (self.potential_field[i, j] + 3) / 6
                color_idx = int(potential_norm * 255) % 256

                # Add influence from nearby attractors
                x, y = self.X[i, j], self.Y[i, j]
                for attractor in self.attractors:
                    dist = np.sqrt((x - attractor['position'][0])**2 +
                                 (y - attractor['position'][1])**2)
                    if dist < 2:
                        influence = np.exp(-dist**2 / 2)
                        color_idx = int((color_idx + attractor['color_index'] * influence) /
                                       (1 + influence)) % 256

                # Add time-based color evolution
                color_idx = (color_idx + int(t * 10)) % 256

                self.color_field[i, j] = GRADIENT_COLORS[color_idx]

        # Update attractors
        for attractor in self.attractors:
            attractor['rotation'] += attractor['frequency'] * 0.1
            attractor['depth'] = np.sin(t * attractor['frequency']) * 0.5

    def update_particles(self, dt):
        """Update particle dynamics"""
        for particle in self.particles:
            x, y = particle['position']

            # Get field values at particle position
            if -5 <= x <= 5 and -5 <= y <= 5:
                # Interpolate velocity from field
                ix = int((x + 5) * (self.field_resolution - 1) / 10)
                iy = int((y + 5) * (self.field_resolution - 1) / 10)

                ix = np.clip(ix, 0, self.field_resolution - 1)
                iy = np.clip(iy, 0, self.field_resolution - 1)

                vx = self.velocity_field_x[iy, ix]
                vy = self.velocity_field_y[iy, ix]

                # Update velocity with field influence
                particle['velocity'] = 0.9 * particle['velocity'] + 0.1 * np.array([vx, vy])

                # Add random walk component
                particle['velocity'] += np.random.randn(2) * 0.01

                # Update position
                particle['position'] += particle['velocity'] * dt

                # Store trail
                particle['trail'].append(particle['position'].copy())

                # Update color based on velocity
                speed = np.linalg.norm(particle['velocity'])
                particle['color_index'] = (particle['color_index'] + int(speed * 50)) % 256

                # Age and energy
                particle['age'] += dt
                particle['energy'] = 0.5 + 0.5 * np.sin(particle['age'] * 0.5)

            # Boundary conditions - wrap around
            if abs(particle['position'][0]) > 5:
                particle['position'][0] = -particle['position'][0] * 0.9
            if abs(particle['position'][1]) > 5:
                particle['position'][1] = -particle['position'][1] * 0.9


class HyperdimensionalVisualizer:
    """Advanced visualization system for attractor dynamics"""

    def __init__(self, figsize=(20, 12)):
        self.fig = plt.figure(figsize=figsize, facecolor='#000000')

        # Create layout - larger main view
        gs = self.fig.add_gridspec(2, 3, hspace=0.2, wspace=0.2,
                                  left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Main 3D attractor field
        self.ax_main = self.fig.add_subplot(gs[:, 0:2], projection='3d')

        # Side panels
        self.ax_flow = self.fig.add_subplot(gs[0, 2])
        self.ax_phase = self.fig.add_subplot(gs[1, 2])

        self._style_axes()

        # Initialize system
        self.system = AttractorSystem()
        self.time = 0

    def _style_axes(self):
        """Apply dark styling with better visibility"""
        # 3D axis
        self.ax_main.set_facecolor('#000000')
        self.ax_main.xaxis.pane.fill = False
        self.ax_main.yaxis.pane.fill = False
        self.ax_main.zaxis.pane.fill = False
        self.ax_main.grid(True, alpha=0.1)

        # 2D axes
        for ax in [self.ax_flow, self.ax_phase]:
            ax.set_facecolor('#000011')
            ax.grid(True, alpha=0.1, color='#333333')

    def update_visualization(self, frame):
        """Update all visualizations"""
        self.time = frame * 0.05

        # Update system
        self.system.update_fields(self.time)
        self.system.update_particles(0.05)

        # Clear and redraw
        self._clear_axes()
        self._render_all()

    def _clear_axes(self):
        """Clear all axes"""
        self.ax_main.clear()
        self.ax_flow.clear()
        self.ax_phase.clear()
        self._style_axes()

    def _render_all(self):
        """Render all components"""
        self._render_main_field()
        self._render_flow_field()
        self._render_phase_space()

    def _render_main_field(self):
        """Render main 3D attractor field with full color"""
        self.ax_main.set_title('Hyperdimensional Attractor Dynamics',
                               color='#FFFFFF', fontsize=16)

        # Create colored surface based on potential and color field
        Z = self.system.potential_field

        # Create face colors from color field
        face_colors = np.zeros((self.system.field_resolution-1,
                                self.system.field_resolution-1, 4))

        for i in range(self.system.field_resolution-1):
            for j in range(self.system.field_resolution-1):
                # Average color of vertices
                color = self.system.color_field[i, j]
                # Add transparency based on height
                alpha = 0.8 + 0.2 * np.tanh(Z[i, j] / 2)
                face_colors[i, j] = [color[0], color[1], color[2], alpha]

        # Plot surface with dynamic colors
        surf = self.ax_main.plot_surface(
            self.system.X, self.system.Y, Z,
            facecolors=face_colors,
            shade=True,
            antialiased=True,
            rstride=2, cstride=2,
            alpha=0.9
        )

        # Add contour lines at different heights
        for level in [-2, -1, 0, 1, 2]:
            contour_color = GRADIENT_COLORS[int((level + 3) * 40) % 256]
            self.ax_main.contour(self.system.X, self.system.Y, Z,
                                levels=[level], colors=[contour_color],
                                linewidths=1, alpha=0.7, offset=level)

        # Render particles as glowing points
        for particle in self.system.particles[::5]:  # Sample for performance
            x, y = particle['position']
            if -5 <= x <= 5 and -5 <= y <= 5:
                # Get height from field
                ix = int((x + 5) * (self.system.field_resolution - 1) / 10)
                iy = int((y + 5) * (self.system.field_resolution - 1) / 10)
                ix = np.clip(ix, 0, self.system.field_resolution - 1)
                iy = np.clip(iy, 0, self.system.field_resolution - 1)
                z = self.system.potential_field[iy, ix]

                # Particle color
                color = GRADIENT_COLORS[particle['color_index']]

                # Draw particle with glow
                self.ax_main.scatter(x, y, z + 0.1,
                                   s=20 * particle['energy'],
                                   c=[color], alpha=0.9,
                                   marker='o', edgecolors='white',
                                   linewidth=0.5)

                # Draw trail
                if len(particle['trail']) > 2:
                    trail_points = np.array(list(particle['trail']))
                    trail_z = []
                    for tp in trail_points:
                        tix = int((tp[0] + 5) * (self.system.field_resolution - 1) / 10)
                        tiy = int((tp[1] + 5) * (self.system.field_resolution - 1) / 10)
                        tix = np.clip(tix, 0, self.system.field_resolution - 1)
                        tiy = np.clip(tiy, 0, self.system.field_resolution - 1)
                        trail_z.append(self.system.potential_field[tiy, tix])

                    self.ax_main.plot(trail_points[:, 0], trail_points[:, 1], trail_z,
                                    color=color, alpha=0.3, linewidth=1)

        # Render attractor centers
        for attractor in self.system.attractors:
            pos = attractor['position'] + 0.5 * np.array([
                np.sin(self.time * attractor['frequency'] + attractor['phase']),
                np.cos(self.time * attractor['frequency'] * 1.3 + attractor['phase'])
            ])

            # Get attractor color
            color = GRADIENT_COLORS[attractor['color_index']]

            # Draw attractor core
            self.ax_main.scatter(pos[0], pos[1], attractor['depth'] * 2,
                               s=200 * attractor['strength'],
                               c=[color], alpha=0.6,
                               marker='*', edgecolors='white',
                               linewidth=1)

            # Draw influence sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            radius = attractor['strength'] * 0.5

            sphere_x = pos[0] + radius * np.outer(np.cos(u), np.sin(v))
            sphere_y = pos[1] + radius * np.outer(np.sin(u), np.sin(v))
            sphere_z = attractor['depth'] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

            self.ax_main.plot_surface(sphere_x, sphere_y, sphere_z,
                                     color=color, alpha=0.1)

        # Set limits and labels
        self.ax_main.set_xlim(-5, 5)
        self.ax_main.set_ylim(-5, 5)
        self.ax_main.set_zlim(-4, 4)
        self.ax_main.set_xlabel('X', color='#FFFFFF')
        self.ax_main.set_ylabel('Y', color='#FFFFFF')
        self.ax_main.set_zlabel('Potential', color='#FFFFFF')

        # Set view angle for better visibility
        self.ax_main.view_init(elev=25, azim=self.time * 10)

    def _render_flow_field(self):
        """Render vector flow field with colors"""
        self.ax_flow.set_title('Vector Flow Field', color='#FFFFFF', fontsize=12)

        # Sample the field for arrows
        step = 5
        x_sample = self.system.x[::step]
        y_sample = self.system.y[::step]
        X_sample, Y_sample = np.meshgrid(x_sample, y_sample)

        U = self.system.velocity_field_x[::step, ::step]
        V = self.system.velocity_field_y[::step, ::step]

        # Calculate magnitude for colors
        M = np.sqrt(U**2 + V**2)

        # Create color map based on magnitude
        colors = []
        for i in range(len(x_sample)):
            for j in range(len(y_sample)):
                mag_norm = M[j, i] / (M.max() + 0.001)
                color_idx = int(mag_norm * 255) % 256
                colors.append(GRADIENT_COLORS[color_idx])

        # Plot arrows with colors
        self.ax_flow.quiver(X_sample, Y_sample, U, V,
                           color=colors, alpha=0.7,
                           scale=30, width=0.003)

        # Add streamlines
        strm = self.ax_flow.streamplot(self.system.x, self.system.y,
                                      self.system.velocity_field_x.T,
                                      self.system.velocity_field_y.T,
                                      color=M.T, cmap='rainbow',
                                      density=1, linewidth=1, alpha=0.5)

        # Show attractor positions
        for attractor in self.system.attractors:
            color = GRADIENT_COLORS[attractor['color_index']]
            self.ax_flow.scatter(attractor['position'][0],
                               attractor['position'][1],
                               s=100 * attractor['strength'],
                               c=[color], alpha=0.8,
                               marker='o', edgecolors='white',
                               linewidth=1)

        self.ax_flow.set_xlim(-5, 5)
        self.ax_flow.set_ylim(-5, 5)
        self.ax_flow.set_aspect('equal')
        self.ax_flow.set_xlabel('X', color='#FFFFFF')
        self.ax_flow.set_ylabel('Y', color='#FFFFFF')

    def _render_phase_space(self):
        """Render phase space portrait with trajectories"""
        self.ax_phase.set_title('Phase Space Portrait', color='#FFFFFF', fontsize=12)

        # Plot particle trajectories in phase space
        for i, particle in enumerate(self.system.particles[::10]):
            if len(particle['trail']) > 2:
                trail = np.array(list(particle['trail']))

                # Use position and velocity for phase space
                color = GRADIENT_COLORS[particle['color_index']]

                # Plot trajectory
                self.ax_phase.plot(trail[:, 0], trail[:, 1],
                                 color=color, alpha=0.5, linewidth=0.5)

                # Current position
                self.ax_phase.scatter(particle['position'][0],
                                    particle['position'][1],
                                    s=10, c=[color], alpha=0.8)

        # Add attractor basins
        theta = np.linspace(0, 2*np.pi, 50)
        for attractor in self.system.attractors:
            color = GRADIENT_COLORS[attractor['color_index']]
            radius = attractor['strength']

            x_circle = attractor['position'][0] + radius * np.cos(theta)
            y_circle = attractor['position'][1] + radius * np.sin(theta)

            self.ax_phase.fill(x_circle, y_circle, color=color, alpha=0.1)
            self.ax_phase.plot(x_circle, y_circle, color=color, alpha=0.3, linewidth=1)

        self.ax_phase.set_xlim(-5, 5)
        self.ax_phase.set_ylim(-5, 5)
        self.ax_phase.set_aspect('equal')
        self.ax_phase.set_xlabel('Position X', color='#FFFFFF')
        self.ax_phase.set_ylabel('Position Y', color='#FFFFFF')

    def animate(self):
        """Start animation"""
        def update(frame):
            try:
                self.update_visualization(frame)
            except Exception as e:
                print(f"Frame {frame} error: {e}")
            return []

        self.anim = animation.FuncAnimation(
            self.fig, update,
            frames=2000,
            interval=50,
            blit=False,
            repeat=True
        )

        plt.tight_layout()
        plt.show()


def launch_attractor_engine():
    """Launch the Hyperdimensional Attractor Dynamics Engine"""
    print()
    print("HYPERDIMENSIONAL ATTRACTOR DYNAMICS ENGINE")
    print("Advanced Multi-Attractor System with Full Color Gradients")
    print()
    print()
    print("FEATURES:")
    print("• Multiple interacting strange attractors (Lorenz, Rössler, Hénon, etc.)")
    print("• 256-color smooth gradient system with dynamic evolution")
    print("• 500+ particles flowing through attractor fields")
    print("• Real-time potential field calculations")
    print("• Vector flow fields with streamlines")
    print("• Phase space portraits")
    print("• Rotating 3D perspective")
    print("• Full-spectrum color mapping across entire surface")
    print()
    print("Initializing hyperdimensional space...")

    visualizer = HyperdimensionalVisualizer()
    visualizer.animate()


if __name__ == "__main__":
    launch_attractor_engine()
