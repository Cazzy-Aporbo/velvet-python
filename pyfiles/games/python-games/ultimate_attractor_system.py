"""
ULTIMATE PHD-LEVEL HYPERDIMENSIONAL ATTRACTOR SYSTEM
Complete dynamical systems workstation with full explainability
Features: 12 analysis panels, real-time explanations, interactive legends,
statistical measures, chaos indicators, and comprehensive tooltips
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
import colorsys
from collections import deque
import random

# Generate ultra-sophisticated color palette
def generate_scientific_gradient(n_colors=256):
    """Generate scientifically meaningful color gradient"""
    colors = []
    for i in range(n_colors):
        t = i / n_colors
        
        # Multi-frequency interference for complex patterns
        h = (np.sin(t * np.pi * 2) * 0.3 + 
             np.cos(t * np.pi * 3) * 0.3 + 
             t) % 1.0
        s = 0.85 + 0.15 * np.sin(t * np.pi * 4)
        v = 0.75 + 0.25 * np.cos(t * np.pi * 2)
        
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    
    return colors

GRADIENT_COLORS = generate_scientific_gradient(256)

# Explanatory text for each phenomenon
EXPLANATIONS = {
    'chaos': "CHAOS: System exhibits sensitive dependence on initial conditions",
    'stable': "STABLE: System converges to fixed point or limit cycle",
    'bifurcation': "BIFURCATION: Qualitative change in dynamics at critical parameter",
    'attractor': "ATTRACTOR: Region of phase space toward which system evolves",
    'lyapunov_positive': "λ > 0: Exponential divergence indicates chaos",
    'lyapunov_negative': "λ < 0: Convergence indicates stability",
    'poincare': "POINCARÉ: Cross-section reveals periodic structure",
    'fourier': "FOURIER: Frequency decomposition shows dominant oscillations",
    'entropy_high': "HIGH ENTROPY: System explores many states (complex)",
    'entropy_low': "LOW ENTROPY: System confined to few states (ordered)",
    'energy_conserved': "ENERGY CONSERVED: Hamiltonian dynamics",
    'energy_dissipated': "ENERGY DISSIPATED: Non-conservative forces present"
}

class UltimateAttractorSystem:
    """Ultimate attractor system with comprehensive analysis"""
    
    def __init__(self):
        self.field_resolution = 100
        self.time = 0
        self.dt = 0.05
        
        # Coordinate system
        self.x = np.linspace(-5, 5, self.field_resolution)
        self.y = np.linspace(-5, 5, self.field_resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Attractor types with descriptions
        self.attractor_info = {
            'lorenz': {'name': 'Lorenz', 'description': 'Butterfly attractor - weather chaos'},
            'rossler': {'name': 'Rössler', 'description': 'Minimal chaos generator'},
            'henon': {'name': 'Hénon', 'description': 'Discrete-time strange attractor'},
            'duffing': {'name': 'Duffing', 'description': 'Driven oscillator with double well'},
            'vanderpol': {'name': 'Van der Pol', 'description': 'Self-sustained oscillations'},
            'chua': {'name': 'Chua', 'description': 'Electronic circuit chaos'},
            'burke_shaw': {'name': 'Burke-Shaw', 'description': 'Geophysical fluid dynamics'}
        }
        
        # Initialize system components
        self.attractors = []
        self._initialize_attractors()
        
        self.particles = []
        self._initialize_particles()
        
        # Field arrays
        self.potential_field = np.zeros((self.field_resolution, self.field_resolution))
        self.velocity_field_x = np.zeros_like(self.potential_field)
        self.velocity_field_y = np.zeros_like(self.potential_field)
        self.divergence_field = np.zeros_like(self.potential_field)
        self.curl_field = np.zeros_like(self.potential_field)
        
        # Analysis metrics storage
        self.lyapunov_history = deque(maxlen=300)
        self.energy_history = deque(maxlen=300)
        self.entropy_history = deque(maxlen=300)
        self.poincare_points = deque(maxlen=1000)
        self.fourier_spectrum = np.zeros(50)
        self.correlation_dimension = 0
        self.hurst_exponent = 0
        self.mutual_information = 0
        self.kolmogorov_complexity = 0
        
        # System state
        self.system_state = 'initializing'
        self.dominant_behavior = 'transient'
        self.current_explanation = ''
        
    def _initialize_attractors(self):
        """Create diverse attractor ensemble"""
        attractor_types = list(self.attractor_info.keys())
        n_attractors = 7
        
        for i in range(n_attractors):
            angle = 2 * np.pi * i / n_attractors
            radius = 2.5 + 0.5 * np.random.random()
            
            attractor_type = attractor_types[i % len(attractor_types)]
            
            attractor = {
                'position': np.array([radius * np.cos(angle), radius * np.sin(angle)]),
                'strength': np.random.uniform(0.5, 2.0),
                'frequency': np.random.uniform(0.1, 0.5),
                'phase': np.random.random() * 2 * np.pi,
                'type': attractor_type,
                'info': self.attractor_info[attractor_type],
                'color_index': int(i * 256 / n_attractors),
                'rotation': 0,
                'depth': np.random.uniform(-1, 1),
                # Attractor-specific parameters
                'sigma': 10.0,   # Lorenz
                'rho': 28.0,     # Lorenz
                'beta': 8/3,     # Lorenz
                'a': 0.2,        # Rössler
                'b': 0.2,        # Rössler
                'c': 5.7,        # Rössler
            }
            self.attractors.append(attractor)
    
    def _initialize_particles(self):
        """Create particle ensemble with tracking"""
        n_particles = 500
        
        for i in range(n_particles):
            particle = {
                'id': i,
                'position': np.random.uniform(-4, 4, 2),
                'velocity': np.random.randn(2) * 0.1,
                'acceleration': np.zeros(2),
                'color_index': np.random.randint(0, 256),
                'trail': deque(maxlen=50),
                'age': 0,
                'energy': 1.0,
                'lyapunov_sum': 0,
                'trajectory': deque(maxlen=200),
                'nearest_neighbor': None,
                'cluster_id': -1
            }
            self.particles.append(particle)
    
    def calculate_potential(self, x, y, t):
        """Calculate multi-scale potential field"""
        potential = 0
        
        for attractor in self.attractors:
            # Dynamic attractor position
            pos = attractor['position'] + 0.5 * np.array([
                np.sin(t * attractor['frequency'] + attractor['phase']),
                np.cos(t * attractor['frequency'] * 1.3 + attractor['phase'])
            ])
            
            dx = x - pos[0]
            dy = y - pos[1]
            r = np.sqrt(dx**2 + dy**2 + 0.1)
            
            # Type-specific potential
            if attractor['type'] == 'lorenz':
                # Lorenz-like spiral structure
                potential += attractor['strength'] * np.sin(r * 2) / (r + 0.5)
                potential += 0.5 * np.cos(np.arctan2(dy, dx) * 3 + t) / (r + 1)
                
            elif attractor['type'] == 'rossler':
                # Rössler folding
                potential += attractor['strength'] * (np.sin(r * 3) * np.cos(r)) / (r + 0.3)
                
            elif attractor['type'] == 'henon':
                # Hénon sharp transitions
                potential += attractor['strength'] * np.exp(-r**2 / 2) * np.cos(r * 5)
                
            elif attractor['type'] == 'duffing':
                # Duffing double-well
                potential += attractor['strength'] * (r**2 - 2) * np.exp(-r/3)
                
            elif attractor['type'] == 'vanderpol':
                # Van der Pol relaxation oscillations
                potential += attractor['strength'] * np.sin(r * 4 + t) * np.exp(-r/4)
                
            elif attractor['type'] == 'chua':
                # Chua's circuit nonlinearity
                potential += attractor['strength'] * np.tanh(r) * np.sin(np.arctan2(dy, dx) * 5)
                
            elif attractor['type'] == 'burke_shaw':
                # Burke-Shaw vortex
                potential += attractor['strength'] * np.exp(-r/2) * (dx * dy) / (r**2 + 1)
            
            # Add rotation and mixing
            potential += 0.2 * np.sin(np.arctan2(dy, dx) * 2 + t * 0.5)
        
        # Multi-scale waves
        potential += 0.3 * np.sin(x * 0.5 + t) * np.cos(y * 0.5 - t)
        potential += 0.2 * np.sin(np.sqrt(x**2 + y**2) * 2 - t * 2)
        potential += 0.1 * np.cos(x * y * 0.3 + t * 0.7)
        
        return potential
    
    def update_fields(self, t):
        """Update all fields with advanced analysis"""
        self.time = t
        
        # Calculate potential field
        for i in range(self.field_resolution):
            for j in range(self.field_resolution):
                self.potential_field[i, j] = self.calculate_potential(
                    self.X[i, j], self.Y[i, j], t
                )
        
        # Apply smoothing
        self.potential_field = gaussian_filter(self.potential_field, sigma=0.5)
        
        # Calculate velocity field (negative gradient)
        gy, gx = np.gradient(self.potential_field)
        self.velocity_field_x = -gx
        self.velocity_field_y = -gy
        
        # Calculate divergence (source/sink detection)
        div_x = np.gradient(self.velocity_field_x, axis=1)
        div_y = np.gradient(self.velocity_field_y, axis=0)
        self.divergence_field = div_x + div_y
        
        # Calculate curl (vorticity)
        curl_x = np.gradient(self.velocity_field_y, axis=1)
        curl_y = np.gradient(self.velocity_field_x, axis=0)
        self.curl_field = curl_x - curl_y
        
        # Perform advanced analysis
        self._calculate_lyapunov_spectrum()
        self._calculate_system_energy()
        self._calculate_information_measures()
        self._update_poincare_section()
        self._calculate_fourier_spectrum()
        self._calculate_correlation_dimension()
        self._detect_system_state()
        
        # Update attractors
        for attractor in self.attractors:
            attractor['rotation'] += attractor['frequency'] * 0.1
            attractor['depth'] = np.sin(t * attractor['frequency']) * 0.5
    
    def update_particles(self, dt):
        """Update particles with full dynamics"""
        for particle in self.particles:
            x, y = particle['position']
            
            if -5 <= x <= 5 and -5 <= y <= 5:
                # Get field values at particle position
                ix = int((x + 5) * (self.field_resolution - 1) / 10)
                iy = int((y + 5) * (self.field_resolution - 1) / 10)
                ix = np.clip(ix, 0, self.field_resolution - 1)
                iy = np.clip(iy, 0, self.field_resolution - 1)
                
                # Get velocity from field
                vx = self.velocity_field_x[iy, ix]
                vy = self.velocity_field_y[iy, ix]
                
                # Calculate acceleration
                ax = (vx - particle['velocity'][0]) / dt
                ay = (vy - particle['velocity'][1]) / dt
                particle['acceleration'] = np.array([ax, ay])
                
                # Update velocity with damping
                particle['velocity'] = 0.95 * particle['velocity'] + 0.05 * np.array([vx, vy])
                
                # Add stochastic component
                particle['velocity'] += np.random.randn(2) * 0.01
                
                # Update position
                particle['position'] += particle['velocity'] * dt
                
                # Store full trajectory data
                particle['trail'].append(particle['position'].copy())
                particle['trajectory'].append({
                    'pos': particle['position'].copy(),
                    'vel': particle['velocity'].copy(),
                    'acc': particle['acceleration'].copy(),
                    'time': self.time,
                    'energy': 0.5 * np.linalg.norm(particle['velocity'])**2
                })
                
                # Update properties
                speed = np.linalg.norm(particle['velocity'])
                particle['color_index'] = (particle['color_index'] + int(speed * 50)) % 256
                particle['energy'] = 0.5 * speed**2 + self.potential_field[iy, ix]
                particle['age'] += dt
                
                # Find nearest neighbor
                self._update_nearest_neighbor(particle)
            
            # Boundary handling
            if abs(particle['position'][0]) > 5:
                particle['position'][0] *= -0.9
                particle['velocity'][0] *= -0.5
            if abs(particle['position'][1]) > 5:
                particle['position'][1] *= -0.9
                particle['velocity'][1] *= -0.5
    
    def _update_nearest_neighbor(self, particle):
        """Find nearest neighbor for Lyapunov calculation"""
        min_dist = float('inf')
        nearest = None
        
        for other in self.particles[:50]:  # Sample for efficiency
            if other['id'] != particle['id']:
                dist = np.linalg.norm(particle['position'] - other['position'])
                if dist < min_dist:
                    min_dist = dist
                    nearest = other
        
        particle['nearest_neighbor'] = nearest['id'] if nearest else None
    
    def _calculate_lyapunov_spectrum(self):
        """Calculate full Lyapunov spectrum"""
        if len(self.particles) < 10:
            return
        
        # Sample particle pairs
        lyapunov_values = []
        
        for i in range(0, min(20, len(self.particles)-1), 2):
            p1 = self.particles[i]
            p2 = self.particles[i+1]
            
            separation = np.linalg.norm(p1['position'] - p2['position'])
            
            if separation > 0.001:
                # Local Lyapunov exponent
                lyap = np.log(separation / 0.001) / (self.time + 0.001)
                lyapunov_values.append(lyap)
        
        if lyapunov_values:
            # Store average Lyapunov
            avg_lyapunov = np.mean(lyapunov_values)
            self.lyapunov_history.append({
                'mean': avg_lyapunov,
                'std': np.std(lyapunov_values),
                'max': np.max(lyapunov_values),
                'min': np.min(lyapunov_values)
            })
    
    def _calculate_system_energy(self):
        """Calculate detailed energy budget"""
        # Kinetic energy
        kinetic = sum(0.5 * np.linalg.norm(p['velocity'])**2 for p in self.particles)
        
        # Potential energy
        potential = 0
        for p in self.particles:
            ix = int((p['position'][0] + 5) * (self.field_resolution - 1) / 10)
            iy = int((p['position'][1] + 5) * (self.field_resolution - 1) / 10)
            ix = np.clip(ix, 0, self.field_resolution - 1)
            iy = np.clip(iy, 0, self.field_resolution - 1)
            potential += self.potential_field[iy, ix]
        
        # Field energy
        field_energy = np.sum(self.potential_field**2) / self.field_resolution**2
        
        total_energy = kinetic + potential + field_energy
        
        self.energy_history.append({
            'total': total_energy,
            'kinetic': kinetic,
            'potential': potential,
            'field': field_energy,
            'ratio': kinetic / (potential + 0.001)
        })
    
    def _calculate_information_measures(self):
        """Calculate information theory metrics"""
        # Shannon entropy of particle distribution
        positions = np.array([p['position'] for p in self.particles])
        H, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], 
                                           bins=25, range=[[-5, 5], [-5, 5]])
        
        # Normalize and calculate entropy
        H_norm = H / (H.sum() + 1e-10)
        H_norm = H_norm[H_norm > 0]
        shannon_entropy = -np.sum(H_norm * np.log2(H_norm + 1e-10))
        
        # Mutual information between x and y coordinates
        H_x, _ = np.histogram(positions[:, 0], bins=25)
        H_y, _ = np.histogram(positions[:, 1], bins=25)
        H_x = H_x / (H_x.sum() + 1e-10)
        H_y = H_y / (H_y.sum() + 1e-10)
        
        H_x = H_x[H_x > 0]
        H_y = H_y[H_y > 0]
        
        entropy_x = -np.sum(H_x * np.log2(H_x + 1e-10))
        entropy_y = -np.sum(H_y * np.log2(H_y + 1e-10))
        
        self.mutual_information = entropy_x + entropy_y - shannon_entropy
        
        self.entropy_history.append({
            'shannon': shannon_entropy,
            'mutual_info': self.mutual_information,
            'max_entropy': np.log2(625)  # 25x25 bins
        })
    
    def _update_poincare_section(self):
        """Update Poincaré section with multiple surfaces"""
        for particle in self.particles[::10]:
            # y=0 section
            if len(particle['trail']) > 1:
                y_prev = particle['trail'][-2][1] if len(particle['trail']) > 1 else 0
                y_curr = particle['position'][1]
                
                if y_prev * y_curr < 0:  # Sign change
                    self.poincare_points.append({
                        'x': particle['position'][0],
                        'vx': particle['velocity'][0],
                        'vy': particle['velocity'][1],
                        'energy': particle['energy'],
                        'color': particle['color_index'],
                        'section': 'y=0'
                    })
            
            # x=0 section
            if len(particle['trail']) > 1:
                x_prev = particle['trail'][-2][0] if len(particle['trail']) > 1 else 0
                x_curr = particle['position'][0]
                
                if x_prev * x_curr < 0:  # Sign change
                    self.poincare_points.append({
                        'y': particle['position'][1],
                        'vy': particle['velocity'][1],
                        'vx': particle['velocity'][0],
                        'energy': particle['energy'],
                        'color': particle['color_index'],
                        'section': 'x=0'
                    })
    
    def _calculate_fourier_spectrum(self):
        """Calculate power spectrum with windowing"""
        # Take slice through potential field
        center_slice = self.potential_field[self.field_resolution//2, :]
        
        # Apply Hann window
        window = np.hanning(len(center_slice))
        windowed_slice = center_slice * window
        
        # Compute FFT
        fft_vals = np.abs(fft(windowed_slice))[:50]
        
        # Smooth into existing spectrum
        self.fourier_spectrum = 0.9 * self.fourier_spectrum + 0.1 * fft_vals
    
    def _calculate_correlation_dimension(self):
        """Estimate correlation dimension"""
        # Sample positions
        sample_size = min(100, len(self.particles))
        positions = np.array([p['position'] for p in self.particles[:sample_size]])
        
        # Calculate pairwise distances
        distances = []
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist > 0:
                    distances.append(dist)
        
        if distances:
            # Correlation sum
            distances = np.array(distances)
            r_values = np.logspace(-2, 1, 20)
            correlation_sum = []
            
            for r in r_values:
                count = np.sum(distances < r)
                correlation_sum.append(count / len(distances))
            
            # Estimate slope (correlation dimension)
            log_r = np.log(r_values[5:15])
            log_c = np.log(np.array(correlation_sum[5:15]) + 1e-10)
            
            if len(log_r) > 2:
                coeffs = np.polyfit(log_r, log_c, 1)
                self.correlation_dimension = coeffs[0]
    
    def _detect_system_state(self):
        """Detect current dynamical regime"""
        if len(self.lyapunov_history) > 10:
            recent_lyapunov = [l['mean'] for l in list(self.lyapunov_history)[-10:]]
            avg_lyapunov = np.mean(recent_lyapunov)
            
            if avg_lyapunov > 0.1:
                self.system_state = 'chaotic'
                self.dominant_behavior = 'strange_attractor'
                self.current_explanation = EXPLANATIONS['chaos']
            elif avg_lyapunov < -0.1:
                self.system_state = 'stable'
                self.dominant_behavior = 'fixed_point'
                self.current_explanation = EXPLANATIONS['stable']
            else:
                self.system_state = 'edge_of_chaos'
                self.dominant_behavior = 'critical'
                self.current_explanation = "EDGE OF CHAOS: System at critical transition"
        
        # Check energy dissipation
        if len(self.energy_history) > 10:
            recent_energy = [e['total'] for e in list(self.energy_history)[-10:]]
            energy_trend = recent_energy[-1] - recent_energy[0]
            
            if abs(energy_trend) < 0.1:
                self.current_explanation += " | " + EXPLANATIONS['energy_conserved']
            else:
                self.current_explanation += " | " + EXPLANATIONS['energy_dissipated']


class UltimateVisualizationSystem:
    """Ultimate visualization with 12 panels and full explainability"""
    
    def __init__(self, figsize=(28, 18)):
        self.fig = plt.figure(figsize=figsize, facecolor='#050505')
        self.fig.suptitle('Ultimate PhD Attractor Dynamics - Complete Analysis Suite', 
                          fontsize=22, color='#FFFFFF', fontweight='bold')
        
        # Create 4x4 grid (using 12 panels)
        gs = self.fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35,
                                  left=0.04, right=0.96, top=0.92, bottom=0.06)
        
        # Main panels
        self.ax_main = self.fig.add_subplot(gs[0:2, 0:2], projection='3d')  # Main 3D
        self.ax_flow = self.fig.add_subplot(gs[0, 2])       # Flow field
        self.ax_phase = self.fig.add_subplot(gs[0, 3])      # Phase space
        self.ax_divergence = self.fig.add_subplot(gs[1, 2]) # Divergence field
        self.ax_curl = self.fig.add_subplot(gs[1, 3])       # Curl/vorticity
        
        # Analysis panels
        self.ax_lyapunov = self.fig.add_subplot(gs[2, 0])   # Lyapunov spectrum
        self.ax_poincare = self.fig.add_subplot(gs[2, 1])   # Poincaré section
        self.ax_fourier = self.fig.add_subplot(gs[2, 2])    # Fourier spectrum
        self.ax_correlation = self.fig.add_subplot(gs[2, 3]) # Correlation dimension
        
        # Information panels
        self.ax_energy = self.fig.add_subplot(gs[3, 0])     # Energy budget
        self.ax_entropy = self.fig.add_subplot(gs[3, 1])    # Entropy measures
        self.ax_bifurcation = self.fig.add_subplot(gs[3, 2]) # Bifurcation
        self.ax_explanation = self.fig.add_subplot(gs[3, 3]) # Live explanation
        
        self._style_axes()
        
        # Initialize system
        self.system = UltimateAttractorSystem()
        self.time = 0
        self.frame = 0
        
        # Color bars storage
        self.colorbars = {}
        
    def _style_axes(self):
        """Apply sophisticated styling"""
        # 3D axis
        self.ax_main.set_facecolor('#0a0a0a')
        self.ax_main.xaxis.pane.fill = False
        self.ax_main.yaxis.pane.fill = False
        self.ax_main.zaxis.pane.fill = False
        self.ax_main.grid(True, alpha=0.15, color='#666666')
        
        # 2D axes
        all_2d_axes = [
            self.ax_flow, self.ax_phase, self.ax_divergence, self.ax_curl,
            self.ax_lyapunov, self.ax_poincare, self.ax_fourier, self.ax_correlation,
            self.ax_energy, self.ax_entropy, self.ax_bifurcation, self.ax_explanation
        ]
        
        for ax in all_2d_axes:
            ax.set_facecolor('#0a1122')
            ax.grid(True, alpha=0.2, color='#334455', linestyle=':')
            ax.tick_params(colors='#FFFFFF', labelsize=7)
            
            # Add subtle frame
            for spine in ax.spines.values():
                spine.set_color('#445566')
                spine.set_linewidth(0.5)
    
    def update_visualization(self, frame):
        """Update all panels with explanations"""
        self.frame = frame
        self.time = frame * 0.05
        
        # Update system
        self.system.update_fields(self.time)
        self.system.update_particles(0.05)
        
        # Clear all axes
        self._clear_axes()
        
        # Render all panels
        self._render_all_panels()
        
        # Add frame counter
        self.fig.text(0.98, 0.02, f'Frame: {frame} | t = {self.time:.2f}', 
                     color='#888888', fontsize=8, ha='right')
    
    def _clear_axes(self):
        """Clear all axes"""
        for ax in [
            self.ax_main, self.ax_flow, self.ax_phase, self.ax_divergence, self.ax_curl,
            self.ax_lyapunov, self.ax_poincare, self.ax_fourier, self.ax_correlation,
            self.ax_energy, self.ax_entropy, self.ax_bifurcation, self.ax_explanation
        ]:
            ax.clear()
        self._style_axes()
    
    def _render_all_panels(self):
        """Render all 12 panels"""
        self._render_main_field()
        self._render_flow_field()
        self._render_phase_space()
        self._render_divergence_field()
        self._render_curl_field()
        self._render_lyapunov_spectrum()
        self._render_poincare_section()
        self._render_fourier_spectrum()
        self._render_correlation_analysis()
        self._render_energy_budget()
        self._render_entropy_measures()
        self._render_bifurcation_diagram()
        self._render_explanation_panel()
    
    def _render_main_field(self):
        """Enhanced main 3D field with annotations"""
        self.ax_main.set_title('3D Attractor Field | Rainbow = Height | Stars = Attractors', 
                               color='#FFFFFF', fontsize=11)
        
        # Plot main surface
        Z = self.system.potential_field
        surf = self.ax_main.plot_surface(
            self.system.X, self.system.Y, Z,
            cmap='rainbow',
            shade=True,
            antialiased=True,
            rstride=2, cstride=2,
            alpha=0.9,
            linewidth=0.05,
            edgecolors='black'
        )
        
        # Multi-level contours
        contour_levels = np.linspace(Z.min(), Z.max(), 7)
        colors_cycle = ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00', 
                       '#FF0000', '#FF00FF', '#00FFFF']
        
        for level, color in zip(contour_levels, colors_cycle):
            self.ax_main.contour(self.system.X, self.system.Y, Z, 
                                levels=[level], colors=[color],
                                linewidths=1.5, alpha=0.7)
        
        # Particle cloud with trails
        for particle in self.system.particles[::8]:
            x, y = particle['position']
            if -5 <= x <= 5 and -5 <= y <= 5:
                ix = int((x + 5) * 99 / 10)
                iy = int((y + 5) * 99 / 10)
                z = self.system.potential_field[iy, ix]
                
                color = GRADIENT_COLORS[particle['color_index']]
                
                # Particle with size based on energy
                self.ax_main.scatter(x, y, z + 0.1, 
                                   s=20 + particle['energy'] * 10,
                                   c=[color], alpha=0.9,
                                   marker='o', edgecolors='white',
                                   linewidth=0.3)
                
                # Trail - with safety check
                if len(particle['trail']) > 5:
                    trail = np.array(list(particle['trail']))[-10:]
                    if len(trail) > 0:  # Safety check
                        trail_z = []
                        for pt in trail:
                            ix = int((pt[0] + 5) * 99 / 10)
                            iy = int((pt[1] + 5) * 99 / 10)
                            ix = np.clip(ix, 0, 99)
                            iy = np.clip(iy, 0, 99)
                            trail_z.append(self.system.potential_field[iy, ix])
                        
                        # Only plot if we have valid z coordinates
                        if len(trail_z) > 1:
                            try:
                                self.ax_main.plot(trail[:, 0], trail[:, 1], trail_z,
                                                color=color, alpha=0.4, linewidth=0.8)
                            except:
                                pass  # Skip if plotting fails
        
        # Annotated attractors
        for i, attractor in enumerate(self.system.attractors):
            pos = attractor['position'] + 0.5 * np.array([
                np.sin(self.time * attractor['frequency']),
                np.cos(self.time * attractor['frequency'] * 1.3)
            ])
            
            # Star marker
            self.ax_main.scatter(pos[0], pos[1], attractor['depth'] * 2 + 1,
                               s=400 * attractor['strength'],
                               c='yellow', alpha=0.95,
                               marker='*', edgecolors='white',
                               linewidth=2)
            
            # Label
            self.ax_main.text(pos[0], pos[1], attractor['depth'] * 2 + 1.5,
                            attractor['info']['name'],
                            color='white', fontsize=6, ha='center')
        
        # Set limits and labels
        self.ax_main.set_xlim(-5, 5)
        self.ax_main.set_ylim(-5, 5)
        self.ax_main.set_zlim(-4, 4)
        self.ax_main.set_xlabel('X', color='#FFFFFF', fontsize=9)
        self.ax_main.set_ylabel('Y', color='#FFFFFF', fontsize=9)
        self.ax_main.set_zlabel('Φ(x,y,t)', color='#FFFFFF', fontsize=9)
        self.ax_main.view_init(elev=25, azim=self.time * 10)
    
    def _render_flow_field(self):
        """Vector flow field with streamlines"""
        self.ax_flow.set_title('Flow Field | Arrows=Velocity | Lines=Streamlines', 
                              color='#FFFFFF', fontsize=9)
        
        # Background magnitude
        magnitude = np.sqrt(self.system.velocity_field_x**2 + 
                          self.system.velocity_field_y**2)
        
        im = self.ax_flow.imshow(magnitude.T, extent=[-5, 5, -5, 5],
                                origin='lower', cmap='viridis', alpha=0.5)
        
        # Quiver plot
        step = 8
        x_q = self.system.x[::step]
        y_q = self.system.y[::step]
        X_q, Y_q = np.meshgrid(x_q, y_q)
        U = self.system.velocity_field_x[::step, ::step]
        V = self.system.velocity_field_y[::step, ::step]
        
        M = np.sqrt(U**2 + V**2)
        self.ax_flow.quiver(X_q, Y_q, U, V, M,
                          cmap='hot', alpha=0.8,
                          scale=25, width=0.004,
                          edgecolors='black', linewidth=0.5)
        
        # Streamlines
        try:
            self.ax_flow.streamplot(self.system.x, self.system.y,
                                  self.system.velocity_field_x.T,
                                  self.system.velocity_field_y.T,
                                  color='cyan', density=1.2, linewidth=0.8,
                                  arrowsize=0.5)
        except:
            pass
        
        # Attractors
        for att in self.system.attractors:
            self.ax_flow.scatter(att['position'][0], att['position'][1],
                               s=150, c='yellow', marker='*',
                               edgecolors='red', linewidth=1)
        
        self.ax_flow.set_xlim(-5, 5)
        self.ax_flow.set_ylim(-5, 5)
        self.ax_flow.set_aspect('equal')
    
    def _render_phase_space(self):
        """Phase space with multiple projections"""
        self.ax_phase.set_title('Phase Space | Circles=Basins | Dots=States', 
                               color='#FFFFFF', fontsize=9)
        
        # Density background
        positions = np.array([p['position'] for p in self.system.particles])
        H, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1],
                                          bins=30, range=[[-5, 5], [-5, 5]])
        
        self.ax_phase.imshow(H.T, extent=[-5, 5, -5, 5], origin='lower',
                           cmap='plasma', alpha=0.3)
        
        # Attractor basins
        theta = np.linspace(0, 2*np.pi, 50)
        for att in self.system.attractors:
            color = GRADIENT_COLORS[att['color_index']]
            
            for scale in [0.5, 1.0, 1.5]:
                radius = att['strength'] * scale
                x_c = att['position'][0] + radius * np.cos(theta)
                y_c = att['position'][1] + radius * np.sin(theta)
                
                self.ax_phase.fill(x_c, y_c, color=color, alpha=0.05)
                self.ax_phase.plot(x_c, y_c, color=color, 
                                 alpha=0.2 * scale, linewidth=0.5)
        
        # Particles
        for p in self.system.particles[::3]:
            color = GRADIENT_COLORS[p['color_index']]
            self.ax_phase.scatter(p['position'][0], p['position'][1],
                                s=3, c=[color], alpha=0.7)
            
            if len(p['trail']) > 3:
                trail = np.array(list(p['trail']))[-15:]
                self.ax_phase.plot(trail[:, 0], trail[:, 1],
                                 color=color, alpha=0.2, linewidth=0.3)
        
        self.ax_phase.set_xlim(-5, 5)
        self.ax_phase.set_ylim(-5, 5)
        self.ax_phase.set_aspect('equal')
    
    def _render_divergence_field(self):
        """Divergence field showing sources/sinks"""
        self.ax_divergence.set_title('Divergence | Red=Source | Blue=Sink', 
                                    color='#FFFFFF', fontsize=9)
        
        div = self.ax_divergence.imshow(self.system.divergence_field.T,
                                       extent=[-5, 5, -5, 5],
                                       origin='lower', cmap='RdBu_r',
                                       vmin=-2, vmax=2, alpha=0.8)
        
        # Contours for critical points
        self.ax_divergence.contour(self.system.X, self.system.Y,
                                  self.system.divergence_field,
                                  levels=[0], colors='yellow',
                                  linewidths=1.5)
        
        # Mark strong sources/sinks
        sources = np.where(self.system.divergence_field > 1.5)
        sinks = np.where(self.system.divergence_field < -1.5)
        
        if len(sources[0]) > 0:
            self.ax_divergence.scatter(self.system.X[sources],
                                      self.system.Y[sources],
                                      s=20, c='red', marker='^', alpha=0.7)
        
        if len(sinks[0]) > 0:
            self.ax_divergence.scatter(self.system.X[sinks],
                                      self.system.Y[sinks],
                                      s=20, c='blue', marker='v', alpha=0.7)
        
        self.ax_divergence.set_xlim(-5, 5)
        self.ax_divergence.set_ylim(-5, 5)
        self.ax_divergence.set_aspect('equal')
    
    def _render_curl_field(self):
        """Vorticity/curl field"""
        self.ax_curl.set_title('Vorticity | Purple=CW | Green=CCW', 
                             color='#FFFFFF', fontsize=9)
        
        curl = self.ax_curl.imshow(self.system.curl_field.T,
                                  extent=[-5, 5, -5, 5],
                                  origin='lower', cmap='PRGn',
                                  vmin=-2, vmax=2, alpha=0.8)
        
        # Vortex centers
        self.ax_curl.contour(self.system.X, self.system.Y,
                           self.system.curl_field,
                           levels=5, colors='white',
                           linewidths=0.5, alpha=0.5)
        
        # Mark strong vortices
        vortices = np.where(np.abs(self.system.curl_field) > 1.5)
        if len(vortices[0]) > 0:
            self.ax_curl.scatter(self.system.X[vortices],
                               self.system.Y[vortices],
                               s=30, c='yellow', marker='o',
                               edgecolors='red', linewidth=1)
        
        self.ax_curl.set_xlim(-5, 5)
        self.ax_curl.set_ylim(-5, 5)
        self.ax_curl.set_aspect('equal')
    
    def _render_lyapunov_spectrum(self):
        """Lyapunov exponent with error bars"""
        self.ax_lyapunov.set_title('Lyapunov λ | Red=Chaos | Blue=Stable', 
                                  color='#FFFFFF', fontsize=9)
        
        if len(self.system.lyapunov_history) > 2:
            data = list(self.system.lyapunov_history)
            x = np.arange(len(data))
            
            means = [d['mean'] for d in data]
            stds = [d['std'] for d in data]
            
            # Color based on value
            colors = ['red' if m > 0 else 'blue' for m in means]
            
            # Main line
            self.ax_lyapunov.plot(x, means, color='yellow', linewidth=1.5, alpha=0.9)
            
            # Error bars
            self.ax_lyapunov.fill_between(x, 
                                         np.array(means) - np.array(stds),
                                         np.array(means) + np.array(stds),
                                         color='gray', alpha=0.2)
            
            # Scatter points
            self.ax_lyapunov.scatter(x[::5], means[::5], c=colors[::5],
                                    s=15, alpha=0.8, edgecolors='white',
                                    linewidth=0.5)
            
            # Zero line
            self.ax_lyapunov.axhline(y=0, color='white', linestyle='--',
                                    alpha=0.3, linewidth=0.5)
            
            # Add interpretation
            current_lyap = means[-1] if means else 0
            if current_lyap > 0.1:
                self.ax_lyapunov.text(0.5, 0.9, 'CHAOTIC',
                                     transform=self.ax_lyapunov.transAxes,
                                     color='red', fontsize=10, ha='center',
                                     weight='bold')
            elif current_lyap < -0.1:
                self.ax_lyapunov.text(0.5, 0.9, 'STABLE',
                                     transform=self.ax_lyapunov.transAxes,
                                     color='blue', fontsize=10, ha='center',
                                     weight='bold')
            else:
                self.ax_lyapunov.text(0.5, 0.9, 'CRITICAL',
                                     transform=self.ax_lyapunov.transAxes,
                                     color='yellow', fontsize=10, ha='center',
                                     weight='bold')
        
        self.ax_lyapunov.set_xlabel('Time', color='#FFFFFF', fontsize=8)
        self.ax_lyapunov.set_ylabel('λ', color='#FFFFFF', fontsize=8)
    
    def _render_poincare_section(self):
        """Multi-section Poincaré map"""
        self.ax_poincare.set_title('Poincaré Sections | y=0 & x=0', 
                                  color='#FFFFFF', fontsize=9)
        
        if self.system.poincare_points:
            # Separate by section
            y0_points = [p for p in self.system.poincare_points if p['section'] == 'y=0']
            x0_points = [p for p in self.system.poincare_points if p['section'] == 'x=0']
            
            # Plot y=0 section
            for p in y0_points[-200:]:
                color = GRADIENT_COLORS[p['color']]
                self.ax_poincare.scatter(p['x'], p['vx'],
                                       s=8, c=[color], alpha=0.6,
                                       marker='o')
            
            # Plot x=0 section
            for p in x0_points[-200:]:
                color = GRADIENT_COLORS[p['color']]
                self.ax_poincare.scatter(p['y'], p['vy'],
                                       s=8, c=[color], alpha=0.6,
                                       marker='^')
        
        # Reference structures
        theta = np.linspace(0, 2*np.pi, 100)
        for r in [1, 2, 3]:
            x_ref = r * np.cos(theta)
            y_ref = r * 0.5 * np.sin(theta)
            self.ax_poincare.plot(x_ref, y_ref, 'w--', alpha=0.1, linewidth=0.5)
        
        self.ax_poincare.set_xlim(-5, 5)
        self.ax_poincare.set_ylim(-3, 3)
        self.ax_poincare.set_xlabel('Position', color='#FFFFFF', fontsize=8)
        self.ax_poincare.set_ylabel('Velocity', color='#FFFFFF', fontsize=8)
    
    def _render_fourier_spectrum(self):
        """Power spectrum with peaks labeled"""
        self.ax_fourier.set_title('Power Spectrum | Peaks=Frequencies', 
                                 color='#FFFFFF', fontsize=9)
        
        freqs = np.arange(len(self.system.fourier_spectrum))
        spectrum = self.system.fourier_spectrum
        
        # Gradient bars
        for i, (f, s) in enumerate(zip(freqs, spectrum)):
            color = GRADIENT_COLORS[int(i * 5) % 256]
            self.ax_fourier.bar(f, s, color=color, alpha=0.7, edgecolor='black',
                              linewidth=0.5)
        
        # Envelope
        self.ax_fourier.plot(freqs, spectrum, color='yellow',
                           linewidth=1.5, alpha=0.9)
        
        # Find and label peaks
        if len(spectrum) > 10:
            peaks = []
            for i in range(1, len(spectrum)-1):
                if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                    if spectrum[i] > np.mean(spectrum) + np.std(spectrum):
                        peaks.append((i, spectrum[i]))
            
            # Label top peaks
            peaks.sort(key=lambda x: x[1], reverse=True)
            for i, (freq_idx, amp) in enumerate(peaks[:3]):
                self.ax_fourier.annotate(f'f{i+1}',
                                       xy=(freq_idx, amp),
                                       xytext=(freq_idx, amp + 0.1 * spectrum.max()),
                                       color='white', fontsize=7,
                                       ha='center',
                                       arrowprops=dict(arrowstyle='->', 
                                                     color='white',
                                                     lw=0.5))
        
        self.ax_fourier.set_xlabel('Frequency', color='#FFFFFF', fontsize=8)
        self.ax_fourier.set_ylabel('Power', color='#FFFFFF', fontsize=8)
        self.ax_fourier.set_xlim(0, 50)
    
    def _render_correlation_analysis(self):
        """Correlation dimension estimation"""
        self.ax_correlation.set_title(f'Correlation Dim ≈ {self.system.correlation_dimension:.2f}', 
                                     color='#FFFFFF', fontsize=9)
        
        # Create correlation integral plot
        positions = np.array([p['position'] for p in self.system.particles[:100]])
        
        # Calculate correlation integral
        r_values = np.logspace(-1.5, 1, 30)
        correlation_integral = []
        
        for r in r_values:
            count = 0
            total = 0
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < r:
                        count += 1
                    total += 1
            
            if total > 0:
                correlation_integral.append(count / total)
            else:
                correlation_integral.append(0)
        
        # Log-log plot
        valid_idx = np.array(correlation_integral) > 0
        if np.any(valid_idx):
            log_r = np.log10(r_values[valid_idx])
            log_c = np.log10(np.array(correlation_integral)[valid_idx])
            
            self.ax_correlation.plot(log_r, log_c, 'o-', color='cyan',
                                   markersize=4, linewidth=1.5, alpha=0.8)
            
            # Fit line in scaling region
            if len(log_r) > 10:
                mid_start = len(log_r) // 3
                mid_end = 2 * len(log_r) // 3
                
                fit = np.polyfit(log_r[mid_start:mid_end],
                               log_c[mid_start:mid_end], 1)
                fit_line = np.poly1d(fit)
                
                self.ax_correlation.plot(log_r[mid_start:mid_end],
                                       fit_line(log_r[mid_start:mid_end]),
                                       'r--', linewidth=2, alpha=0.8,
                                       label=f'D ≈ {fit[0]:.2f}')
            
            self.ax_correlation.legend(loc='upper left', fontsize=8,
                                     facecolor='black', edgecolor='white')
        
        self.ax_correlation.set_xlabel('log(r)', color='#FFFFFF', fontsize=8)
        self.ax_correlation.set_ylabel('log(C(r))', color='#FFFFFF', fontsize=8)
    
    def _render_energy_budget(self):
        """Detailed energy breakdown"""
        self.ax_energy.set_title('Energy Budget | K+U+Field', 
                                color='#FFFFFF', fontsize=9)
        
        if len(self.system.energy_history) > 2:
            data = list(self.system.energy_history)
            x = np.arange(len(data))
            
            kinetic = [d['kinetic'] for d in data]
            potential = [d['potential'] for d in data]
            field = [d['field'] for d in data]
            total = [d['total'] for d in data]
            
            # Stacked area plot
            self.ax_energy.fill_between(x, 0, kinetic,
                                       color='red', alpha=0.5, label='Kinetic')
            self.ax_energy.fill_between(x, kinetic,
                                       np.array(kinetic) + np.array(potential),
                                       color='blue', alpha=0.5, label='Potential')
            self.ax_energy.fill_between(x,
                                       np.array(kinetic) + np.array(potential),
                                       total,
                                       color='green', alpha=0.5, label='Field')
            
            # Total energy line
            self.ax_energy.plot(x, total, color='yellow', linewidth=2,
                              alpha=0.9, label='Total')
            
            # Energy dissipation indicator
            if len(total) > 10:
                dissipation = (total[-1] - total[-10]) / 10
                if abs(dissipation) < 0.01:
                    status = "CONSERVED"
                    color = 'green'
                else:
                    status = "DISSIPATING" if dissipation < 0 else "GAINING"
                    color = 'red' if dissipation < 0 else 'cyan'
                
                self.ax_energy.text(0.95, 0.95, status,
                                  transform=self.ax_energy.transAxes,
                                  color=color, fontsize=9, ha='right',
                                  va='top', weight='bold')
            
            self.ax_energy.legend(loc='upper left', fontsize=6,
                                facecolor='black', edgecolor='white')
        
        self.ax_energy.set_xlabel('Time', color='#FFFFFF', fontsize=8)
        self.ax_energy.set_ylabel('Energy', color='#FFFFFF', fontsize=8)
    
    def _render_entropy_measures(self):
        """Information theory metrics"""
        self.ax_entropy.set_title('Information Measures', 
                                color='#FFFFFF', fontsize=9)
        
        if len(self.system.entropy_history) > 2:
            data = list(self.system.entropy_history)
            x = np.arange(len(data))
            
            shannon = [d['shannon'] for d in data]
            mutual = [d['mutual_info'] for d in data]
            max_ent = data[0]['max_entropy']
            
            # Shannon entropy
            self.ax_entropy.fill_between(x, 0, shannon,
                                        color='purple', alpha=0.5,
                                        label='Shannon H')
            
            # Mutual information
            self.ax_entropy.plot(x, np.array(mutual) * 10, color='cyan',
                               linewidth=1.5, alpha=0.8,
                               label='Mutual I × 10')
            
            # Maximum entropy line
            self.ax_entropy.axhline(y=max_ent, color='white',
                                  linestyle='--', alpha=0.3,
                                  linewidth=0.5, label='Max H')
            
            # Complexity indicator
            if shannon:
                current_h = shannon[-1]
                complexity_ratio = current_h / max_ent
                
                if complexity_ratio > 0.7:
                    status = "HIGH COMPLEXITY"
                    color = 'red'
                elif complexity_ratio > 0.3:
                    status = "MEDIUM COMPLEXITY"
                    color = 'yellow'
                else:
                    status = "LOW COMPLEXITY"
                    color = 'blue'
                
                self.ax_entropy.text(0.95, 0.95, status,
                                   transform=self.ax_entropy.transAxes,
                                   color=color, fontsize=9, ha='right',
                                   va='top', weight='bold')
            
            self.ax_entropy.legend(loc='upper left', fontsize=6,
                                 facecolor='black', edgecolor='white')
        
        self.ax_entropy.set_xlabel('Time', color='#FFFFFF', fontsize=8)
        self.ax_entropy.set_ylabel('Information', color='#FFFFFF', fontsize=8)
    
    def _render_bifurcation_diagram(self):
        """Enhanced bifurcation with multiple parameters"""
        self.ax_bifurcation.set_title('Bifurcation Cascade', 
                                     color='#FFFFFF', fontsize=9)
        
        # Parameter range
        params = np.linspace(0.5, 4.0, 100)
        
        for i, r in enumerate(params):
            # Logistic map iteration
            x = 0.5
            
            # Transient
            for _ in range(200):
                x = r * x * (1 - x)
            
            # Collect steady states
            states = []
            for _ in range(100):
                x = r * x * (1 - x)
                states.append(x)
            
            # Remove duplicates (approximately)
            unique_states = []
            for state in states:
                is_unique = True
                for us in unique_states:
                    if abs(state - us) < 0.001:
                        is_unique = False
                        break
                if is_unique:
                    unique_states.append(state)
            
            # Color based on parameter
            color_idx = int((i / len(params)) * 255)
            color = GRADIENT_COLORS[color_idx]
            
            # Plot states
            self.ax_bifurcation.scatter([r] * len(unique_states),
                                       unique_states,
                                       s=0.3, c=[color], alpha=0.7)
        
        # Mark critical points
        critical_points = [1, 3, 1 + np.sqrt(6), 3.57]
        for cp in critical_points:
            if 0.5 <= cp <= 4.0:
                self.ax_bifurcation.axvline(x=cp, color='yellow',
                                          linestyle='--', alpha=0.3,
                                          linewidth=0.5)
        
        self.ax_bifurcation.set_xlim(0.5, 4.0)
        self.ax_bifurcation.set_ylim(0, 1)
        self.ax_bifurcation.set_xlabel('r', color='#FFFFFF', fontsize=8)
        self.ax_bifurcation.set_ylabel('x*', color='#FFFFFF', fontsize=8)
    
    def _render_explanation_panel(self):
        """Live explanation of current dynamics"""
        self.ax_explanation.set_title('System Analysis', 
                                     color='#FFFFFF', fontsize=9)
        
        # Clear for text
        self.ax_explanation.set_xticks([])
        self.ax_explanation.set_yticks([])
        
        # System state box
        state_box = FancyBboxPatch((0.05, 0.7), 0.9, 0.25,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#001133',
                                  edgecolor='cyan',
                                  linewidth=1)
        self.ax_explanation.add_patch(state_box)
        
        # State text
        state_color = {'chaotic': 'red', 'stable': 'blue', 
                      'edge_of_chaos': 'yellow'}.get(self.system.system_state, 'white')
        
        self.ax_explanation.text(0.5, 0.85,
                               f"STATE: {self.system.system_state.upper()}",
                               transform=self.ax_explanation.transAxes,
                               color=state_color, fontsize=11, ha='center',
                               weight='bold')
        
        self.ax_explanation.text(0.5, 0.75,
                               f"Behavior: {self.system.dominant_behavior}",
                               transform=self.ax_explanation.transAxes,
                               color='white', fontsize=9, ha='center')
        
        # Metrics
        metrics_text = []
        
        if len(self.system.lyapunov_history) > 0:
            lyap = self.system.lyapunov_history[-1]['mean']
            metrics_text.append(f"Lyapunov λ: {lyap:.3f}")
        
        metrics_text.append(f"Correlation D: {self.system.correlation_dimension:.2f}")
        metrics_text.append(f"Mutual Info: {self.system.mutual_information:.3f}")
        
        if len(self.system.energy_history) > 0:
            energy = self.system.energy_history[-1]['total']
            metrics_text.append(f"Total Energy: {energy:.1f}")
        
        if len(self.system.entropy_history) > 0:
            entropy = self.system.entropy_history[-1]['shannon']
            metrics_text.append(f"Entropy H: {entropy:.2f}")
        
        # Display metrics
        y_pos = 0.55
        for metric in metrics_text:
            self.ax_explanation.text(0.1, y_pos, metric,
                                   transform=self.ax_explanation.transAxes,
                                   color='#88CCFF', fontsize=8)
            y_pos -= 0.08
        
        # Explanation text
        self.ax_explanation.text(0.5, 0.1,
                               self.system.current_explanation,
                               transform=self.ax_explanation.transAxes,
                               color='#FFFF88', fontsize=7, ha='center',
                               wrap=True, style='italic')
        
        self.ax_explanation.set_xlim(0, 1)
        self.ax_explanation.set_ylim(0, 1)
    
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
            frames=3000,
            interval=50,
            blit=False,
            repeat=True
        )
        
        # Don't use tight_layout with 3D plots
        plt.show()


def launch_ultimate_system():
    """Launch the ultimate PhD visualization system"""
    print("=" * 80)
    print("ULTIMATE PHD-LEVEL ATTRACTOR DYNAMICS SYSTEM")
    print("Complete Nonlinear Dynamics Analysis Suite")
    print("=" * 80)
    print()
    print("12 ANALYSIS PANELS WITH FULL EXPLAINABILITY:")
    print()
    print("PRIMARY VISUALIZATION:")
    print("  1. 3D Attractor Field - Rainbow surface with labeled attractors")
    print("  2. Vector Flow Field - Velocity field with streamlines")
    print("  3. Phase Space Portrait - Density map with attractor basins")
    print()
    print("FIELD ANALYSIS:")
    print("  4. Divergence Field - Sources and sinks detection")
    print("  5. Curl/Vorticity Field - Rotational dynamics")
    print()
    print("CHAOS INDICATORS:")
    print("  6. Lyapunov Spectrum - Chaos vs stability with error bars")
    print("  7. Poincaré Sections - Multiple cross-sections")
    print("  8. Fourier Spectrum - Frequency analysis with peak detection")
    print("  9. Correlation Dimension - Fractal dimension estimation")
    print()
    print("STATISTICAL MEASURES:")
    print(" 10. Energy Budget - Kinetic, potential, and field energy")
    print(" 11. Information Entropy - Shannon entropy and mutual information")
    print(" 12. Bifurcation Diagram - Parameter sensitivity cascade")
    print()
    print("LIVE EXPLANATIONS:")
    print(" 13. System State Panel - Real-time analysis and explanations")
    print()
    print("FEATURES:")
    print("• 7 different attractor types (Lorenz, Rössler, Hénon, etc.)")
    print("• 500 tracked particles with full trajectory history")
    print("• Real-time chaos detection and classification")
    print("• Automatic peak detection and labeling")
    print("• Energy conservation monitoring")
    print("• Complexity measures and information theory")
    print("• Color-coded interpretations throughout")
    print()
    
    viz = UltimateVisualizationSystem()
    viz.animate()


if __name__ == "__main__":
    launch_ultimate_system()