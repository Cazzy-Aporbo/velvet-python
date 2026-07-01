"""
HYPERSATURATED CRYSTAL UNIVERSE 2025
Real-time crystallization dynamics with explosive color reactions
Featuring: Supersaturation zones, nucleation cascades, dendritic growth, 
chemical gardens, crystal polymorphs, and thermodynamic phase transitions
"""

import colorsys
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

# EXPLOSIVE COLOR PALETTE - Ultra vibrant crystallization colors
CRYSTAL_COLORS = {
    # Primary crystal types (super saturated)
    'copper_sulfate': '#00FFFF',      # Electric cyan
    'potassium_dichromate': '#FF6600', # Blazing orange
    'cobalt_chloride': '#FF00FF',      # Magenta
    'nickel_sulfate': '#00FF00',       # Neon green
    'chrome_alum': '#9400D3',          # Violet
    'iron_sulfate': '#FFFF00',         # Yellow

    # Secondary formations
    'bismuth': ['#FF1493', '#00CED1', '#FFD700', '#FF69B4', '#7FFF00'],  # Iridescent
    'amethyst': ['#9B30FF', '#BA55D3', '#DDA0DD', '#E6E6FA'],
    'fluorite': ['#00FA9A', '#00BFFF', '#FF1493', '#FFD700'],
    'opal': ['#FF6B9D', '#C44569', '#F8B195', '#F67280', '#355C7D'],
    'labradorite': ['#1E90FF', '#00FFFF', '#FFD700', '#FF4500'],

    # Chemical reaction zones
    'supersaturated': '#FF00FF',
    'nucleation': '#00FF00',
    'growth_zone': '#00FFFF',
    'depletion': '#FF4500',
    'precipitation': '#FFFF00',

    # Energy states
    'high_energy': '#FF0000',
    'medium_energy': '#FFA500',
    'low_energy': '#0000FF',
    'phase_boundary': '#FFFFFF',
    'critical_point': '#FF00FF',

    # Solution colors
    'solution_1': '#001F3F',  # Deep blue
    'solution_2': '#0F1F0F',  # Deep green
    'solution_3': '#1F0F1F',  # Deep purple
    'solution_4': '#1F1F00',  # Deep yellow
    'gradient_zone': '#FF1493',
}

def generate_rainbow_gradient(n_colors):
    """Generate rainbow gradient colors"""
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}')
    return colors

@dataclass
class CrystalNucleus:
    """A crystal nucleation site with growth dynamics"""

    nucleus_id: str
    position: np.ndarray
    crystal_type: str
    size: float
    growth_rate: float
    saturation_level: float
    temperature: float
    lattice_type: str  # cubic, hexagonal, orthorhombic, etc.
    color_sequence: list[str]
    branches: list[np.ndarray] = field(default_factory=list)
    age: float = 0.0
    energy: float = 1.0
    growth_direction: np.ndarray = field(default_factory=lambda: np.random.randn(2))
    dendritic_angle: float = 60.0  # degrees
    fractal_dimension: float = 1.8
    is_growing: bool = True
    precipitated_mass: float = 0.0
    defects: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if self.position.shape[0] != 2:
            self.position = np.random.randn(2) * 20

        # Set color sequence based on crystal type
        if self.crystal_type in CRYSTAL_COLORS:
            if isinstance(CRYSTAL_COLORS[self.crystal_type], list):
                self.color_sequence = CRYSTAL_COLORS[self.crystal_type]
            else:
                self.color_sequence = [CRYSTAL_COLORS[self.crystal_type]]
        else:
            self.color_sequence = generate_rainbow_gradient(5)

        # Initialize first branch at nucleus
        self.branches = [self.position.copy()]

    def grow(self, dt: float, concentration_field: np.ndarray, temperature_field: np.ndarray):
        """Grow crystal based on local conditions"""
        if not self.is_growing:
            return

        self.age += dt

        # Get local concentration
        grid_x = int(np.clip(self.position[0] + 50, 0, 99))
        grid_y = int(np.clip(self.position[1] + 50, 0, 99))
        local_concentration = concentration_field[grid_x, grid_y]
        local_temp = temperature_field[grid_x, grid_y]

        # Growth rate depends on supersaturation and temperature
        supersaturation = local_concentration - self.saturation_level
        if supersaturation <= 0:
            self.is_growing = False
            return

        # Arrhenius-like temperature dependence
        temp_factor = np.exp(-1.0 / (local_temp + 273))
        actual_growth_rate = self.growth_rate * supersaturation * temp_factor

        # Grow existing branches
        new_branches = []
        for branch in self.branches[-10:]:  # Only grow recent branches
            if random.random() < actual_growth_rate:
                # Determine growth direction
                if self.lattice_type == 'dendritic':
                    # Dendritic growth with preferred angles
                    angles = [0, 60, 120, 180, 240, 300]
                    angle = random.choice(angles) * np.pi / 180
                    direction = np.array([np.cos(angle), np.sin(angle)])
                elif self.lattice_type == 'cubic':
                    # Cubic growth along axes
                    direction = random.choice([
                        np.array([1, 0]), np.array([-1, 0]),
                        np.array([0, 1]), np.array([0, -1])
                    ])
                elif self.lattice_type == 'hexagonal':
                    # Hexagonal growth
                    angle = random.choice([i * 60 for i in range(6)]) * np.pi / 180
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    # Random growth
                    direction = np.random.randn(2)
                    direction /= np.linalg.norm(direction)

                # Calculate new position
                step_size = self.size * actual_growth_rate
                new_pos = branch + direction * step_size
                new_branches.append(new_pos)

                # Consume concentration
                concentration_field[grid_x, grid_y] *= 0.98

                # Add mass
                self.precipitated_mass += step_size * 0.1

                # Branching probability
                if random.random() < 0.1 * actual_growth_rate:
                    # Create side branch
                    side_angle = random.choice([-1, 1]) * self.dendritic_angle * np.pi / 180
                    cos_a, sin_a = np.cos(side_angle), np.sin(side_angle)
                    rotated_dir = np.array([
                        direction[0] * cos_a - direction[1] * sin_a,
                        direction[0] * sin_a + direction[1] * cos_a
                    ])
                    side_branch = branch + rotated_dir * step_size * 0.7
                    new_branches.append(side_branch)

        self.branches.extend(new_branches)

        # Limit branch count for performance
        if len(self.branches) > 500:
            self.branches = self.branches[-500:]

        # Update size based on mass
        self.size = np.sqrt(self.precipitated_mass) * 0.5

        # Add occasional defects
        if random.random() < 0.01:
            defect_pos = self.position + np.random.randn(2) * self.size
            self.defects.append(defect_pos)


class ChemicalSolution:
    """Manages the chemical solution and concentration fields"""

    def __init__(self, size=(100, 100)):
        self.size = size
        self.concentration_field = np.ones(size) * 0.8  # Start supersaturated
        self.temperature_field = np.ones(size) * 25  # Room temperature
        self.velocity_field = np.zeros((*size, 2))
        self.ph_field = np.ones(size) * 7.0

        # Add initial concentration gradients
        self._create_concentration_gradients()

        # Diffusion kernel
        self.diffusion_kernel = np.array([[0.05, 0.1, 0.05],
                                          [0.1, 0.4, 0.1],
                                          [0.05, 0.1, 0.05]])

    def _create_concentration_gradients(self):
        """Create interesting concentration patterns"""
        # Add multiple concentration zones
        for _ in range(5):
            center = np.random.randint(20, 80, 2)
            radius = np.random.randint(10, 30)
            concentration = np.random.uniform(0.5, 1.5)

            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if dist < radius:
                        self.concentration_field[i, j] = concentration * np.exp(-dist/radius)

        # Add temperature gradients
        self.temperature_field += np.random.randn(*self.size) * 5
        self.temperature_field = gaussian_filter(self.temperature_field, sigma=5)

    def update(self, dt: float, crystals: list[CrystalNucleus]):
        """Update solution dynamics"""
        # Diffusion
        self.concentration_field = convolve2d(
            self.concentration_field, self.diffusion_kernel,
            mode='same', boundary='wrap'
        )

        # Convection (simplified)
        # Add some flow patterns
        self.velocity_field[:, :, 0] = np.sin(np.linspace(0, 2*np.pi, self.size[0])[:, np.newaxis]) * 0.1
        self.velocity_field[:, :, 1] = np.cos(np.linspace(0, 2*np.pi, self.size[1])[np.newaxis, :]) * 0.1

        # Temperature evolution
        self.temperature_field += np.random.randn(*self.size) * 0.1
        self.temperature_field = gaussian_filter(self.temperature_field, sigma=1)

        # Crystal depletion zones
        for crystal in crystals:
            if crystal.is_growing:
                x, y = int(crystal.position[0] + 50), int(crystal.position[1] + 50)
                if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                    # Create depletion halo around crystal
                    depletion_radius = int(crystal.size * 2)
                    for dx in range(-depletion_radius, depletion_radius+1):
                        for dy in range(-depletion_radius, depletion_radius+1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]:
                                dist = np.sqrt(dx**2 + dy**2)
                                if dist < depletion_radius:
                                    depletion = 0.01 * np.exp(-dist/depletion_radius)
                                    self.concentration_field[nx, ny] *= (1 - depletion)

        # Maintain boundaries
        self.concentration_field = np.clip(self.concentration_field, 0, 2)
        self.temperature_field = np.clip(self.temperature_field, 0, 100)

    def add_reagent(self, position: tuple[int, int], amount: float, reagent_type: str):
        """Add reagent to solution at position"""
        x, y = position
        radius = 10

        for i in range(max(0, x-radius), min(self.size[0], x+radius)):
            for j in range(max(0, y-radius), min(self.size[1], y+radius)):
                dist = np.sqrt((i-x)**2 + (j-y)**2)
                if dist < radius:
                    self.concentration_field[i, j] += amount * np.exp(-dist/radius)


class CrystalGrowthSimulator:
    """Main crystallization simulation engine"""

    def __init__(self):
        self.solution = ChemicalSolution()
        self.crystals = []
        self.time = 0
        self.nucleation_sites = []
        self.reaction_zones = []

        # Crystal type probabilities
        self.crystal_types = [
            ('copper_sulfate', 'dendritic'),
            ('potassium_dichromate', 'cubic'),
            ('cobalt_chloride', 'hexagonal'),
            ('nickel_sulfate', 'dendritic'),
            ('chrome_alum', 'cubic'),
            ('bismuth', 'spiral'),
            ('amethyst', 'hexagonal'),
            ('fluorite', 'cubic')
        ]

    def nucleate(self, position: np.ndarray = None):
        """Create new crystal nucleus"""
        if position is None:
            # Find supersaturated region
            supersaturated = np.where(self.solution.concentration_field > 1.0)
            if len(supersaturated[0]) > 0:
                idx = random.randint(0, len(supersaturated[0])-1)
                position = np.array([
                    supersaturated[0][idx] - 50,
                    supersaturated[1][idx] - 50
                ])
            else:
                position = np.random.uniform(-30, 30, 2)

        crystal_type, lattice = random.choice(self.crystal_types)

        nucleus = CrystalNucleus(
            nucleus_id=f"crystal_{random.randint(1000, 9999)}",
            position=position,
            crystal_type=crystal_type,
            size=random.uniform(0.5, 1.5),
            growth_rate=random.uniform(0.05, 0.2),
            saturation_level=random.uniform(0.3, 0.7),
            temperature=25.0,
            lattice_type=lattice,
            color_sequence=[],
            dendritic_angle=random.choice([30, 45, 60, 90])
        )

        self.crystals.append(nucleus)
        self.nucleation_sites.append({
            'position': position,
            'time': self.time,
            'type': crystal_type
        })

    def simulate_step(self, dt: float):
        """Run one simulation step"""
        self.time += dt

        # Update solution
        self.solution.update(dt, self.crystals)

        # Grow crystals
        for crystal in self.crystals:
            crystal.grow(dt, self.solution.concentration_field,
                        self.solution.temperature_field)

        # Spontaneous nucleation
        if random.random() < 0.02:
            self.nucleate()

        # Add reagent occasionally
        if random.random() < 0.01:
            position = (random.randint(10, 90), random.randint(10, 90))
            self.solution.add_reagent(position, 0.5, 'random')
            self.reaction_zones.append({
                'position': position,
                'time': self.time,
                'intensity': 1.0
            })

        # Update reaction zones
        self.reaction_zones = [r for r in self.reaction_zones
                               if self.time - r['time'] < 5]
        for zone in self.reaction_zones:
            zone['intensity'] *= 0.95


class HypersaturatedCrystalVisualizer:
    """Visualization system for crystal growth dynamics"""

    def __init__(self, figsize=(24, 14)):
        self.fig = plt.figure(figsize=figsize, facecolor='#000000')
        self.fig.suptitle('HYPERSATURATED CRYSTAL UNIVERSE - Real-time Crystallization',
                          fontsize=22, color='#FFFFFF', fontweight='bold')

        # Create layout
        gs = self.fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3,
                                  left=0.05, right=0.95, top=0.93, bottom=0.05)

        # Main crystal growth chamber
        self.ax_main = self.fig.add_subplot(gs[:, 0:3])

        # Concentration field
        self.ax_concentration = self.fig.add_subplot(gs[0, 3])

        # Temperature field
        self.ax_temperature = self.fig.add_subplot(gs[0, 4])

        # Crystal structure detail
        self.ax_structure = self.fig.add_subplot(gs[1, 3])

        # Phase diagram
        self.ax_phase = self.fig.add_subplot(gs[1, 4])

        # Growth kinetics
        self.ax_kinetics = self.fig.add_subplot(gs[2, 3])

        # Crystal gallery
        self.ax_gallery = self.fig.add_subplot(gs[2, 4])

        self._style_axes()

        # Initialize simulator
        self.simulator = CrystalGrowthSimulator()
        self.time = 0

        # Data tracking
        self.growth_history = defaultdict(list)
        self.nucleation_events = deque(maxlen=100)

    def _style_axes(self):
        """Style all axes with black background"""
        for ax in [self.ax_main, self.ax_concentration, self.ax_temperature,
                   self.ax_structure, self.ax_phase, self.ax_kinetics, self.ax_gallery]:
            ax.set_facecolor('#000000')
            for spine in ax.spines.values():
                spine.set_color('#333333')
                spine.set_linewidth(0.5)
            ax.tick_params(colors='#FFFFFF', labelsize=8)

    def update_visualization(self, frame):
        """Update all visualization components"""
        self.time = frame * 0.1

        # Run simulation
        self.simulator.simulate_step(0.1)

        # Track data
        for crystal in self.simulator.crystals:
            self.growth_history[crystal.crystal_type].append(crystal.size)

        # Clear and redraw
        self._clear_axes()
        self._render_all()

    def _clear_axes(self):
        """Clear all axes"""
        for ax in [self.ax_main, self.ax_concentration, self.ax_temperature,
                   self.ax_structure, self.ax_phase, self.ax_kinetics, self.ax_gallery]:
            ax.clear()
        self._style_axes()

    def _render_all(self):
        """Render all visualization components"""
        self._render_main_chamber()
        self._render_concentration_field()
        self._render_temperature_field()
        self._render_crystal_structure()
        self._render_phase_diagram()
        self._render_growth_kinetics()
        self._render_crystal_gallery()

    def _render_main_chamber(self):
        """Render main crystallization chamber"""
        self.ax_main.set_title('Crystal Growth Chamber', color='#FFFFFF', fontsize=14)

        # Background gradient
        x = np.linspace(-50, 50, 200)
        y = np.linspace(-50, 50, 200)
        X, Y = np.meshgrid(x, y)

        # Create solution background with color zones
        background = np.zeros((200, 200, 3))

        # Add color gradients based on concentration
        conc_resized = np.kron(self.simulator.solution.concentration_field,
                               np.ones((2, 2)))[:200, :200]

        # Map concentration to colors
        background[:, :, 2] = np.clip(conc_resized, 0, 1)  # Blue channel
        background[:, :, 1] = np.clip(conc_resized * 0.5, 0, 1)  # Green channel
        background[:, :, 0] = np.clip((2 - conc_resized) * 0.3, 0, 1)  # Red channel

        self.ax_main.imshow(background, extent=[-50, 50, -50, 50],
                           alpha=0.5, origin='lower')

        # Render crystals
        for crystal in self.simulator.crystals:
            if len(crystal.branches) > 1:
                # Get color from sequence
                color_idx = min(int(crystal.age * 2), len(crystal.color_sequence) - 1)
                color = crystal.color_sequence[color_idx]

                # Draw branches
                for i, branch in enumerate(crystal.branches):
                    # Fade older branches
                    alpha = 0.9 * (1 - i / len(crystal.branches)) + 0.1
                    size = crystal.size * 20 * (1 + i / len(crystal.branches))

                    # Main crystal points
                    self.ax_main.scatter(branch[0], branch[1],
                                        s=size, c=color, alpha=alpha,
                                        marker='h', edgecolors='white', linewidth=0.3)

                    # Connect branches
                    if i > 0:
                        self.ax_main.plot([crystal.branches[i-1][0], branch[0]],
                                        [crystal.branches[i-1][1], branch[1]],
                                        color=color, alpha=alpha * 0.7, linewidth=1)

                # Crystal glow effect
                if crystal.is_growing:
                    glow_size = crystal.size * 200
                    self.ax_main.scatter(crystal.position[0], crystal.position[1],
                                       s=glow_size, c=color, alpha=0.2, marker='o')

                # Show defects
                for defect in crystal.defects:
                    self.ax_main.scatter(defect[0], defect[1], s=10,
                                       c='#FF0000', alpha=0.8, marker='x')

        # Nucleation flashes
        for site in self.simulator.nucleation_sites[-5:]:
            age = self.time - site['time']
            if age < 2:
                flash_alpha = (2 - age) / 2
                self.ax_main.scatter(site['position'][0], site['position'][1],
                                   s=500 * (2 - age), c='#FFFFFF',
                                   alpha=flash_alpha * 0.5, marker='*')

        # Reaction zones
        for zone in self.simulator.reaction_zones:
            self.ax_main.scatter(zone['position'][0] - 50, zone['position'][1] - 50,
                               s=1000 * zone['intensity'], c='#FFFF00',
                               alpha=zone['intensity'] * 0.3, marker='o')

        self.ax_main.set_xlim(-50, 50)
        self.ax_main.set_ylim(-50, 50)
        self.ax_main.set_xlabel('X (μm)', color='#FFFFFF', fontsize=10)
        self.ax_main.set_ylabel('Y (μm)', color='#FFFFFF', fontsize=10)

    def _render_concentration_field(self):
        """Render concentration field heatmap"""
        self.ax_concentration.set_title('Concentration Field', color='#FFFFFF', fontsize=10)

        # Create custom colormap
        colors = ['#000033', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('concentration', colors, N=n_bins)

        im = self.ax_concentration.imshow(self.simulator.solution.concentration_field.T,
                                         cmap=cmap, vmin=0, vmax=2, origin='lower')

        # Add contour lines
        self.ax_concentration.contour(self.simulator.solution.concentration_field.T,
                                     levels=[0.5, 1.0, 1.5], colors='white',
                                     alpha=0.3, linewidths=0.5)

        self.ax_concentration.set_xticks([])
        self.ax_concentration.set_yticks([])

    def _render_temperature_field(self):
        """Render temperature field"""
        self.ax_temperature.set_title('Temperature Field', color='#FFFFFF', fontsize=10)

        # Temperature colormap
        colors = ['#0000FF', '#00FFFF', '#FFFFFF', '#FFFF00', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('temperature', colors, N=100)

        im = self.ax_temperature.imshow(self.simulator.solution.temperature_field.T,
                                       cmap=cmap, vmin=0, vmax=50, origin='lower')

        self.ax_temperature.set_xticks([])
        self.ax_temperature.set_yticks([])

    def _render_crystal_structure(self):
        """Render detailed crystal structure"""
        self.ax_structure.set_title('Crystal Lattice Structure', color='#FFFFFF', fontsize=10)

        # Show lattice patterns for different crystal types
        lattice_patterns = {
            'cubic': [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)],
            'hexagonal': [(0.5, 0), (1, 0.25), (1, 0.75), (0.5, 1), (0, 0.75), (0, 0.25)],
            'dendritic': [(0.5, 0.5)] + [(0.5 + 0.3*np.cos(i*np.pi/3),
                                         0.5 + 0.3*np.sin(i*np.pi/3)) for i in range(6)]
        }

        y_offset = 0.8
        for pattern_name, points in lattice_patterns.items():
            x_offset = 0.2

            # Draw lattice points
            for i, (x, y) in enumerate(points):
                self.ax_structure.scatter(x * 0.3 + x_offset, y * 0.3 + y_offset,
                                        s=50, c='#00FFFF', alpha=0.8)

                # Connect points
                for j, (x2, y2) in enumerate(points[i+1:], i+1):
                    if np.sqrt((x-x2)**2 + (y-y2)**2) < 0.6:
                        self.ax_structure.plot([x * 0.3 + x_offset, x2 * 0.3 + x_offset],
                                             [y * 0.3 + y_offset, y2 * 0.3 + y_offset],
                                             color='#00FF00', alpha=0.5, linewidth=1)

            self.ax_structure.text(x_offset, y_offset - 0.15, pattern_name,
                                  color='#FFFFFF', fontsize=7, ha='left')

            y_offset -= 0.35

        self.ax_structure.set_xlim(0, 1)
        self.ax_structure.set_ylim(0, 1)
        self.ax_structure.set_xticks([])
        self.ax_structure.set_yticks([])

    def _render_phase_diagram(self):
        """Render phase diagram"""
        self.ax_phase.set_title('Phase Diagram', color='#FFFFFF', fontsize=10)

        # Create phase boundaries
        T = np.linspace(0, 100, 100)
        C = np.linspace(0, 2, 100)
        T_grid, C_grid = np.meshgrid(T, C)

        # Phase field (simplified)
        phase = np.zeros_like(T_grid)
        phase[C_grid < 0.5] = 0  # Undersaturated
        phase[(C_grid >= 0.5) & (C_grid < 1.0)] = 1  # Saturated
        phase[(C_grid >= 1.0) & (C_grid < 1.5)] = 2  # Supersaturated
        phase[C_grid >= 1.5] = 3  # Precipitation

        # Add temperature dependence
        phase += 0.5 * np.sin(T_grid / 20)

        colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('phases', colors, N=4)

        self.ax_phase.contourf(T_grid, C_grid, phase, levels=4, cmap=cmap, alpha=0.7)

        # Add current system state points
        for crystal in self.simulator.crystals[:5]:
            x = int(crystal.position[0] + 50)
            y = int(crystal.position[1] + 50)
            if 0 <= x < 100 and 0 <= y < 100:
                T_point = self.simulator.solution.temperature_field[x, y]
                C_point = self.simulator.solution.concentration_field[x, y]
                self.ax_phase.scatter(T_point, C_point, s=30, c='#FFFFFF',
                                    alpha=0.8, marker='o')

        self.ax_phase.set_xlabel('Temperature (°C)', color='#FFFFFF', fontsize=8)
        self.ax_phase.set_ylabel('Concentration', color='#FFFFFF', fontsize=8)
        self.ax_phase.set_xlim(0, 100)
        self.ax_phase.set_ylim(0, 2)

    def _render_growth_kinetics(self):
        """Render growth kinetics plots"""
        self.ax_kinetics.set_title('Growth Kinetics', color='#FFFFFF', fontsize=10)

        # Plot growth curves for different crystal types
        for i, (crystal_type, sizes) in enumerate(list(self.growth_history.items())[:5]):
            if len(sizes) > 1:
                color = CRYSTAL_COLORS.get(crystal_type, '#FFFFFF')
                if isinstance(color, list):
                    color = color[0]

                x = np.arange(len(sizes))
                self.ax_kinetics.plot(x, sizes, color=color, alpha=0.8,
                                    linewidth=1.5, label=crystal_type[:8])

        self.ax_kinetics.set_xlabel('Time', color='#FFFFFF', fontsize=8)
        self.ax_kinetics.set_ylabel('Size', color='#FFFFFF', fontsize=8)

        if self.growth_history:
            self.ax_kinetics.legend(loc='upper left', fontsize=6,
                                   facecolor='#000000', edgecolor='#333333',
                                   labelcolor='#FFFFFF')

    def _render_crystal_gallery(self):
        """Render crystal type gallery"""
        self.ax_gallery.set_title('Crystal Gallery', color='#FFFFFF', fontsize=10)

        # Show different crystal types as icons
        crystal_icons = [
            ('Cu₂SO₄', '#00FFFF', 'star'),
            ('K₂Cr₂O₇', '#FF6600', 'h'),
            ('CoCl₂', '#FF00FF', 'D'),
            ('NiSO₄', '#00FF00', 'o'),
            ('Bi', '#FF1493', 's'),
        ]

        for i, (formula, color, marker) in enumerate(crystal_icons):
            x = 0.2 + (i % 3) * 0.3
            y = 0.7 - (i // 3) * 0.4

            self.ax_gallery.scatter(x, y, s=200, c=color, alpha=0.8,
                                   marker=marker, edgecolors='white', linewidth=1)
            self.ax_gallery.text(x, y - 0.15, formula, color=color,
                                fontsize=7, ha='center')

        self.ax_gallery.set_xlim(0, 1)
        self.ax_gallery.set_ylim(0, 1)
        self.ax_gallery.set_xticks([])
        self.ax_gallery.set_yticks([])

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


def launch_crystal_universe():
    """Launch the Hypersaturated Crystal Universe"""
    print()
    print("HYPERSATURATED CRYSTAL UNIVERSE 2025")
    print("Real Crystallization Physics with Explosive Colors")
    print()
    print()
    print("FEATURES:")
    print("• Real crystal growth physics: nucleation, dendritic growth, depletion zones")
    print("• Multiple crystal types: copper sulfate, chrome alum, bismuth, etc.")
    print("• Supersaturation dynamics and concentration gradients")
    print("• Temperature-dependent growth kinetics")
    print("• Lattice structures: cubic, hexagonal, dendritic")
    print("• Chemical reaction zones with explosive colors")
    print("• Phase diagrams and thermodynamics")
    print("• Crystal defects and growth patterns")
    print()
    print("Watch as crystals nucleate and grow in real-time...")

    visualizer = HypersaturatedCrystalVisualizer()
    visualizer.animate()


if __name__ == "__main__":
    launch_crystal_universe()
