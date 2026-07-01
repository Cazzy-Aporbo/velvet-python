"""
HORMONE CASCADE VISUALIZATION SYSTEM 2025
More advanced hormone analysis with snake-like animated visualizations
Novel architectural patterns and educational approaches
Cazzy Aporbo 
with dynamicvisualizations
"""

import colorsys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TypeVar

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage

# Advanced pastel color palette (darker tones as requested)
PASTEL_PALETTE = {
    'dusty_rose': '#B57E8A',
    'sage_green': '#7A9B76',
    'lavender_dusk': '#8B7AA8',
    'coral_shadow': '#C88B7C',
    'mint_twilight': '#6B9B8C',
    'periwinkle_depth': '#7A8BB5',
    'peach_ember': '#D4937B',
    'mauve_mist': '#A57B9B',
    'seafoam_deep': '#5B8B7B',
    'butter_ochre': '#C4A67B',
    'lilac_storm': '#9B7BC4',
    'blush_granite': '#B57B7B'
}

T = TypeVar('T')

class HormoneSnake:
    """Snake-like animated representation of hormone pathways"""

    def __init__(self, length: int = 50, complexity: float = 1.0):
        self.length = length
        self.complexity = complexity
        self.segments = deque(maxlen=length)
        self.phase = 0
        self.color_phase = 0
        self.thickness_pattern = self._generate_thickness_pattern()

    def _generate_thickness_pattern(self) -> np.ndarray:
        """Generate undulating thickness pattern for snake body"""
        t = np.linspace(0, 4 * np.pi, self.length)
        base_thickness = 2 + np.sin(t) * 0.5
        modulation = 1 + np.sin(t * 3) * 0.2
        return base_thickness * modulation * self.complexity

    def update_position(self, target: tuple[float, float], time: float) -> list[tuple[float, float]]:
        """Update snake position with organic movement"""
        # Lissajous curve for natural movement
        freq_x = 2.1 * self.complexity
        freq_y = 3.7 * self.complexity

        x = target[0] + 30 * np.sin(freq_x * time + self.phase)
        y = target[1] + 30 * np.cos(freq_y * time + self.phase * 0.7)

        # Add Perlin-like noise for organic feel
        noise_x = 5 * np.sin(time * 0.3) * np.cos(time * 0.7)
        noise_y = 5 * np.cos(time * 0.4) * np.sin(time * 0.6)

        self.segments.append((x + noise_x, y + noise_y))
        self.phase += 0.02

        return list(self.segments)

    def get_color_gradient(self, base_color: str) -> list[str]:
        """Generate smooth color gradient along snake body"""
        # Convert hex to RGB
        hex_color = base_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

        # Convert to HSV for smooth transitions
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        colors = []
        for i in range(len(self.segments)):
            # Modulate hue and value along body
            h_mod = h + 0.1 * np.sin(i/10 + self.color_phase)
            v_mod = v * (0.7 + 0.3 * np.sin(i/5))

            r, g, b = colorsys.hsv_to_rgb(h_mod % 1, s, v_mod)
            colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')

        self.color_phase += 0.05
        return colors


class QuantumHormoneField:
    """Quantum field theory inspired hormone interaction visualization"""

    def __init__(self, resolution: int = 100):
        self.resolution = resolution
        self.field = np.zeros((resolution, resolution), dtype=complex)
        self.potential = np.zeros((resolution, resolution))
        self.time = 0

    def add_hormone_source(self, x: int, y: int, strength: float, frequency: float):
        """Add hormone source as quantum field excitation"""
        # Create Gaussian wave packet
        xx, yy = np.meshgrid(range(self.resolution), range(self.resolution))
        r = np.sqrt((xx - x)**2 + (yy - y)**2)

        # Wave function with phase
        psi = strength * np.exp(-r**2 / 100) * np.exp(1j * frequency * self.time)
        self.field += psi

    def evolve(self, dt: float = 0.1):
        """Evolve field using Schrödinger-like equation"""
        # Simplified quantum evolution
        laplacian = ndimage.laplace(self.field.real) + 1j * ndimage.laplace(self.field.imag)
        self.field += dt * (-0.5 * laplacian + self.potential * self.field)

        # Normalize to prevent explosion
        self.field = self.field / (1 + 0.01 * np.abs(self.field))
        self.time += dt

    def get_probability_density(self) -> np.ndarray:
        """Get probability density for visualization"""
        return np.abs(self.field) ** 2

    def get_interference_pattern(self) -> np.ndarray:
        """Calculate interference pattern from field"""
        real_part = self.field.real
        imag_part = self.field.imag

        # Create interference from phase differences
        phase = np.angle(self.field)
        interference = np.cos(phase * 5) * np.abs(self.field)

        return interference


@dataclass
class HormoneNode:
    """Node in hormone interaction network"""
    name: str
    concentration: float
    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    connections: list[str] = field(default_factory=list)
    snake: HormoneSnake | None = None
    color: str = '#B57E8A'

    def __post_init__(self):
        if self.snake is None:
            complexity = 0.5 + self.concentration / 100
            self.snake = HormoneSnake(length=30, complexity=complexity)


class HormoneNetworkGraph:
    """Advanced hormone interaction network with physics simulation"""

    def __init__(self):
        self.nodes: dict[str, HormoneNode] = {}
        self.edges: list[tuple[str, str, float]] = []
        self.time = 0
        self.history = deque(maxlen=100)

        # Physics parameters
        self.spring_constant = 0.01
        self.repulsion_force = 500
        self.damping = 0.95

    def add_node(self, node: HormoneNode):
        """Add hormone node to network"""
        self.nodes[node.name] = node

    def add_edge(self, source: str, target: str, weight: float = 1.0):
        """Add interaction edge between hormones"""
        self.edges.append((source, target, weight))
        if source in self.nodes:
            self.nodes[source].connections.append(target)
        if target in self.nodes:
            self.nodes[target].connections.append(source)

    def update_physics(self, dt: float = 0.1):
        """Update network using force-directed layout"""
        forces = {name: np.array([0.0, 0.0]) for name in self.nodes}

        # Spring forces from edges
        for source, target, weight in self.edges:
            if source in self.nodes and target in self.nodes:
                s_pos = np.array(self.nodes[source].position)
                t_pos = np.array(self.nodes[target].position)

                diff = t_pos - s_pos
                dist = np.linalg.norm(diff)
                if dist > 0:
                    # Hooke's law with desired distance based on weight
                    desired_dist = 100 / weight
                    force = self.spring_constant * (dist - desired_dist) * diff / dist
                    forces[source] += force
                    forces[target] -= force

        # Repulsion forces between all nodes
        node_list = list(self.nodes.values())
        for i, node1 in enumerate(node_list):
            for node2 in node_list[i+1:]:
                diff = np.array(node2.position) - np.array(node1.position)
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < 200:
                    # Coulomb-like repulsion
                    force = self.repulsion_force / (dist ** 2) * diff / dist
                    forces[node1.name] -= force
                    forces[node2.name] += force

        # Update positions and velocities
        for name, node in self.nodes.items():
            # Update velocity with damping
            velocity = np.array(node.velocity)
            velocity = velocity * self.damping + forces[name] * dt

            # Update position
            position = np.array(node.position) + velocity * dt

            # Boundary conditions
            position = np.clip(position, [50, 50], [750, 550])

            node.velocity = tuple(velocity)
            node.position = tuple(position)

        self.time += dt

        # Record history
        self.history.append({name: node.concentration for name, node in self.nodes.items()})


class WaveformAnalyzer:
    """Analyzes hormone patterns as waveforms"""

    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        self.buffer_size = 1024
        self.buffer = np.zeros(self.buffer_size)
        self.write_pos = 0

    def add_sample(self, value: float):
        """Add sample to circular buffer"""
        self.buffer[self.write_pos] = value
        self.write_pos = (self.write_pos + 1) % self.buffer_size

    def get_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Get frequency spectrum using FFT"""
        # Apply window to reduce spectral leakage
        window = np.hanning(self.buffer_size)
        windowed = self.buffer * window

        # Compute FFT
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(self.buffer_size, 1/self.sampling_rate)

        magnitude = np.abs(fft)
        phase = np.angle(fft)

        return freqs, magnitude

    def get_phase_portrait(self, delay: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Create phase portrait for dynamical analysis"""
        # Delay embedding for phase space reconstruction
        x = self.buffer[:-delay]
        y = self.buffer[delay:]

        return x, y

    def calculate_entropy(self) -> float:
        """Calculate Shannon entropy of signal"""
        # Normalize to probability distribution
        positive = self.buffer - np.min(self.buffer) + 1e-10
        prob = positive / np.sum(positive)

        # Shannon entropy
        entropy = -np.sum(prob * np.log2(prob))

        return entropy


class PrebuiltHormoneData:
    """Prebuilt hormone datasets for demonstration"""

    @staticmethod
    def generate_menstrual_cycle() -> dict[str, np.ndarray]:
        """Generate 28-day menstrual cycle hormone data"""
        days = np.linspace(0, 28, 280)

        # Follicular phase (days 1-14)
        estradiol = np.where(days < 14,
                           30 + 120 * (days/14) ** 2,
                           150 * np.exp(-(days-14)/7))

        # LH surge around day 14
        lh = 10 + 50 * np.exp(-((days-14)/2) ** 2)

        # Progesterone in luteal phase
        progesterone = np.where(days < 14,
                              0.5 + 0.5 * days/14,
                              15 * np.exp(-((days-21)/7) ** 2))

        # FSH with early rise
        fsh = 8 + 4 * np.exp(-((days-5)/3) ** 2) + 2 * np.sin(days * np.pi/14)

        return {
            'estradiol': estradiol + np.random.normal(0, 5, len(days)),
            'lh': lh + np.random.normal(0, 2, len(days)),
            'progesterone': progesterone + np.random.normal(0, 1, len(days)),
            'fsh': fsh + np.random.normal(0, 1, len(days)),
            'days': days
        }

    @staticmethod
    def generate_pregnancy_progression() -> dict[str, np.ndarray]:
        """Generate 40-week pregnancy hormone progression"""
        weeks = np.linspace(0, 40, 400)

        # hCG peaks at week 10
        hcg = np.where(weeks < 10,
                      100 * np.exp(0.5 * weeks),
                      50000 * np.exp(-0.1 * (weeks - 10)))

        # Progesterone steady rise
        progesterone = 10 + 2 * weeks + 0.05 * weeks ** 2

        # Estradiol exponential rise
        estradiol = 50 * np.exp(0.08 * weeks)

        # Prolactin gradual increase
        prolactin = 10 + 5 * weeks + np.random.normal(0, 2, len(weeks))

        return {
            'hcg': hcg,
            'progesterone': progesterone,
            'estradiol': estradiol,
            'prolactin': prolactin,
            'weeks': weeks
        }

    @staticmethod
    def generate_circadian_rhythm() -> dict[str, np.ndarray]:
        """Generate 48-hour circadian hormone patterns"""
        hours = np.linspace(0, 48, 480)

        # Cortisol with morning peak
        cortisol = 12 + 8 * np.cos(2 * np.pi * (hours - 8) / 24)
        cortisol += 2 * np.sin(2 * np.pi * hours / 12)  # Ultradian rhythm

        # Melatonin with night peak
        melatonin = 5 + 15 * np.cos(2 * np.pi * (hours - 20) / 24)
        melatonin = np.maximum(0, melatonin)

        # TSH with late night peak
        tsh = 2 + 1.5 * np.cos(2 * np.pi * (hours - 2) / 24)

        # Growth hormone pulsatile secretion
        gh = np.zeros_like(hours)
        for peak_time in [22, 46, 2, 26]:  # Night peaks
            gh += 10 * np.exp(-((hours - peak_time) / 1.5) ** 2)

        return {
            'cortisol': cortisol + np.random.normal(0, 1, len(hours)),
            'melatonin': melatonin + np.random.normal(0, 0.5, len(hours)),
            'tsh': tsh + np.random.normal(0, 0.2, len(hours)),
            'growth_hormone': gh + np.random.normal(0, 0.5, len(hours)),
            'hours': hours
        }

    @staticmethod
    def generate_menopause_transition() -> dict[str, np.ndarray]:
        """Generate 5-year menopause transition data"""
        months = np.linspace(0, 60, 600)

        # Declining estradiol with fluctuations
        estradiol = 100 * np.exp(-months/30)
        estradiol += 20 * np.sin(months * np.pi/6) * np.exp(-months/40)

        # Rising FSH
        fsh = 10 + 50 * (1 - np.exp(-months/20))
        fsh += 10 * np.sin(months * np.pi/3)

        # Irregular LH
        lh = 8 + 30 * (1 - np.exp(-months/25))
        lh += 15 * np.random.random(len(months)) * np.exp(-months/50)

        # Declining progesterone
        progesterone = 10 * np.exp(-months/15)
        progesterone += 3 * np.sin(months * np.pi/4) * np.exp(-months/30)

        return {
            'estradiol': np.maximum(5, estradiol + np.random.normal(0, 3, len(months))),
            'fsh': fsh + np.random.normal(0, 2, len(months)),
            'lh': lh + np.random.normal(0, 1.5, len(months)),
            'progesterone': np.maximum(0.1, progesterone + np.random.normal(0, 0.5, len(months))),
            'months': months
        }


class HormoneVisualizationEngine:
    """Main visualization engine with advanced animations"""

    def __init__(self, figsize: tuple[int, int] = (16, 10)):
        self.fig = plt.figure(figsize=figsize, facecolor='#1a1a2a')
        self.fig.suptitle('Hormone Cascade Visualization System 2025',
                         fontsize=20, color='#E8E8F0', fontweight='bold')

        # Create subplots with custom layout
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        self.ax_network = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_waveform = self.fig.add_subplot(gs[0, 2])
        self.ax_spectrum = self.fig.add_subplot(gs[1, 2])
        self.ax_field = self.fig.add_subplot(gs[2, 0])
        self.ax_phase = self.fig.add_subplot(gs[2, 1])
        self.ax_timeline = self.fig.add_subplot(gs[2, 2])

        # Style all axes
        for ax in [self.ax_network, self.ax_waveform, self.ax_spectrum,
                   self.ax_field, self.ax_phase, self.ax_timeline]:
            ax.set_facecolor('#2a2a3a')
            ax.tick_params(colors='#A0A0B0')
            for spine in ax.spines.values():
                spine.set_color('#404050')

        # Initialize components
        self.network = HormoneNetworkGraph()
        self.quantum_field = QuantumHormoneField(resolution=50)
        self.waveform_analyzer = WaveformAnalyzer()

        # Initialize hormone network nodes
        self._initialize_network()

        # Animation data
        self.frame = 0
        self.snake_trails = defaultdict(deque)

        # Load prebuilt data
        self.menstrual_data = PrebuiltHormoneData.generate_menstrual_cycle()
        self.pregnancy_data = PrebuiltHormoneData.generate_pregnancy_progression()
        self.circadian_data = PrebuiltHormoneData.generate_circadian_rhythm()
        self.menopause_data = PrebuiltHormoneData.generate_menopause_transition()

    def _initialize_network(self):
        """Initialize hormone network with realistic interactions"""
        # Define hormone nodes with positions
        hormones = [
            ('Estradiol', 80, (200, 300), PASTEL_PALETTE['dusty_rose']),
            ('Progesterone', 60, (400, 300), PASTEL_PALETTE['sage_green']),
            ('LH', 40, (300, 200), PASTEL_PALETTE['lavender_dusk']),
            ('FSH', 45, (300, 400), PASTEL_PALETTE['coral_shadow']),
            ('Cortisol', 70, (500, 200), PASTEL_PALETTE['mint_twilight']),
            ('Insulin', 50, (500, 400), PASTEL_PALETTE['periwinkle_depth']),
            ('TSH', 35, (600, 300), PASTEL_PALETTE['peach_ember']),
            ('Prolactin', 55, (200, 200), PASTEL_PALETTE['mauve_mist']),
            ('Melatonin', 30, (400, 450), PASTEL_PALETTE['seafoam_deep']),
            ('hCG', 90, (150, 400), PASTEL_PALETTE['butter_ochre'])
        ]

        for name, conc, pos, color in hormones:
            node = HormoneNode(name, conc, pos, color=color)
            self.network.add_node(node)

        # Define interactions (feedback loops and cascades)
        interactions = [
            ('Estradiol', 'LH', 0.8),
            ('Estradiol', 'FSH', 0.7),
            ('Progesterone', 'LH', 0.9),
            ('LH', 'Progesterone', 1.2),
            ('FSH', 'Estradiol', 1.1),
            ('Cortisol', 'Insulin', 0.6),
            ('Insulin', 'Cortisol', 0.5),
            ('TSH', 'Cortisol', 0.4),
            ('Melatonin', 'Cortisol', 0.3),
            ('Prolactin', 'Estradiol', 0.5),
            ('hCG', 'Progesterone', 1.5)
        ]

        for source, target, weight in interactions:
            self.network.add_edge(source, target, weight)

    def update_frame(self, frame_num: int):
        """Update all visualization components"""
        self.frame = frame_num

        # Clear all axes
        for ax in [self.ax_network, self.ax_waveform, self.ax_spectrum,
                   self.ax_field, self.ax_phase, self.ax_timeline]:
            ax.clear()
            ax.set_facecolor('#2a2a3a')

        # Update physics
        self.network.update_physics(dt=0.2)

        # Update and draw network
        self._draw_network()

        # Update waveform analysis
        self._update_waveform()

        # Update quantum field
        self._update_quantum_field()

        # Draw phase portrait
        self._draw_phase_portrait()

        # Draw timeline
        self._draw_timeline()

        # Update title with frame info
        self.fig.suptitle(f'Hormone Cascade Visualization System 2025 - Frame {frame_num}',
                         fontsize=20, color='#E8E8F0', fontweight='bold')

    def _draw_network(self):
        """Draw hormone network with snake animations"""
        self.ax_network.set_xlim(0, 800)
        self.ax_network.set_ylim(0, 600)
        self.ax_network.set_aspect('equal')
        self.ax_network.set_title('Hormone Interaction Network', color='#E8E8F0')

        # Draw edges with varying thickness based on weight
        for source, target, weight in self.network.edges:
            if source in self.network.nodes and target in self.network.nodes:
                s_pos = self.network.nodes[source].position
                t_pos = self.network.nodes[target].position

                # Create curved edge using Bezier curve
                mid_x = (s_pos[0] + t_pos[0]) / 2 + 30 * np.sin(self.frame * 0.05)
                mid_y = (s_pos[1] + t_pos[1]) / 2 + 30 * np.cos(self.frame * 0.05)

                t = np.linspace(0, 1, 50)
                bezier_x = (1-t)**2 * s_pos[0] + 2*(1-t)*t * mid_x + t**2 * t_pos[0]
                bezier_y = (1-t)**2 * s_pos[1] + 2*(1-t)*t * mid_y + t**2 * t_pos[1]

                # Animated edge color
                edge_color = self._interpolate_colors(
                    self.network.nodes[source].color,
                    self.network.nodes[target].color,
                    0.5 + 0.5 * np.sin(self.frame * 0.1)
                )

                self.ax_network.plot(bezier_x, bezier_y,
                                   color=edge_color,
                                   linewidth=weight * 2,
                                   alpha=0.4)

        # Draw nodes with snakes
        for name, node in self.network.nodes.items():
            # Update snake position
            snake_positions = node.snake.update_position(node.position, self.frame * 0.1)

            if len(snake_positions) > 1:
                # Draw snake trail
                colors = node.snake.get_color_gradient(node.color)
                for i in range(len(snake_positions) - 1):
                    self.ax_network.plot(
                        [snake_positions[i][0], snake_positions[i+1][0]],
                        [snake_positions[i][1], snake_positions[i+1][1]],
                        color=colors[i] if i < len(colors) else node.color,
                        linewidth=node.snake.thickness_pattern[i % len(node.snake.thickness_pattern)],
                        alpha=0.7
                    )

            # Draw main node
            circle = Circle(node.position,
                          radius=np.sqrt(node.concentration) * 3,
                          facecolor=node.color,
                          edgecolor='white',
                          linewidth=2,
                          alpha=0.8)
            self.ax_network.add_patch(circle)

            # Add label
            self.ax_network.text(node.position[0], node.position[1],
                               name[:3],
                               ha='center', va='center',
                               color='white', fontsize=8, fontweight='bold')

    def _update_waveform(self):
        """Update waveform display"""
        # Use menstrual cycle data
        current_idx = (self.frame * 3) % len(self.menstrual_data['days'])

        # Display estradiol waveform
        window_size = 50
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(self.menstrual_data['days']), current_idx + window_size)

        days = self.menstrual_data['days'][start_idx:end_idx]
        estradiol = self.menstrual_data['estradiol'][start_idx:end_idx]

        self.ax_waveform.plot(days, estradiol,
                            color=PASTEL_PALETTE['dusty_rose'],
                            linewidth=2)
        self.ax_waveform.fill_between(days, 0, estradiol,
                                     color=PASTEL_PALETTE['dusty_rose'],
                                     alpha=0.3)

        self.ax_waveform.set_title('Estradiol Waveform', color='#E8E8F0')
        self.ax_waveform.set_xlabel('Days', color='#A0A0B0')
        self.ax_waveform.set_ylabel('pg/mL', color='#A0A0B0')

        # Add current value to analyzer
        self.waveform_analyzer.add_sample(estradiol[len(estradiol)//2] if len(estradiol) > 0 else 0)

        # Draw spectrum
        freqs, magnitude = self.waveform_analyzer.get_spectrum()
        self.ax_spectrum.semilogy(freqs[:50], magnitude[:50],
                                 color=PASTEL_PALETTE['lavender_dusk'],
                                 linewidth=2)
        self.ax_spectrum.fill_between(freqs[:50], 1e-10, magnitude[:50],
                                     color=PASTEL_PALETTE['lavender_dusk'],
                                     alpha=0.3)
        self.ax_spectrum.set_title('Frequency Spectrum', color='#E8E8F0')
        self.ax_spectrum.set_xlabel('Frequency (Hz)', color='#A0A0B0')
        self.ax_spectrum.set_ylabel('Magnitude', color='#A0A0B0')

    def _update_quantum_field(self):
        """Update quantum field visualization"""
        # Add hormone sources based on network nodes
        for name, node in list(self.network.nodes.items())[:3]:  # Use first 3 nodes
            x = int(node.position[0] / 16)
            y = int(node.position[1] / 12)
            self.quantum_field.add_hormone_source(x, y,
                                                 node.concentration / 100,
                                                 0.1 + self.frame * 0.01)

        # Evolve field
        self.quantum_field.evolve(dt=0.1)

        # Get interference pattern
        interference = self.quantum_field.get_interference_pattern()

        # Display field
        im = self.ax_field.imshow(interference,
                                 cmap='viridis',
                                 alpha=0.9,
                                 interpolation='bicubic')

        self.ax_field.set_title('Quantum Hormone Field', color='#E8E8F0')
        self.ax_field.set_xticks([])
        self.ax_field.set_yticks([])

    def _draw_phase_portrait(self):
        """Draw phase portrait from waveform data"""
        x, y = self.waveform_analyzer.get_phase_portrait(delay=15)

        # Create gradient color based on time
        colors = plt.cm.plasma(np.linspace(0, 1, len(x)))

        for i in range(len(x) - 1):
            self.ax_phase.plot(x[i:i+2], y[i:i+2],
                             color=colors[i],
                             linewidth=2,
                             alpha=0.7)

        # Add attractor points
        self.ax_phase.scatter(x[::20], y[::20],
                            c=colors[::20],
                            s=20,
                            alpha=0.8,
                            edgecolors='white',
                            linewidths=0.5)

        self.ax_phase.set_title('Phase Portrait', color='#E8E8F0')
        self.ax_phase.set_xlabel('x(t)', color='#A0A0B0')
        self.ax_phase.set_ylabel('x(t + τ)', color='#A0A0B0')

    def _draw_timeline(self):
        """Draw hormone timeline with multiple datasets"""
        # Select dataset based on frame
        dataset_idx = (self.frame // 100) % 4

        if dataset_idx == 0:
            # Menstrual cycle
            data = self.menstrual_data
            x_data = data['days']
            title = 'Menstrual Cycle'
            x_label = 'Days'
        elif dataset_idx == 1:
            # Pregnancy
            data = self.pregnancy_data
            x_data = data['weeks']
            title = 'Pregnancy Progression'
            x_label = 'Weeks'
        elif dataset_idx == 2:
            # Circadian
            data = self.circadian_data
            x_data = data['hours']
            title = 'Circadian Rhythm'
            x_label = 'Hours'
        else:
            # Menopause
            data = self.menopause_data
            x_data = data['months']
            title = 'Menopause Transition'
            x_label = 'Months'

        # Plot multiple hormones with different colors
        color_idx = 0
        for key in list(data.keys())[:4]:  # Plot up to 4 hormones
            if key not in ['days', 'weeks', 'hours', 'months']:
                color = list(PASTEL_PALETTE.values())[color_idx % len(PASTEL_PALETTE)]

                # Normalize data for comparison
                y_data = data[key]
                y_norm = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data) + 1e-10)

                self.ax_timeline.plot(x_data, y_norm,
                                    color=color,
                                    linewidth=2,
                                    label=key.replace('_', ' ').title(),
                                    alpha=0.8)

                color_idx += 1

        self.ax_timeline.set_title(title, color='#E8E8F0')
        self.ax_timeline.set_xlabel(x_label, color='#A0A0B0')
        self.ax_timeline.set_ylabel('Normalized Level', color='#A0A0B0')
        self.ax_timeline.legend(loc='upper right', fontsize=8,
                              framealpha=0.5, facecolor='#2a2a3a')
        self.ax_timeline.grid(True, alpha=0.2, color='#404050')

    def _interpolate_colors(self, color1: str, color2: str, t: float) -> str:
        """Interpolate between two hex colors"""
        # Convert hex to RGB
        c1 = tuple(int(color1.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
        c2 = tuple(int(color2.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))

        # Interpolate
        r = c1[0] * (1 - t) + c2[0] * t
        g = c1[1] * (1 - t) + c2[1] * t
        b = c1[2] * (1 - t) + c2[2] * t

        # Convert back to hex
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    def animate(self):
        """Start animation"""
        def update(frame):
            self.update_frame(frame)
            return []

        self.anim = animation.FuncAnimation(
            self.fig, update,
            frames=1000,
            interval=50,
            blit=False,
            repeat=True
        )

        plt.show()


def run_visualization():
    """Main function to run the visualization"""
    print("Initializing Hormone Cascade Visualization System 2025")
    print("Novel Features:")
    print("- Snake-like animated hormone pathways")
    print("- Quantum field hormone interactions")
    print("- Phase portrait dynamical analysis")
    print("- Multi-dataset timeline visualization")
    print("- 12 custom pastel colors with darker tones")
    print("- Physics-based network simulation")
    print("- Real-time waveform analysis")
    print("Starting animation...")

    engine = HormoneVisualizationEngine()
    engine.animate()


if __name__ == "__main__":
    run_visualization()
