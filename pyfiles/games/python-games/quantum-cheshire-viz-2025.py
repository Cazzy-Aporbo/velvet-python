"""
QUANTUM CHESHIRE CAT PARADOX VISUALIZER 2025
Where particles separate from their properties - The Grin Without The Cat
Featuring: Quantum tunneling mazes, spin-orbit decoupling, and probability clouds
Novel architecture: Properties that exist independently of their particles
"""

import cmath
from collections import deque
from dataclasses import dataclass, field

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Quantum twilight palette - ethereal and paradoxical
QUANTUM_PALETTE = {
    'void_black': '#0A0A0F',
    'quantum_purple': '#6B4C8A',
    'grin_pink': '#D47B9E',
    'particle_blue': '#4B7BC8',
    'spin_gold': '#C8A64B',
    'phase_teal': '#4BC8B7',
    'tunnel_violet': '#8A4BC8',
    'probability_mint': '#7BC84B',
    'entangle_coral': '#C8604B',
    'superposition_silver': '#B8B8C8',
    'measurement_crimson': '#C84B6B',
    'uncertainty_fog': '#9B9BAB'
}

@dataclass
class CheshireCat:
    """Quantum particle that can separate from its properties"""

    position: complex  # Position in complex plane
    momentum: complex
    spin: complex
    mass: float = 1.0
    charge: float = 0.0

    # Separated properties
    grin_position: complex = 0+0j
    grin_strength: float = 1.0
    separation_distance: float = 0.0

    # Quantum state
    wavefunction: np.ndarray = field(default_factory=lambda: np.zeros(100, dtype=complex))
    phase: float = 0.0
    coherence: float = 1.0

    # Visual properties
    color: str = '#6B4C8A'
    is_grinning: bool = False
    visibility: float = 1.0

    def __post_init__(self):
        """Initialize quantum state"""
        # Create Gaussian wave packet
        x = np.linspace(-10, 10, 100)
        self.wavefunction = np.exp(-x**2/4 + 1j*self.momentum.real*x)
        self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2))

    def separate_grin(self, target: complex):
        """Separate the grin (property) from the cat (particle)"""
        self.is_grinning = True
        self.grin_position = target
        self.separation_distance = abs(target - self.position)

        # Grin strength decreases with distance (quantum correlation)
        self.grin_strength = np.exp(-self.separation_distance/10)

    def evolve_schrodinger(self, dt: float, potential: np.ndarray):
        """Evolve wavefunction using Schrödinger equation"""
        # Kinetic energy operator (momentum space)
        k = np.fft.fftfreq(len(self.wavefunction), d=0.2) * 2 * np.pi
        kinetic = np.exp(-1j * k**2 * dt / (2 * self.mass))

        # Potential energy operator (position space)
        potential_op = np.exp(-1j * potential * dt)

        # Split-step Fourier method
        self.wavefunction = np.fft.ifft(kinetic * np.fft.fft(potential_op * self.wavefunction))

        # Normalize
        self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2))

        # Update phase
        self.phase += dt

    def measure_position(self) -> float:
        """Collapse wavefunction and measure position"""
        probabilities = np.abs(self.wavefunction)**2
        x = np.linspace(-10, 10, len(self.wavefunction))

        # Weighted random choice based on probability distribution
        position = np.random.choice(x, p=probabilities/np.sum(probabilities))

        # Collapse wavefunction
        self.coherence *= 0.5

        return position

    def calculate_expectation(self, operator: np.ndarray) -> complex:
        """Calculate expectation value of operator"""
        return np.sum(np.conj(self.wavefunction) * operator * self.wavefunction)


class WeakMeasurement:
    """Weak measurement apparatus - measures without destroying quantum state"""

    def __init__(self, coupling_strength: float = 0.1):
        self.coupling_strength = coupling_strength
        self.measurement_record = deque(maxlen=100)
        self.pointer_position = 0.0
        self.pointer_momentum = 0.0

    def measure_weakly(self, cat: CheshireCat, observable: str) -> float:
        """Perform weak measurement on quantum system"""
        # Weak value formula: <ψf|A|ψi> / <ψf|ψi>

        if observable == 'position':
            # Position operator
            x = np.linspace(-10, 10, len(cat.wavefunction))
            weak_value = np.sum(x * np.abs(cat.wavefunction)**2).real
        elif observable == 'spin':
            # Spin operator (simplified)
            weak_value = cat.spin.real
        elif observable == 'momentum':
            # Momentum operator
            k = np.fft.fft(cat.wavefunction)
            weak_value = np.sum(np.abs(k)**2 * np.fft.fftfreq(len(k))).real
        else:
            weak_value = 0.0

        # Add measurement backaction (very small)
        cat.coherence *= (1 - self.coupling_strength * 0.01)

        # Update pointer
        self.pointer_position += weak_value * self.coupling_strength
        self.pointer_momentum += np.random.normal(0, self.coupling_strength)

        # Record measurement
        self.measurement_record.append(weak_value)

        return weak_value

    def get_weak_trajectory(self) -> np.ndarray:
        """Get history of weak measurements"""
        return np.array(self.measurement_record)


class QuantumInterferometer:
    """Mach-Zehnder interferometer for Cheshire Cat experiment"""

    def __init__(self, arm_length: float = 10.0):
        self.arm_length = arm_length
        self.phase_shift_upper = 0.0
        self.phase_shift_lower = 0.0
        self.visibility = 1.0

        # Beam splitter parameters
        self.reflectivity = 0.5
        self.transmissivity = 0.5

        # Detection results
        self.detector_1_counts = []
        self.detector_2_counts = []

    def split_beam(self, wavefunction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split quantum state at beam splitter"""
        # Hadamard-like transformation
        upper = np.sqrt(self.transmissivity) * wavefunction
        lower = 1j * np.sqrt(self.reflectivity) * wavefunction

        return upper, lower

    def apply_phase_shift(self, wavefunction: np.ndarray, phase: float) -> np.ndarray:
        """Apply phase shift in interferometer arm"""
        return wavefunction * np.exp(1j * phase)

    def recombine_beams(self, upper: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recombine at second beam splitter"""
        # Output port 1
        out_1 = np.sqrt(self.transmissivity) * upper + 1j * np.sqrt(self.reflectivity) * lower

        # Output port 2
        out_2 = 1j * np.sqrt(self.reflectivity) * upper + np.sqrt(self.transmissivity) * lower

        return out_1, out_2

    def detect(self, wavefunction: np.ndarray) -> float:
        """Detect particle with probability"""
        return np.sum(np.abs(wavefunction)**2)


class QuantumTunnel:
    """Quantum tunneling through potential barriers"""

    def __init__(self, barrier_width: float = 5.0, barrier_height: float = 10.0):
        self.barrier_width = barrier_width
        self.barrier_height = barrier_height
        self.tunnel_probability = 0.0
        self.phase_accumulation = 0.0

    def calculate_transmission(self, energy: float) -> float:
        """Calculate tunneling probability through barrier"""
        if energy >= self.barrier_height:
            # Classical transmission
            return 1.0

        # WKB approximation for tunneling
        kappa = np.sqrt(2 * (self.barrier_height - energy))

        # Transmission coefficient
        if kappa * self.barrier_width > 30:
            # Avoid numerical overflow
            T = 16 * (energy/self.barrier_height) * (1 - energy/self.barrier_height) * np.exp(-2 * kappa * self.barrier_width)
        else:
            sinh_term = np.sinh(kappa * self.barrier_width)
            T = 1 / (1 + (sinh_term**2) / (4 * (energy/self.barrier_height) * (1 - energy/self.barrier_height)))

        return T

    def tunnel_wavepacket(self, wavefunction: np.ndarray, energy: float) -> np.ndarray:
        """Apply tunneling to wave packet"""
        T = self.calculate_transmission(energy)

        # Transmitted wave with reduced amplitude
        transmitted = wavefunction * np.sqrt(T)

        # Add phase from tunneling
        self.phase_accumulation += np.arctan(np.sqrt(1/T - 1))
        transmitted *= np.exp(1j * self.phase_accumulation)

        return transmitted


class SpinOrbitLattice:
    """Lattice where spin and position can be decoupled"""

    def __init__(self, size: int = 50):
        self.size = size
        self.lattice = np.zeros((size, size), dtype=complex)
        self.spin_field = np.zeros((size, size, 3))  # 3D spin vectors
        self.coupling_strength = 0.1

        # Initialize with random quantum fluctuations
        self.lattice = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        self.lattice *= 0.01

    def separate_spin_orbit(self, x: int, y: int) -> tuple[complex, np.ndarray]:
        """Separate spin from orbital motion at lattice site"""
        # Get orbital wavefunction
        orbital = self.lattice[x, y]

        # Get spin state
        spin = self.spin_field[x, y, :]

        # Apply decoupling transformation
        decoupled_orbital = orbital * np.exp(-1j * self.coupling_strength * np.linalg.norm(spin))

        # Move spin to different location (Cheshire effect)
        new_x = (x + 5) % self.size
        new_y = (y + 5) % self.size
        self.spin_field[new_x, new_y, :] = spin

        return decoupled_orbital, spin

    def evolve_lattice(self, dt: float):
        """Evolve lattice with spin-orbit coupling"""
        # Discrete Schrödinger evolution
        laplacian = (
            np.roll(self.lattice, 1, axis=0) +
            np.roll(self.lattice, -1, axis=0) +
            np.roll(self.lattice, 1, axis=1) +
            np.roll(self.lattice, -1, axis=1) -
            4 * self.lattice
        )

        # Add spin-orbit coupling term
        spin_magnitude = np.linalg.norm(self.spin_field, axis=2)
        coupling_term = self.coupling_strength * spin_magnitude * self.lattice

        # Update lattice
        self.lattice += 1j * dt * (laplacian + coupling_term)

        # Rotate spins
        for i in range(self.size):
            for j in range(self.size):
                angle = dt * np.linalg.norm(self.spin_field[i, j, :])
                if angle > 0:
                    axis = self.spin_field[i, j, :] / np.linalg.norm(self.spin_field[i, j, :])
                    # Rodrigues rotation formula
                    self.spin_field[i, j, :] = (
                        self.spin_field[i, j, :] * np.cos(angle) +
                        np.cross(axis, self.spin_field[i, j, :]) * np.sin(angle)
                    )


class ParadoxVisualizer:
    """Main visualization engine for Quantum Cheshire Cat paradox"""

    def __init__(self, figsize: tuple[int, int] = (16, 10)):
        self.fig = plt.figure(figsize=figsize, facecolor=QUANTUM_PALETTE['void_black'])
        self.fig.suptitle('Quantum Cheshire Cat Paradox - The Grin Without The Cat',
                         fontsize=16, color='#E8E8F8', fontweight='bold')

        # Create layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        self.ax_main = self.fig.add_subplot(gs[0:2, 0:2])           # Main Cheshire visualization
        self.ax_interferometer = self.fig.add_subplot(gs[0, 2:])    # Interferometer
        self.ax_tunneling = self.fig.add_subplot(gs[1, 2])          # Tunneling diagram
        self.ax_weak = self.fig.add_subplot(gs[1, 3])               # Weak measurement
        self.ax_lattice = self.fig.add_subplot(gs[2, 0])            # Spin-orbit lattice
        self.ax_probability = self.fig.add_subplot(gs[2, 1])        # Probability distribution
        self.ax_phase = self.fig.add_subplot(gs[2, 2], projection='polar')  # Phase diagram
        self.ax_paradox = self.fig.add_subplot(gs[2, 3])            # Paradox meter

        # Style all axes
        for ax in [self.ax_main, self.ax_interferometer, self.ax_tunneling, self.ax_weak,
                   self.ax_lattice, self.ax_probability, self.ax_paradox]:
            ax.set_facecolor(QUANTUM_PALETTE['void_black'])
            ax.tick_params(colors='#606070', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#303040')
                spine.set_linewidth(0.5)

        # Special styling for polar plot
        self.ax_phase.set_facecolor(QUANTUM_PALETTE['void_black'])
        self.ax_phase.tick_params(colors='#606070', labelsize=7)

        # Initialize quantum systems
        self.cats = []
        self.interferometer = QuantumInterferometer()
        self.tunnel = QuantumTunnel()
        self.lattice = SpinOrbitLattice()
        self.weak_measurement = WeakMeasurement()

        # Animation state
        self.frame = 0
        self.time = 0

        # Initialize Cheshire Cats
        self._initialize_quantum_cats()

        # Data tracking
        self.paradox_strength = deque(maxlen=200)
        self.grin_positions = []
        self.measurement_outcomes = deque(maxlen=50)

    def _initialize_quantum_cats(self):
        """Create initial quantum Cheshire cats"""
        # Create multiple cats in superposition
        for i in range(3):
            cat = CheshireCat(
                position=complex(200 + i*100, 250),
                momentum=complex(np.random.randn(), np.random.randn()),
                spin=complex(np.random.randn(), np.random.randn()),
                color=list(QUANTUM_PALETTE.values())[3 + i]
            )

            # Separate some grins
            if i > 0:
                target = complex(300 + i*50, 150 + i*30)
                cat.separate_grin(target)

            self.cats.append(cat)

    def update_frame(self, frame_num: int):
        """Update all quantum systems"""
        self.frame = frame_num
        self.time = frame_num * 0.05

        # Clear axes
        for ax in [self.ax_main, self.ax_interferometer, self.ax_tunneling, self.ax_weak,
                   self.ax_lattice, self.ax_probability, self.ax_phase, self.ax_paradox]:
            ax.clear()
            if ax != self.ax_phase:
                ax.set_facecolor(QUANTUM_PALETTE['void_black'])

        # Update quantum systems
        self._update_cats()
        self._update_interferometer()
        self._update_tunneling()
        self._update_weak_measurements()
        self.lattice.evolve_lattice(0.05)

        # Draw all visualizations
        self._draw_cheshire_cats()
        self._draw_interferometer()
        self._draw_tunneling()
        self._draw_weak_measurement()
        self._draw_lattice()
        self._draw_probability()
        self._draw_phase_space()
        self._draw_paradox_meter()

    def _update_cats(self):
        """Update Cheshire cat quantum states"""
        # Create potential landscape
        x = np.linspace(-10, 10, 100)
        potential = 0.5 * x**2  # Harmonic potential

        for cat in self.cats:
            # Evolve quantum state
            cat.evolve_schrodinger(0.05, potential)

            # Update position in complex plane
            cat.position += 0.1 * cat.momentum

            # Quantum walk for the grin
            if cat.is_grinning:
                # Grin performs independent quantum walk
                phase_walk = np.exp(1j * self.time * cat.separation_distance / 10)
                cat.grin_position += 5 * phase_walk

                # Grin can tunnel through barriers
                tunnel_prob = self.tunnel.calculate_transmission(abs(cat.momentum)**2)
                if np.random.random() < tunnel_prob * 0.1:
                    cat.grin_position += 20 * np.exp(1j * np.random.random() * 2 * np.pi)

            # Decoherence
            cat.coherence *= 0.999

            # Randomly separate/recombine grins
            if self.frame % 100 == 0:
                if cat.is_grinning and np.random.random() < 0.3:
                    # Recombine
                    cat.is_grinning = False
                elif not cat.is_grinning and np.random.random() < 0.5:
                    # Separate
                    target = complex(
                        np.random.uniform(100, 600),
                        np.random.uniform(100, 400)
                    )
                    cat.separate_grin(target)

    def _update_interferometer(self):
        """Update interferometer simulation"""
        if self.cats:
            cat = self.cats[0]

            # Split cat's wavefunction
            upper, lower = self.interferometer.split_beam(cat.wavefunction)

            # Apply phase shifts (different for particle and property)
            if cat.is_grinning:
                # Grin takes different path!
                upper = self.interferometer.apply_phase_shift(upper, self.time * 0.5)
                lower = self.interferometer.apply_phase_shift(lower, self.time * 0.3)
            else:
                upper = self.interferometer.apply_phase_shift(upper, self.time * 0.2)
                lower = self.interferometer.apply_phase_shift(lower, self.time * 0.2)

            # Recombine
            out1, out2 = self.interferometer.recombine_beams(upper, lower)

            # Detect
            self.interferometer.detector_1_counts.append(self.interferometer.detect(out1))
            self.interferometer.detector_2_counts.append(self.interferometer.detect(out2))

    def _update_tunneling(self):
        """Update tunneling calculations"""
        energies = np.linspace(0, 20, 100)
        self.tunnel.tunnel_probabilities = [
            self.tunnel.calculate_transmission(E) for E in energies
        ]

    def _update_weak_measurements(self):
        """Perform weak measurements on cats"""
        if self.cats:
            cat = self.cats[0]

            # Measure position weakly
            pos_weak = self.weak_measurement.measure_weakly(cat, 'position')

            # Measure spin weakly
            spin_weak = self.weak_measurement.measure_weakly(cat, 'spin')

            # Check for paradox: spin without particle
            if cat.is_grinning:
                paradox_value = abs(cat.grin_position - cat.position) * cat.grin_strength
            else:
                paradox_value = 0

            self.paradox_strength.append(paradox_value)

    def _draw_cheshire_cats(self):
        """Draw main Cheshire cat visualization"""
        self.ax_main.set_xlim(0, 700)
        self.ax_main.set_ylim(0, 500)
        self.ax_main.set_title('Quantum Cheshire Cats - Particles & Grins',
                              color='#E8E8F8', fontsize=9)
        self.ax_main.set_xlabel('Re[z]', color='#606070', fontsize=8)
        self.ax_main.set_ylabel('Im[z]', color='#606070', fontsize=8)

        for cat in self.cats:
            # Draw probability cloud around cat
            theta = np.linspace(0, 2*np.pi, 50)

            # Size based on wavefunction spread
            spread = 20 * (1 + 0.3 * np.sin(self.time * 2))

            # Quantum probability cloud
            for r_factor in [0.3, 0.6, 0.9]:
                r = spread * r_factor * cat.coherence
                x = cat.position.real + r * np.cos(theta)
                y = cat.position.imag + r * np.sin(theta)

                self.ax_main.fill(x, y,
                                color=cat.color,
                                alpha=0.1 * (1 - r_factor) * cat.coherence)

            # Draw the cat (particle)
            cat_circle = Circle(
                (cat.position.real, cat.position.imag),
                radius=10,
                facecolor=cat.color,
                edgecolor='white',
                alpha=cat.coherence,
                linewidth=1.5
            )
            self.ax_main.add_patch(cat_circle)

            # Draw spin arrow
            spin_length = 20
            spin_angle = cmath.phase(cat.spin)
            self.ax_main.arrow(
                cat.position.real, cat.position.imag,
                spin_length * np.cos(spin_angle),
                spin_length * np.sin(spin_angle),
                head_width=5, head_length=3,
                fc=QUANTUM_PALETTE['spin_gold'],
                ec=QUANTUM_PALETTE['spin_gold'],
                alpha=0.7
            )

            # Draw the grin (separated property)
            if cat.is_grinning:
                # Quantum correlation line
                self.ax_main.plot(
                    [cat.position.real, cat.grin_position.real],
                    [cat.position.imag, cat.grin_position.imag],
                    color=QUANTUM_PALETTE['grin_pink'],
                    alpha=cat.grin_strength * 0.3,
                    linewidth=1,
                    linestyle='--'
                )

                # The grin itself (crescent moon shape)
                grin_theta = np.linspace(0.5, 2.5, 30)
                grin_r = 15
                grin_x = cat.grin_position.real + grin_r * np.cos(grin_theta)
                grin_y = cat.grin_position.imag + grin_r * np.sin(grin_theta)

                self.ax_main.plot(grin_x, grin_y,
                                color=QUANTUM_PALETTE['grin_pink'],
                                linewidth=3 * cat.grin_strength,
                                alpha=cat.grin_strength)

                # Grin "exists" without the cat!
                grin_glow = Circle(
                    (cat.grin_position.real, cat.grin_position.imag),
                    radius=20 * cat.grin_strength,
                    facecolor='none',
                    edgecolor=QUANTUM_PALETTE['grin_pink'],
                    alpha=cat.grin_strength * 0.5,
                    linewidth=2
                )
                self.ax_main.add_patch(grin_glow)

                # Show paradox: property without particle
                self.ax_main.text(
                    cat.grin_position.real,
                    cat.grin_position.imag - 30,
                    'Grin without Cat!',
                    color=QUANTUM_PALETTE['grin_pink'],
                    fontsize=7,
                    alpha=cat.grin_strength,
                    ha='center'
                )

    def _draw_interferometer(self):
        """Draw Mach-Zehnder interferometer"""
        self.ax_interferometer.set_title('Quantum Interferometer',
                                        color='#E8E8F8', fontsize=9)
        self.ax_interferometer.set_xlim(0, 10)
        self.ax_interferometer.set_ylim(0, 10)

        # Draw interferometer paths
        # Input beam
        self.ax_interferometer.arrow(0, 5, 2, 0, head_width=0.3, head_length=0.2,
                                   fc=QUANTUM_PALETTE['particle_blue'],
                                   ec=QUANTUM_PALETTE['particle_blue'])

        # Beam splitter 1
        bs1 = plt.Rectangle((2.5, 4.5), 0.2, 1,
                           facecolor=QUANTUM_PALETTE['superposition_silver'],
                           alpha=0.5)
        self.ax_interferometer.add_patch(bs1)

        # Upper arm
        self.ax_interferometer.plot([2.7, 7.3], [5.5, 5.5],
                                  color=QUANTUM_PALETTE['phase_teal'],
                                  linewidth=2, alpha=0.7)

        # Lower arm
        self.ax_interferometer.plot([2.7, 7.3], [4.5, 4.5],
                                  color=QUANTUM_PALETTE['tunnel_violet'],
                                  linewidth=2, alpha=0.7)

        # Show cat in upper arm, grin in lower arm
        if self.cats and self.cats[0].is_grinning:
            # Cat in upper
            self.ax_interferometer.plot(5, 5.5, 'o',
                                      color=QUANTUM_PALETTE['particle_blue'],
                                      markersize=8)
            # Grin in lower
            self.ax_interferometer.plot(5, 4.5, '^',
                                      color=QUANTUM_PALETTE['grin_pink'],
                                      markersize=8)

            self.ax_interferometer.text(5, 6.5, 'Cat',
                                      color=QUANTUM_PALETTE['particle_blue'],
                                      fontsize=7, ha='center')
            self.ax_interferometer.text(5, 3.5, 'Grin',
                                      color=QUANTUM_PALETTE['grin_pink'],
                                      fontsize=7, ha='center')

        # Beam splitter 2
        bs2 = plt.Rectangle((7.3, 4.5), 0.2, 1,
                           facecolor=QUANTUM_PALETTE['superposition_silver'],
                           alpha=0.5)
        self.ax_interferometer.add_patch(bs2)

        # Detectors
        self.ax_interferometer.plot(9, 5.5, 's', color=QUANTUM_PALETTE['measurement_crimson'],
                                  markersize=10, label='D1')
        self.ax_interferometer.plot(9, 4.5, 's', color=QUANTUM_PALETTE['measurement_crimson'],
                                  markersize=10, label='D2')

        # Show interference pattern
        if len(self.interferometer.detector_1_counts) > 1:
            x = np.linspace(8, 10, len(self.interferometer.detector_1_counts[-20:]))
            y1 = 5.5 + 0.3 * np.array(self.interferometer.detector_1_counts[-20:])
            y2 = 4.5 - 0.3 * np.array(self.interferometer.detector_2_counts[-20:])

            self.ax_interferometer.plot(x, y1,
                                      color=QUANTUM_PALETTE['probability_mint'],
                                      linewidth=1, alpha=0.7)
            self.ax_interferometer.plot(x, y2,
                                      color=QUANTUM_PALETTE['entangle_coral'],
                                      linewidth=1, alpha=0.7)

    def _draw_tunneling(self):
        """Draw quantum tunneling diagram"""
        self.ax_tunneling.set_title('Quantum Tunneling', color='#E8E8F8', fontsize=9)
        self.ax_tunneling.set_xlabel('Position', color='#606070', fontsize=7)
        self.ax_tunneling.set_ylabel('Energy / Probability', color='#606070', fontsize=7)

        # Draw potential barrier
        x = np.linspace(0, 10, 100)
        barrier = np.zeros_like(x)
        barrier[(x > 3) & (x < 7)] = self.tunnel.barrier_height

        self.ax_tunneling.fill_between(x, 0, barrier,
                                      color=QUANTUM_PALETTE['tunnel_violet'],
                                      alpha=0.3)
        self.ax_tunneling.plot(x, barrier,
                             color=QUANTUM_PALETTE['tunnel_violet'],
                             linewidth=2)

        # Draw transmission probability
        if hasattr(self.tunnel, 'tunnel_probabilities'):
            E = np.linspace(0, 20, len(self.tunnel.tunnel_probabilities))
            T = self.tunnel.tunnel_probabilities

            # Normalize for display
            T_scaled = np.array(T) * 10

            self.ax_tunneling.plot(x[:len(T_scaled)], T_scaled,
                                 color=QUANTUM_PALETTE['probability_mint'],
                                 linewidth=2, alpha=0.7,
                                 label='Transmission')

        # Show cat can tunnel, but grin tunnels differently
        cat_energy = 5 + 3 * np.sin(self.time)
        cat_x = 1.5 + self.time % 8

        self.ax_tunneling.plot(cat_x, cat_energy, 'o',
                             color=QUANTUM_PALETTE['particle_blue'],
                             markersize=8, label='Cat')

        if self.cats and self.cats[0].is_grinning:
            grin_x = cat_x + 2
            if grin_x > 3 and grin_x < 7:
                # Grin tunnels easier!
                grin_x = 7.5
            self.ax_tunneling.plot(grin_x, cat_energy, '^',
                                 color=QUANTUM_PALETTE['grin_pink'],
                                 markersize=8, label='Grin')

        self.ax_tunneling.legend(loc='upper right', fontsize=6,
                                framealpha=0.3, facecolor=QUANTUM_PALETTE['void_black'])
        self.ax_tunneling.set_ylim(0, 15)

    def _draw_weak_measurement(self):
        """Draw weak measurement results"""
        self.ax_weak.set_title('Weak Measurement', color='#E8E8F8', fontsize=9)
        self.ax_weak.set_xlabel('Measurement #', color='#606070', fontsize=7)
        self.ax_weak.set_ylabel('Weak Value', color='#606070', fontsize=7)

        trajectory = self.weak_measurement.get_weak_trajectory()
        if len(trajectory) > 1:
            x = range(len(trajectory))

            self.ax_weak.plot(x, trajectory,
                            color=QUANTUM_PALETTE['measurement_crimson'],
                            linewidth=1.5, alpha=0.7)

            # Show anomalous weak values (outside eigenvalue range)
            anomalous = np.abs(trajectory) > 5
            if np.any(anomalous):
                self.ax_weak.scatter(np.where(anomalous)[0], trajectory[anomalous],
                                   color=QUANTUM_PALETTE['grin_pink'],
                                   s=20, alpha=0.8)

                self.ax_weak.text(len(trajectory)/2, max(trajectory),
                                'Anomalous Values!',
                                color=QUANTUM_PALETTE['grin_pink'],
                                fontsize=6, ha='center')

        # Draw pointer position
        self.ax_weak.axhline(y=self.weak_measurement.pointer_position,
                           color=QUANTUM_PALETTE['spin_gold'],
                           linestyle='--', alpha=0.5)

    def _draw_lattice(self):
        """Draw spin-orbit lattice"""
        self.ax_lattice.set_title('Spin-Orbit Lattice', color='#E8E8F8', fontsize=9)

        # Show lattice amplitude
        lattice_display = np.abs(self.lattice.lattice)**2

        im = self.ax_lattice.imshow(lattice_display,
                                   cmap='twilight',
                                   interpolation='bicubic',
                                   alpha=0.8)

        # Overlay spin field
        skip = 5  # Show every 5th spin for clarity
        x, y = np.meshgrid(range(0, self.lattice.size, skip),
                          range(0, self.lattice.size, skip))

        # Get spin components
        u = self.lattice.spin_field[::skip, ::skip, 0]
        v = self.lattice.spin_field[::skip, ::skip, 1]

        self.ax_lattice.quiver(x, y, u, v,
                              color=QUANTUM_PALETTE['spin_gold'],
                              alpha=0.6, scale=5)

        self.ax_lattice.set_xticks([])
        self.ax_lattice.set_yticks([])

    def _draw_probability(self):
        """Draw probability distribution"""
        self.ax_probability.set_title('Wavefunction |ψ|²', color='#E8E8F8', fontsize=9)
        self.ax_probability.set_xlabel('Position', color='#606070', fontsize=7)
        self.ax_probability.set_ylabel('Probability', color='#606070', fontsize=7)

        if self.cats:
            cat = self.cats[0]

            x = np.linspace(-10, 10, len(cat.wavefunction))
            prob = np.abs(cat.wavefunction)**2

            # Show probability distribution
            self.ax_probability.fill_between(x, 0, prob,
                                           color=QUANTUM_PALETTE['particle_blue'],
                                           alpha=0.3)
            self.ax_probability.plot(x, prob,
                                   color=QUANTUM_PALETTE['particle_blue'],
                                   linewidth=2)

            # Show real and imaginary parts
            self.ax_probability.plot(x, cat.wavefunction.real * 0.5,
                                   color=QUANTUM_PALETTE['phase_teal'],
                                   linewidth=1, alpha=0.5, label='Re[ψ]')
            self.ax_probability.plot(x, cat.wavefunction.imag * 0.5,
                                   color=QUANTUM_PALETTE['entangle_coral'],
                                   linewidth=1, alpha=0.5, label='Im[ψ]')

            self.ax_probability.legend(loc='upper right', fontsize=6,
                                     framealpha=0.3, facecolor=QUANTUM_PALETTE['void_black'])

    def _draw_phase_space(self):
        """Draw phase space on polar plot"""
        self.ax_phase.set_title('Quantum Phase Space', color='#E8E8F8', fontsize=9, pad=20)
        self.ax_phase.set_facecolor(QUANTUM_PALETTE['void_black'])

        # Plot phase evolution for each cat
        for i, cat in enumerate(self.cats):
            # Get phase and amplitude
            phase = cmath.phase(cat.position) + cat.phase
            amplitude = min(abs(cat.position) / 100, 1)

            # Draw phase trajectory
            phases = np.linspace(0, phase, 50)
            radii = amplitude * np.ones_like(phases)

            self.ax_phase.plot(phases, radii,
                             color=cat.color,
                             linewidth=2, alpha=0.7)

            # Mark current position
            self.ax_phase.plot(phase, amplitude, 'o',
                             color=cat.color,
                             markersize=8)

            # If grin is separated, show it separately
            if cat.is_grinning:
                grin_phase = cmath.phase(cat.grin_position)
                grin_amp = min(abs(cat.grin_position) / 100, 1) * cat.grin_strength

                self.ax_phase.plot(grin_phase, grin_amp, '^',
                                 color=QUANTUM_PALETTE['grin_pink'],
                                 markersize=8)

                # Show entanglement
                self.ax_phase.plot([phase, grin_phase], [amplitude, grin_amp],
                                 color=QUANTUM_PALETTE['entangle_coral'],
                                 linewidth=1, alpha=0.3, linestyle='--')

        # Style polar plot
        self.ax_phase.set_rticks([0.25, 0.5, 0.75, 1.0])
        self.ax_phase.set_rlabel_position(45)
        self.ax_phase.grid(True, alpha=0.2, color='#606070')

        # Color the grid
        for line in self.ax_phase.yaxis.get_gridlines():
            line.set_color('#404050')
        for line in self.ax_phase.xaxis.get_gridlines():
            line.set_color('#404050')

    def _draw_paradox_meter(self):
        """Draw paradox strength meter"""
        self.ax_paradox.set_title('Paradox Strength', color='#E8E8F8', fontsize=9)
        self.ax_paradox.set_xlabel('Time', color='#606070', fontsize=7)
        self.ax_paradox.set_ylabel('Grin-Cat Separation', color='#606070', fontsize=7)

        if len(self.paradox_strength) > 1:
            x = range(len(self.paradox_strength))
            y = list(self.paradox_strength)

            # Color based on paradox strength
            self.ax_paradox.fill_between(x, 0, y,
                                        color=QUANTUM_PALETTE['grin_pink'],
                                        alpha=0.3)

            # Highlight strong paradox regions
            y_array = np.array(y)
            strong_paradox = y_array > np.percentile(y_array, 75)

            if np.any(strong_paradox):
                self.ax_paradox.fill_between(x, 0, y_array,
                                           where=strong_paradox,
                                           color=QUANTUM_PALETTE['measurement_crimson'],
                                           alpha=0.5, label='Strong Paradox')

            self.ax_paradox.plot(x, y,
                               color=QUANTUM_PALETTE['grin_pink'],
                               linewidth=2)

            # Add Cheshire Cat smile indicator
            max_paradox = max(y) if y else 0
            if max_paradox > 50:
                self.ax_paradox.text(len(x)/2, max_paradox * 0.9,
                                   'The Cat Has Left, But The Grin Remains!',
                                   color=QUANTUM_PALETTE['grin_pink'],
                                   fontsize=7, ha='center',
                                   style='italic', weight='bold')

        self.ax_paradox.grid(True, alpha=0.2, color='#404050')

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


def run_cheshire_visualization():
    """Main entry point"""
    print("Quantum Cheshire Cat Paradox Visualizer 2025")
    print("Demonstrating: The Grin Without The Cat")
    print("\nNovel Features:")
    print("- Particles that separate from their properties")
    print("- Grins that exist without cats")
    print("- Quantum interferometry showing property-particle separation")
    print("- Weak measurements revealing paradoxical values")
    print("- Spin-orbit decoupling in lattices")
    print("- Tunneling asymmetry between particles and properties")
    print("- Phase space visualization of quantum paradox")
    print("\nQuantum mechanics at its most paradoxical!")

    visualizer = ParadoxVisualizer()
    visualizer.animate()


if __name__ == "__main__":
    run_cheshire_visualization()
