"""
AURORA QUANTUM SYMPHONY 2025
The Northern Lights as Nature's Greatest Light Show
Featuring: Solar wind particles, magnetosphere dancing, atmospheric chemistry,
quantum aurora effects, and the ethereal beauty of the polar light curtains
Cazzy Aporbo MS Architecture: Must use your imagination. This is where space physics meets art
"""

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Aurora Pastel Palette - Ethereal Northern Lights Colors
AURORA_PALETTE = {
    'arctic_mint': '#E8F5F0',           # Soft mint aurora
    'aurora_rose': '#FFE8F0',           # Rose pink lights
    'northern_lavender': '#F0E8FF',     # Lavender curtains
    'polar_aqua': '#E8F8FF',            # Aqua shimmer
    'ice_crystal': '#F8F8FF',           # Ice crystal white
    'midnight_navy': '#1A1A2E',         # Dark polar sky
    'solar_peach': '#FFE8D6',           # Solar wind peach
    'magnetic_mauve': '#E8D6FF',        # Magnetic field lines
    'oxygen_jade': '#D6FFE8',           # Oxygen emission green
    'nitrogen_blush': '#FFD6E8',        # Nitrogen emission pink
    'cosmic_pearl': '#F5F5F8',          # Cosmic ray white
    'twilight_periwinkle': '#E0E8FF',   # Twilight blue
    'polar_sage': '#E8F0E8',            # Sage green aurora
    'stellar_silver': '#F0F0F5',        # Silver starlight
    'frozen_lilac': '#F0E8F5',          # Frozen lilac
    'whisper_white': '#FCFCFE',         # Whisper white
    'aurora_amber': '#FFF0D6',          # Amber glow
    'celestial_cream': '#FFF8F0',       # Celestial cream
    'moonbeam_mist': '#F8F8FC',         # Moonbeam mist
    'plasma_pink': '#FFE0F0',           # Plasma pink
    'ethereal_emerald': '#E0F8E8',      # Ethereal emerald
    'dream_dusk': '#F0E8F8',            # Dream dusk purple
    'pristine_powder': '#F8F0F8',       # Pristine snow
    'aurora_opal': '#F5F0F8'            # Aurora opal
}

@dataclass
class SolarWindParticle:
    """Charged particle from solar wind interacting with magnetosphere"""

    position: np.ndarray
    velocity: np.ndarray
    charge: float = 1.0  # +1 for proton, -1 for electron
    energy: float = 1.0
    particle_type: str = 'proton'  # proton, electron, alpha
    color: str = '#FFE8D6'
    trail: list[np.ndarray] = field(default_factory=list)
    magnetosphere_entry_time: float = 0
    aurora_probability: float = 0

    def __post_init__(self):
        if len(self.position) != 3:
            self.position = np.array([0.0, 0.0, 0.0])
        if len(self.velocity) != 3:
            self.velocity = np.array([1.0, 0.0, 0.0])

        # Set particle properties based on type
        if self.particle_type == 'electron':
            self.charge = -1.0
            self.color = AURORA_PALETTE['nitrogen_blush']
        elif self.particle_type == 'proton':
            self.charge = 1.0
            self.color = AURORA_PALETTE['solar_peach']
        elif self.particle_type == 'alpha':
            self.charge = 2.0
            self.color = AURORA_PALETTE['aurora_amber']

    def move(self, magnetic_field: np.ndarray, electric_field: np.ndarray, dt: float = 0.01):
        """Move particle according to Lorentz force"""
        # Lorentz force: F = q(E + v × B)
        cross_product = np.cross(self.velocity, magnetic_field)
        lorentz_force = self.charge * (electric_field + cross_product)

        # Update velocity (F = ma, assume mass = 1 for simplicity)
        self.velocity += lorentz_force * dt

        # Limit velocity to speed of light (normalized units)
        speed = np.linalg.norm(self.velocity)
        if speed > 10:  # Relativistic limit
            self.velocity = self.velocity / speed * 10

        # Update position
        self.position += self.velocity * dt

        # Add to trail
        self.trail.append(self.position.copy())
        if len(self.trail) > 50:  # Limit trail length
            self.trail.pop(0)

    def calculate_aurora_emission(self, atmosphere_density: float, altitude: float) -> dict[str, float]:
        """Calculate aurora emission based on particle collision with atmosphere"""
        # Aurora typically occurs at 80-500 km altitude
        if altitude < 80 or altitude > 500:
            return {'intensity': 0, 'wavelength': 0, 'color': self.color}

        # Energy deposition rate
        energy_deposition = self.energy * atmosphere_density * 0.1

        # Different atmospheric gases emit different colors
        if self.particle_type == 'electron' and self.energy > 0.5:
            # High-energy electrons excite oxygen -> green/red aurora
            if altitude > 200:
                # Higher altitude -> red oxygen line (630.0 nm)
                return {
                    'intensity': energy_deposition * 0.8,
                    'wavelength': 630.0,
                    'color': AURORA_PALETTE['aurora_rose']
                }
            else:
                # Lower altitude -> green oxygen line (557.7 nm)
                return {
                    'intensity': energy_deposition,
                    'wavelength': 557.7,
                    'color': AURORA_PALETTE['oxygen_jade']
                }
        elif self.particle_type == 'proton':
            # Protons excite nitrogen -> blue/purple aurora
            return {
                'intensity': energy_deposition * 0.6,
                'wavelength': 428.0,
                'color': AURORA_PALETTE['northern_lavender']
            }

        return {'intensity': 0, 'wavelength': 0, 'color': self.color}


class MagnetosphereField:
    """Earth's magnetosphere with field lines and current systems"""

    def __init__(self, earth_radius: float = 10):
        self.earth_radius = earth_radius
        self.dipole_moment = 100  # Earth's magnetic dipole strength
        self.field_lines = []
        self.current_systems = []
        self.solar_wind_pressure = 1.0
        self.substorm_activity = 0.0

        self._generate_field_lines()
        self._create_current_systems()

    def _generate_field_lines(self):
        """Generate dipolar magnetic field lines"""
        n_field_lines = 20

        for i in range(n_field_lines):
            # Magnetic colatitude (angle from magnetic north pole)
            colatitude = np.pi * (i + 1) / (n_field_lines + 1)

            # Generate field line from north to south
            n_points = 100
            field_line_points = []

            for j in range(n_points):
                # Parameter along field line
                t = j / (n_points - 1)

                # Dipolar field line equation in spherical coordinates
                # r = L * sin²(λ) where L is the L-shell parameter
                L_shell = self.earth_radius / (np.sin(colatitude)**2) * 3

                # Latitude along field line
                latitude = np.arccos(np.sqrt((1 - t) * np.cos(colatitude)**2 + t * np.cos(np.pi - colatitude)**2))

                # Radius at this latitude
                r = L_shell * np.sin(latitude)**2

                # Convert to Cartesian coordinates
                x = r * np.sin(latitude) * np.cos(0)  # Simplified: all in x-z plane
                y = 0
                z = r * np.cos(latitude)

                field_line_points.append(np.array([x, y, z]))

            self.field_lines.append({
                'points': field_line_points,
                'L_shell': L_shell,
                'activity': random.uniform(0.3, 1.0),
                'color': random.choice([
                    AURORA_PALETTE['magnetic_mauve'],
                    AURORA_PALETTE['twilight_periwinkle'],
                    AURORA_PALETTE['polar_sage']
                ])
            })

    def _create_current_systems(self):
        """Create magnetospheric current systems"""
        # Ring current
        ring_current = {
            'type': 'ring',
            'radius': self.earth_radius * 4,
            'intensity': 1.0,
            'particles': []
        }

        # Create ring current particles
        n_ring_particles = 30
        for i in range(n_ring_particles):
            angle = 2 * np.pi * i / n_ring_particles
            position = np.array([
                ring_current['radius'] * np.cos(angle),
                ring_current['radius'] * np.sin(angle),
                random.uniform(-2, 2)
            ])

            # Drift velocity
            velocity = np.array([
                -np.sin(angle),  # Westward drift
                np.cos(angle),
                0
            ]) * 2

            particle = SolarWindParticle(
                position=position,
                velocity=velocity,
                particle_type='electron',
                energy=random.uniform(1, 3)
            )
            ring_current['particles'].append(particle)

        self.current_systems.append(ring_current)

        # Field-aligned currents (Birkeland currents)
        for field_line in self.field_lines[::3]:  # Every 3rd field line
            birkeland_current = {
                'type': 'field_aligned',
                'field_line': field_line,
                'intensity': random.uniform(0.5, 2.0),
                'direction': random.choice([1, -1])  # Up or down
            }
            self.current_systems.append(birkeland_current)

    def get_magnetic_field(self, position: np.ndarray) -> np.ndarray:
        """Calculate magnetic field at given position using dipole model"""
        r = np.linalg.norm(position)

        if r < self.earth_radius * 0.9:  # Inside Earth
            return np.array([0, 0, -2])  # Strong downward field

        # Dipole field in Cartesian coordinates
        # B = (μ₀/4π) * (3(m·r̂)r̂ - m) / r³

        # Magnetic dipole moment (pointing north)
        m = np.array([0, 0, self.dipole_moment])

        r_hat = position / r
        m_dot_r = np.dot(m, r_hat)

        # Dipole field
        B = (3 * m_dot_r * r_hat - m) / (r**3)

        # Add perturbations from solar wind
        perturbation = self.solar_wind_pressure * np.array([
            0.1 * np.sin(r * 0.1),
            0.1 * np.cos(r * 0.1),
            0
        ])

        return B + perturbation

    def evolve(self, time: float, solar_wind_strength: float):
        """Evolve magnetosphere based on solar wind conditions"""
        self.solar_wind_pressure = solar_wind_strength

        # Substorm activity
        self.substorm_activity = 0.5 + 0.4 * np.sin(time * 0.05) * solar_wind_strength

        # Evolve ring current
        for current_system in self.current_systems:
            if current_system['type'] == 'ring':
                for particle in current_system['particles']:
                    # Ring current drift
                    magnetic_field = self.get_magnetic_field(particle.position)
                    electric_field = np.array([0, 0, 0])  # Simplified

                    particle.move(magnetic_field, electric_field, 0.02)


class AtmosphericLayer:
    """Atmospheric layers where aurora emissions occur"""

    def __init__(self):
        self.layers = {
            'thermosphere': {
                'altitude_range': (80, 500),
                'density_profile': self._thermosphere_density,
                'composition': {'O': 0.8, 'N2': 0.15, 'NO': 0.05},
                'temperature': lambda alt: 200 + (alt - 80) * 5  # K
            },
            'mesosphere': {
                'altitude_range': (50, 80),
                'density_profile': self._mesosphere_density,
                'composition': {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01},
                'temperature': lambda alt: 270 - (alt - 50) * 3  # K
            }
        }

        self.aurora_emissions = []

    def _thermosphere_density(self, altitude: float) -> float:
        """Density profile of thermosphere"""
        # Exponential decay with altitude
        scale_height = 50  # km
        base_density = 1e-12  # kg/m³ at 80 km
        return base_density * np.exp(-(altitude - 80) / scale_height)

    def _mesosphere_density(self, altitude: float) -> float:
        """Density profile of mesosphere"""
        scale_height = 7  # km
        base_density = 1e-6  # kg/m³ at 50 km
        return base_density * np.exp(-(altitude - 50) / scale_height)

    def particle_collision(self, particle: SolarWindParticle, altitude: float) -> dict[str, Any] | None:
        """Calculate particle collision with atmospheric gases"""
        if altitude < 50 or altitude > 500:
            return None

        # Determine atmospheric layer
        if 80 <= altitude <= 500:
            layer = self.layers['thermosphere']
        else:
            layer = self.layers['mesosphere']

        # Get atmospheric density
        density = layer['density_profile'](altitude)

        # Collision probability
        collision_prob = density * particle.energy * 0.01

        if random.random() < collision_prob:
            # Create aurora emission
            emission = particle.calculate_aurora_emission(density, altitude)

            if emission['intensity'] > 0:
                aurora_point = {
                    'position': particle.position.copy(),
                    'altitude': altitude,
                    'intensity': emission['intensity'],
                    'color': emission['color'],
                    'wavelength': emission['wavelength'],
                    'lifetime': random.uniform(0.1, 2.0),
                    'age': 0,
                    'particle_type': particle.particle_type
                }

                self.aurora_emissions.append(aurora_point)
                return aurora_point

        return None

    def evolve_emissions(self, dt: float):
        """Evolve aurora emission points"""
        for emission in self.aurora_emissions[:]:
            emission['age'] += dt

            # Fade over time
            fade_factor = max(0, 1 - emission['age'] / emission['lifetime'])
            emission['intensity'] *= fade_factor

            # Remove dead emissions
            if emission['intensity'] < 0.01:
                self.aurora_emissions.remove(emission)


class AuroraCurtain:
    """Aurora curtain structure with dancing, flowing motion"""

    def __init__(self, base_altitude: float = 100, width: float = 50):
        self.base_altitude = base_altitude
        self.width = width
        self.height = random.uniform(100, 300)
        self.curtain_points = []
        self.motion_phase = random.uniform(0, 2*np.pi)
        self.wave_frequency = random.uniform(0.1, 0.3)
        self.intensity_profile = []
        self.color_profile = []

        self._generate_curtain_structure()

    def _generate_curtain_structure(self):
        """Generate 3D curtain structure"""
        n_vertical = 20
        n_horizontal = 15

        for i in range(n_horizontal):
            vertical_line = []
            x = (i - n_horizontal/2) * self.width / n_horizontal

            for j in range(n_vertical):
                y = random.uniform(-10, 10)  # Depth variation
                z = self.base_altitude + (j / n_vertical) * self.height

                point = np.array([x, y, z])
                vertical_line.append(point)

            self.curtain_points.append(vertical_line)

            # Intensity profile for this vertical line
            intensities = []
            colors = []

            for j in range(n_vertical):
                # Typical aurora intensity profile
                normalized_height = j / n_vertical

                if normalized_height < 0.2:  # Lower edge - weak
                    intensity = normalized_height * 0.5
                    color = AURORA_PALETTE['aurora_rose']
                elif normalized_height < 0.8:  # Main curtain - bright
                    intensity = 0.5 + 0.5 * np.sin(normalized_height * np.pi)
                    if random.random() < 0.6:
                        color = AURORA_PALETTE['oxygen_jade']
                    else:
                        color = AURORA_PALETTE['northern_lavender']
                else:  # Upper edge - fading
                    intensity = (1 - normalized_height) * 2
                    color = AURORA_PALETTE['polar_aqua']

                intensities.append(intensity)
                colors.append(color)

            self.intensity_profile.append(intensities)
            self.color_profile.append(colors)

    def dance(self, time: float):
        """Animate curtain dancing motion"""
        self.motion_phase += 0.05

        # Wave motion through curtain
        for i, vertical_line in enumerate(self.curtain_points):
            x_offset = i / len(self.curtain_points) * 2 * np.pi

            for j, point in enumerate(vertical_line):
                # Vertical wave
                wave_amplitude = 5 + 3 * np.sin(time * 0.1)
                vertical_wave = wave_amplitude * np.sin(
                    self.wave_frequency * j + self.motion_phase + x_offset
                )

                # Horizontal sway
                horizontal_sway = 3 * np.sin(
                    time * 0.07 + x_offset + j * 0.1
                )

                # Update position with wave motion
                original_x = (i - len(self.curtain_points)/2) * self.width / len(self.curtain_points)
                original_z = self.base_altitude + (j / len(vertical_line)) * self.height

                vertical_line[j] = np.array([
                    original_x + horizontal_sway,
                    vertical_wave,
                    original_z
                ])

        # Intensity fluctuations
        for i, intensities in enumerate(self.intensity_profile):
            for j in range(len(intensities)):
                # Add shimmer effect
                shimmer = 0.1 * np.sin(time * 2 + i * 0.5 + j * 0.3)
                base_intensity = 0.5 + 0.3 * np.sin(j / len(intensities) * np.pi)
                intensities[j] = max(0, min(1, base_intensity + shimmer))


class SolarWindStream:
    """Solar wind particle stream from the Sun"""

    def __init__(self):
        self.particles = []
        self.wind_speed = 400  # km/s typical
        self.density = 5  # particles per cm³
        self.temperature = 100000  # K
        self.magnetic_field_strength = 5e-9  # Tesla
        self.coronal_mass_ejection = False
        self.cme_strength = 0

    def generate_particles(self, n_particles: int = 10):
        """Generate new solar wind particles"""
        for _ in range(n_particles):
            # Start particles from left side (Sun direction)
            position = np.array([
                -200 + random.uniform(-20, 20),
                random.uniform(-100, 100),
                random.uniform(-50, 50)
            ])

            # Solar wind velocity (mostly in +x direction)
            base_velocity = self.wind_speed * 0.01  # Scaled for visualization
            velocity = np.array([
                base_velocity + random.uniform(-0.5, 0.5),
                random.uniform(-0.2, 0.2),
                random.uniform(-0.2, 0.2)
            ])

            # Particle type distribution
            if random.random() < 0.96:  # 96% protons
                particle_type = 'proton'
                energy = random.uniform(0.5, 2.0)
            elif random.random() < 0.99:  # 3% alpha particles
                particle_type = 'alpha'
                energy = random.uniform(1.0, 4.0)
            else:  # 1% electrons
                particle_type = 'electron'
                energy = random.uniform(0.1, 1.0)

            # CME enhancement
            if self.coronal_mass_ejection:
                energy *= (1 + self.cme_strength)
                velocity *= (1 + self.cme_strength * 0.5)

            particle = SolarWindParticle(
                position=position,
                velocity=velocity,
                particle_type=particle_type,
                energy=energy
            )

            self.particles.append(particle)

    def trigger_cme(self):
        """Trigger coronal mass ejection"""
        self.coronal_mass_ejection = True
        self.cme_strength = random.uniform(2, 5)

        # Generate burst of high-energy particles
        self.generate_particles(50)

    def evolve(self, time: float):
        """Evolve solar wind conditions"""
        # Gradually decay CME
        if self.coronal_mass_ejection:
            self.cme_strength *= 0.99
            if self.cme_strength < 0.1:
                self.coronal_mass_ejection = False
                self.cme_strength = 0

        # Randomly trigger CME
        if random.random() < 0.001:  # Rare event
            self.trigger_cme()

        # Variable solar wind conditions
        self.wind_speed = 400 + 100 * np.sin(time * 0.02)
        self.density = 5 + 2 * np.sin(time * 0.03)


class AuroraVisualizer:
    """Main visualization system for aurora phenomena"""

    def __init__(self, figsize: tuple[int, int] = (20, 12)):
        # Setup figure with night sky background
        self.fig = plt.figure(figsize=figsize, facecolor=AURORA_PALETTE['midnight_navy'])
        self.fig.suptitle('Aurora Quantum Symphony - Northern Lights Dancing Across the Sky',
                         fontsize=18, color=AURORA_PALETTE['ice_crystal'], fontweight='bold')

        # Create layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.25, wspace=0.25)

        # Main aurora view (large central panel)
        self.ax_aurora = self.fig.add_subplot(gs[0:2, 0:3], projection='3d')

        # Solar wind monitor (top right)
        self.ax_solar_wind = self.fig.add_subplot(gs[0, 3])

        # Magnetosphere activity (middle right)
        self.ax_magnetosphere = self.fig.add_subplot(gs[1, 3])

        # Aurora spectrum (bottom left)
        self.ax_spectrum = self.fig.add_subplot(gs[2, 0])

        # Atmospheric layers (bottom center-left)
        self.ax_atmosphere = self.fig.add_subplot(gs[2, 1])

        # Aurora curtain profile (bottom center-right)
        self.ax_curtain = self.fig.add_subplot(gs[2, 2])

        # Geomagnetic activity (bottom right)
        self.ax_geomagnetic = self.fig.add_subplot(gs[2, 3])

        # Style all axes
        self._style_axes()

        # Initialize aurora system components
        self.solar_wind = SolarWindStream()
        self.magnetosphere = MagnetosphereField()
        self.atmosphere = AtmosphericLayer()
        self.aurora_curtains = []

        # Animation state
        self.time = 0
        self.aurora_activity = 0.5
        self.geomagnetic_index = 3  # Kp index

        # Data tracking
        self.aurora_intensity_history = deque(maxlen=100)
        self.solar_wind_history = deque(maxlen=100)
        self.spectrum_data = defaultdict(list)

        # Initialize aurora curtains
        self._create_aurora_curtains()

    def _style_axes(self):
        """Style all axes for aurora theme"""
        # Main 3D aurora view
        self.ax_aurora.set_facecolor(AURORA_PALETTE['midnight_navy'])
        self.ax_aurora.xaxis.pane.fill = False
        self.ax_aurora.yaxis.pane.fill = False
        self.ax_aurora.zaxis.pane.fill = False
        self.ax_aurora.grid(False)

        # 2D axes
        for ax in [self.ax_solar_wind, self.ax_magnetosphere, self.ax_spectrum,
                   self.ax_atmosphere, self.ax_curtain, self.ax_geomagnetic]:
            ax.set_facecolor(AURORA_PALETTE['midnight_navy'])
            for spine in ax.spines.values():
                spine.set_color(AURORA_PALETTE['stellar_silver'])
                spine.set_linewidth(0.5)
            ax.tick_params(colors=AURORA_PALETTE['ice_crystal'], labelsize=8)

    def _create_aurora_curtains(self):
        """Create multiple aurora curtains"""
        for i in range(5):
            base_altitude = 80 + i * 50 + random.uniform(-20, 20)
            width = random.uniform(40, 80)

            curtain = AuroraCurtain(base_altitude, width)
            self.aurora_curtains.append(curtain)

    def update_aurora_system(self, frame: int):
        """Update the entire aurora system"""
        self.time = frame * 0.05

        # Generate new solar wind particles
        if frame % 3 == 0:  # Every 3rd frame
            n_new = random.randint(3, 8)
            self.solar_wind.generate_particles(n_new)

        # Evolve solar wind
        self.solar_wind.evolve(self.time)

        # Update magnetosphere
        solar_wind_strength = self.solar_wind.wind_speed / 400  # Normalized
        self.magnetosphere.evolve(self.time, solar_wind_strength)

        # Move solar wind particles through magnetosphere
        for particle in self.solar_wind.particles[:]:
            # Check if particle reached Earth region
            distance_to_earth = np.linalg.norm(particle.position)

            if distance_to_earth > 300:  # Particle escaped
                self.solar_wind.particles.remove(particle)
                continue

            # Get magnetic and electric fields
            magnetic_field = self.magnetosphere.get_magnetic_field(particle.position)
            electric_field = np.array([0, 0, 0])  # Simplified

            # Move particle
            particle.move(magnetic_field, electric_field)

            # Check for atmospheric collision
            altitude = particle.position[2]  # z-coordinate as altitude
            if 50 <= altitude <= 500:
                collision = self.atmosphere.particle_collision(particle, altitude)
                if collision:
                    # Particle created aurora emission
                    pass

        # Evolve atmospheric emissions
        self.atmosphere.evolve_emissions(0.05)

        # Animate aurora curtains
        for curtain in self.aurora_curtains:
            curtain.dance(self.time)

        # Update activity indices
        self._update_activity_indices()

        # Clear and redraw
        self._clear_axes()
        self._render_aurora_system()

    def _update_activity_indices(self):
        """Update aurora and geomagnetic activity indices"""
        # Calculate aurora activity based on emissions
        current_intensity = sum(e['intensity'] for e in self.atmosphere.aurora_emissions)
        self.aurora_intensity_history.append(current_intensity)

        # Solar wind activity
        solar_activity = len(self.solar_wind.particles) * self.solar_wind.wind_speed / 400
        self.solar_wind_history.append(solar_activity)

        # Geomagnetic activity (Kp index simulation)
        base_activity = len(self.atmosphere.aurora_emissions) / 10
        cme_boost = self.solar_wind.cme_strength if self.solar_wind.coronal_mass_ejection else 0
        self.geomagnetic_index = min(9, base_activity + cme_boost)

        # Update spectrum data
        for emission in self.atmosphere.aurora_emissions:
            wavelength = emission['wavelength']
            if wavelength > 0:
                self.spectrum_data[wavelength].append(emission['intensity'])

                # Limit history
                if len(self.spectrum_data[wavelength]) > 50:
                    self.spectrum_data[wavelength].pop(0)

    def _clear_axes(self):
        """Clear all axes for redrawing"""
        self.ax_aurora.clear()
        self.ax_solar_wind.clear()
        self.ax_magnetosphere.clear()
        self.ax_spectrum.clear()
        self.ax_atmosphere.clear()
        self.ax_curtain.clear()
        self.ax_geomagnetic.clear()

        self._style_axes()

    def _render_aurora_system(self):
        """Render the complete aurora system"""
        self._render_3d_aurora()
        self._render_solar_wind_monitor()
        self._render_magnetosphere_activity()
        self._render_aurora_spectrum()
        self._render_atmospheric_layers()
        self._render_curtain_profile()
        self._render_geomagnetic_activity()

    def _render_3d_aurora(self):
        """Render main 3D aurora visualization"""
        self.ax_aurora.set_title('Aurora Borealis - The Northern Lights',
                                color=AURORA_PALETTE['ice_crystal'], fontsize=14, pad=20)

        # Render Earth as a sphere
        earth_radius = 10
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
        earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
        earth_z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        self.ax_aurora.plot_surface(earth_x, earth_y, earth_z,
                                   color=AURORA_PALETTE['midnight_navy'], alpha=0.3)

        # Render magnetosphere field lines
        for field_line in self.magnetosphere.field_lines[::2]:  # Show every other line
            if len(field_line['points']) > 1:
                points = np.array(field_line['points'])

                # Field line color based on activity
                alpha = 0.3 + field_line['activity'] * 0.4

                self.ax_aurora.plot(points[:, 0], points[:, 1], points[:, 2],
                                   color=field_line['color'], alpha=alpha, linewidth=1)

        # Render solar wind particles
        for particle in self.solar_wind.particles:
            # Particle position
            x, y, z = particle.position

            # Size based on energy
            size = 20 + particle.energy * 30

            self.ax_aurora.scatter(x, y, z, s=size, c=particle.color, alpha=0.7)

            # Particle trail
            if len(particle.trail) > 1:
                trail_points = np.array(particle.trail[-10:])  # Last 10 points
                self.ax_aurora.plot(trail_points[:, 0], trail_points[:, 1], trail_points[:, 2],
                                   color=particle.color, alpha=0.3, linewidth=1)

        # Render aurora emissions
        for emission in self.atmosphere.aurora_emissions:
            pos = emission['position']
            intensity = emission['intensity']

            if intensity > 0.1:  # Only render visible emissions
                size = 50 + intensity * 200
                alpha = min(1.0, intensity * 2)

                self.ax_aurora.scatter(pos[0], pos[1], pos[2],
                                      s=size, c=emission['color'], alpha=alpha)

        # Render aurora curtains
        for curtain in self.aurora_curtains:
            for i, vertical_line in enumerate(curtain.curtain_points):
                if len(vertical_line) > 1:
                    points = np.array(vertical_line)
                    intensities = curtain.intensity_profile[i]
                    colors = curtain.color_profile[i]

                    # Draw curtain segments
                    for j in range(len(points) - 1):
                        if intensities[j] > 0.1:  # Only visible segments
                            start = points[j]
                            end = points[j + 1]

                            alpha = intensities[j] * 0.8
                            linewidth = 2 + intensities[j] * 3

                            self.ax_aurora.plot([start[0], end[0]],
                                              [start[1], end[1]],
                                              [start[2], end[2]],
                                              color=colors[j], alpha=alpha,
                                              linewidth=linewidth)

        # Add some stars for atmosphere
        for _ in range(50):
            star_x = random.uniform(-150, 150)
            star_y = random.uniform(-150, 150)
            star_z = random.uniform(200, 400)

            self.ax_aurora.scatter(star_x, star_y, star_z, s=10,
                                  c=AURORA_PALETTE['stellar_silver'], alpha=0.6, marker='*')

        # Set 3D limits
        self.ax_aurora.set_xlim(-150, 150)
        self.ax_aurora.set_ylim(-150, 150)
        self.ax_aurora.set_zlim(-50, 300)

        # Remove axis labels
        self.ax_aurora.set_xticks([])
        self.ax_aurora.set_yticks([])
        self.ax_aurora.set_zticks([])

    def _render_solar_wind_monitor(self):
        """Render solar wind conditions"""
        self.ax_solar_wind.set_title('Solar Wind Monitor',
                                    color=AURORA_PALETTE['ice_crystal'], fontsize=12)

        if len(self.solar_wind_history) > 1:
            time_axis = range(len(self.solar_wind_history))

            # Solar wind speed
            wind_speeds = [self.solar_wind.wind_speed] * len(self.solar_wind_history)
            self.ax_solar_wind.fill_between(time_axis, 0, wind_speeds,
                                           color=AURORA_PALETTE['solar_peach'], alpha=0.3)
            self.ax_solar_wind.plot(time_axis, wind_speeds,
                                   color=AURORA_PALETTE['solar_peach'], linewidth=2)

            # Particle density
            density_scaled = [self.solar_wind.density * 50] * len(self.solar_wind_history)
            self.ax_solar_wind.plot(time_axis, density_scaled,
                                   color=AURORA_PALETTE['aurora_amber'], linewidth=2, linestyle='--')

            # CME indicator
            if self.solar_wind.coronal_mass_ejection:
                cme_level = 400 + self.solar_wind.cme_strength * 100
                self.ax_solar_wind.axhline(y=cme_level, color=AURORA_PALETTE['aurora_rose'],
                                          linewidth=3, alpha=0.8, label='CME')

                # Add CME burst effect
                for _ in range(int(self.solar_wind.cme_strength * 5)):
                    burst_x = random.choice(time_axis[-10:]) if len(time_axis) >= 10 else 0
                    burst_y = random.uniform(cme_level - 50, cme_level + 50)
                    self.ax_solar_wind.scatter(burst_x, burst_y, s=30,
                                              c=AURORA_PALETTE['aurora_rose'], alpha=0.7, marker='*')

        self.ax_solar_wind.set_ylabel('Speed (km/s)', color=AURORA_PALETTE['ice_crystal'], fontsize=9)
        self.ax_solar_wind.set_ylim(0, 800)

        # Add legend
        if hasattr(self, '_solar_legend_added'):
            pass
        else:
            self.ax_solar_wind.text(0.02, 0.98, 'Wind Speed\nDensity (×50)\nCME Events',
                                   transform=self.ax_solar_wind.transAxes,
                                   color=AURORA_PALETTE['ice_crystal'], fontsize=8,
                                   verticalalignment='top')
            self._solar_legend_added = True

    def _render_magnetosphere_activity(self):
        """Render magnetosphere field activity"""
        self.ax_magnetosphere.set_title('Magnetosphere Activity',
                                       color=AURORA_PALETTE['ice_crystal'], fontsize=12)

        # Substorm activity indicator
        substorm_level = self.magnetosphere.substorm_activity

        # Create substorm visualization
        theta = np.linspace(0, 2*np.pi, 100)
        for i in range(3):
            radius = (i + 1) * substorm_level * 0.3
            x = radius * np.cos(theta + self.time * (i + 1) * 0.5)
            y = radius * np.sin(theta + self.time * (i + 1) * 0.5)

            colors = [AURORA_PALETTE['magnetic_mauve'],
                     AURORA_PALETTE['twilight_periwinkle'],
                     AURORA_PALETTE['polar_sage']]

            self.ax_magnetosphere.fill(x, y, color=colors[i], alpha=0.3)
            self.ax_magnetosphere.plot(x, y, color=colors[i], linewidth=2, alpha=0.6)

        # Ring current particles
        for current_system in self.magnetosphere.current_systems:
            if current_system['type'] == 'ring':
                for particle in current_system['particles'][:10]:  # Show subset
                    x, y = particle.position[0] * 0.01, particle.position[1] * 0.01  # Scale down
                    self.ax_magnetosphere.scatter(x, y, s=20, c=particle.color, alpha=0.7)

        self.ax_magnetosphere.set_xlim(-2, 2)
        self.ax_magnetosphere.set_ylim(-2, 2)
        self.ax_magnetosphere.set_aspect('equal')
        self.ax_magnetosphere.set_xticks([])
        self.ax_magnetosphere.set_yticks([])

        # Activity level text
        activity_text = f'Substorm Level: {substorm_level:.2f}'
        self.ax_magnetosphere.text(0.02, 0.98, activity_text,
                                  transform=self.ax_magnetosphere.transAxes,
                                  color=AURORA_PALETTE['ice_crystal'], fontsize=9,
                                  verticalalignment='top')

    def _render_aurora_spectrum(self):
        """Render aurora emission spectrum"""
        self.ax_spectrum.set_title('Aurora Emission Spectrum',
                                  color=AURORA_PALETTE['ice_crystal'], fontsize=12)

        if self.spectrum_data:
            wavelengths = list(self.spectrum_data.keys())
            intensities = [np.mean(self.spectrum_data[wl][-10:])
                          if self.spectrum_data[wl] else 0 for wl in wavelengths]

            # Color map for emission lines
            colors = []
            for wl in wavelengths:
                if 550 < wl < 570:  # Green oxygen line
                    colors.append(AURORA_PALETTE['oxygen_jade'])
                elif 620 < wl < 640:  # Red oxygen line
                    colors.append(AURORA_PALETTE['aurora_rose'])
                elif 420 < wl < 440:  # Blue nitrogen line
                    colors.append(AURORA_PALETTE['northern_lavender'])
                else:
                    colors.append(AURORA_PALETTE['polar_aqua'])

            # Bar chart of emission lines
            bars = self.ax_spectrum.bar(range(len(wavelengths)), intensities,
                                       color=colors, alpha=0.8, width=0.6)

            # Add glow effects for strong lines
            for i, (bar, intensity) in enumerate(zip(bars, intensities, strict=False)):
                if intensity > 0.5:
                    glow_height = bar.get_height()
                    glow_x = bar.get_x() + bar.get_width()/2

                    # Add glow
                    self.ax_spectrum.scatter(glow_x, glow_height, s=intensity*200,
                                           c=colors[i], alpha=0.3)

            # Wavelength labels
            if wavelengths:
                self.ax_spectrum.set_xticks(range(len(wavelengths)))
                wl_labels = [f'{wl:.0f}' for wl in wavelengths]
                self.ax_spectrum.set_xticklabels(wl_labels, rotation=45, fontsize=8)

        self.ax_spectrum.set_ylabel('Intensity', color=AURORA_PALETTE['ice_crystal'], fontsize=9)
        self.ax_spectrum.set_xlabel('Wavelength (nm)', color=AURORA_PALETTE['ice_crystal'], fontsize=9)

    def _render_atmospheric_layers(self):
        """Render atmospheric layers and density profile"""
        self.ax_atmosphere.set_title('Atmospheric Layers',
                                    color=AURORA_PALETTE['ice_crystal'], fontsize=12)

        # Altitude profile
        altitudes = np.linspace(50, 500, 100)
        thermosphere_density = [self.atmosphere._thermosphere_density(alt)
                               if alt >= 80 else 0 for alt in altitudes]
        mesosphere_density = [self.atmosphere._mesosphere_density(alt)
                             if alt < 80 else 0 for alt in altitudes]

        # Log scale for density
        thermo_log = [np.log10(d + 1e-20) for d in thermosphere_density]
        meso_log = [np.log10(d + 1e-20) for d in mesosphere_density]

        # Fill atmospheric layers
        self.ax_atmosphere.fill_between(thermo_log, altitudes,
                                       color=AURORA_PALETTE['aurora_rose'], alpha=0.3,
                                       label='Thermosphere')
        self.ax_atmosphere.fill_between(meso_log, altitudes,
                                       color=AURORA_PALETTE['polar_aqua'], alpha=0.3,
                                       label='Mesosphere')

        # Mark aurora altitude range
        self.ax_atmosphere.axhspan(80, 500, color=AURORA_PALETTE['oxygen_jade'],
                                  alpha=0.1, label='Aurora Zone')

        # Show current aurora emissions
        for emission in self.atmosphere.aurora_emissions[-10:]:  # Recent emissions
            alt = emission['altitude']
            intensity_scaled = emission['intensity'] * 2 - 20  # Scale for x-axis

            self.ax_atmosphere.scatter(intensity_scaled, alt, s=50,
                                      c=emission['color'], alpha=0.8, marker='o')

        self.ax_atmosphere.set_xlabel('Log Density / Intensity',
                                     color=AURORA_PALETTE['ice_crystal'], fontsize=9)
        self.ax_atmosphere.set_ylabel('Altitude (km)',
                                     color=AURORA_PALETTE['ice_crystal'], fontsize=9)
        self.ax_atmosphere.set_ylim(50, 500)
        self.ax_atmosphere.legend(fontsize=8, framealpha=0.3)

    def _render_curtain_profile(self):
        """Render aurora curtain intensity profile"""
        self.ax_curtain.set_title('Aurora Curtain Profile',
                                 color=AURORA_PALETTE['ice_crystal'], fontsize=12)

        if self.aurora_curtains:
            # Show intensity profile of first curtain
            curtain = self.aurora_curtains[0]

            for i, (intensities, colors) in enumerate(zip(curtain.intensity_profile,
                                                         curtain.color_profile, strict=False)):
                if i % 3 == 0:  # Show every 3rd profile line
                    altitudes = np.linspace(curtain.base_altitude,
                                          curtain.base_altitude + curtain.height,
                                          len(intensities))

                    # Offset x position for each profile
                    x_offset = i * 0.1
                    x_values = [x_offset + intensity for intensity in intensities]

                    # Draw profile
                    for j in range(len(altitudes) - 1):
                        self.ax_curtain.plot([x_values[j], x_values[j+1]],
                                           [altitudes[j], altitudes[j+1]],
                                           color=colors[j], alpha=0.7, linewidth=2)

                        # Add glow for bright regions
                        if intensities[j] > 0.7:
                            self.ax_curtain.scatter(x_values[j], altitudes[j], s=100,
                                                   c=colors[j], alpha=0.3)

        self.ax_curtain.set_xlabel('Intensity + Position',
                                  color=AURORA_PALETTE['ice_crystal'], fontsize=9)
        self.ax_curtain.set_ylabel('Altitude (km)',
                                  color=AURORA_PALETTE['ice_crystal'], fontsize=9)
        self.ax_curtain.set_xlim(0, 2)

    def _render_geomagnetic_activity(self):
        """Render geomagnetic activity indices"""
        self.ax_geomagnetic.set_title('Geomagnetic Activity',
                                     color=AURORA_PALETTE['ice_crystal'], fontsize=12)

        # Kp index visualization
        kp_levels = np.arange(10)
        kp_colors = [AURORA_PALETTE['ice_crystal'] if i <= self.geomagnetic_index
                    else AURORA_PALETTE['midnight_navy'] for i in kp_levels]

        bars = self.ax_geomagnetic.barh(kp_levels, np.ones(10),
                                       color=kp_colors, alpha=0.8)

        # Highlight current Kp level
        if self.geomagnetic_index < 9:
            bars[int(self.geomagnetic_index)].set_color(AURORA_PALETTE['aurora_rose'])

        # Activity level indicators
        activity_levels = ['Quiet', 'Unsettled', 'Active', 'Minor Storm', 'Major Storm', 'Severe Storm']
        if self.geomagnetic_index <= 2:
            activity = activity_levels[0]
            activity_color = AURORA_PALETTE['polar_sage']
        elif self.geomagnetic_index <= 4:
            activity = activity_levels[1]
            activity_color = AURORA_PALETTE['aurora_amber']
        elif self.geomagnetic_index <= 5:
            activity = activity_levels[2]
            activity_color = AURORA_PALETTE['northern_lavender']
        elif self.geomagnetic_index <= 6:
            activity = activity_levels[3]
            activity_color = AURORA_PALETTE['aurora_rose']
        elif self.geomagnetic_index <= 8:
            activity = activity_levels[4]
            activity_color = AURORA_PALETTE['nitrogen_blush']
        else:
            activity = activity_levels[5]
            activity_color = AURORA_PALETTE['plasma_pink']

        # Add activity sparkles for high activity
        if self.geomagnetic_index > 6:
            for _ in range(int(self.geomagnetic_index)):
                sparkle_x = random.uniform(0.5, 1.5)
                sparkle_y = random.uniform(0, 9)
                self.ax_geomagnetic.scatter(sparkle_x, sparkle_y, s=20,
                                           c=activity_color, alpha=0.8, marker='*')

        self.ax_geomagnetic.set_xlim(0, 2)
        self.ax_geomagnetic.set_ylim(-0.5, 9.5)
        self.ax_geomagnetic.set_yticks(kp_levels)
        self.ax_geomagnetic.set_ylabel('Kp Index', color=AURORA_PALETTE['ice_crystal'], fontsize=9)

        # Activity level text
        self.ax_geomagnetic.text(0.02, 0.98, f'{activity}\nKp = {self.geomagnetic_index:.1f}',
                                transform=self.ax_geomagnetic.transAxes,
                                color=activity_color, fontsize=9,
                                verticalalignment='top', weight='bold')

    def animate(self):
        """Start the aurora visualization animation"""
        def update(frame):
            try:
                self.update_aurora_system(frame)
            except Exception as e:
                print(f"Animation error at frame {frame}: {e}")
            return []

        self.anim = animation.FuncAnimation(
            self.fig, update,
            frames=2000,
            interval=60,
            blit=False,
            repeat=True
        )

        plt.show()


def run_aurora_symphony():
    """Launch the Aurora Quantum Symphony"""
    print("Aurora Quantum Symphony - Northern Lights Visualization")
    print()
    print("Features:")
    print("• Solar wind particles streaming from the Sun")
    print("• Earth's magnetosphere with dynamic field lines")
    print("• Atmospheric layers where aurora emissions occur")
    print("• Dancing aurora curtains with realistic motion")
    print("• Spectral analysis of emission wavelengths")
    print("• Geomagnetic activity monitoring")
    print("• Coronal mass ejection events")
    print("• 24 ethereal pastel colors capturing arctic beauty")
    print()
    print("Watch as charged particles from the Sun dance through Earth's")
    print("magnetic field, creating the magical Northern Lights...")

    try:
        aurora_system = AuroraVisualizer()
        aurora_system.animate()
    except Exception as e:
        print(f"Error launching aurora visualization: {e}")
        print("Please ensure all dependencies are installed")


if __name__ == "__main__":
    run_aurora_symphony()
