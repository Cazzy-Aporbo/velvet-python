"""
QUANTUM INFORMATION ECOLOGY 2025
Where Information Lives, Evolves, and Forms Consciousness
Information species, data food webs, entropy flows, quantum entanglement networks,
emergent intelligence, information metabolism, and digital evolution
Cazzy Aporbo, MS
 Information theory meets artificial life in visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon, FancyBboxPatch, Wedge
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import deque, defaultdict, Counter
from enum import Enum, auto
import math
import random
import colorsys
import itertools

# Information Ecology Palette - 40+ unique colors representing different information types
INFO_PALETTE = {
    # Base Information Types
    'raw_data': '#FF6B6B',           # Raw unprocessed data
    'structured_info': '#4ECDC4',    # Organized information
    'knowledge': '#45B7D1',          # Processed knowledge
    'wisdom': '#96CEB4',             # Refined wisdom
    'consciousness': '#FFEAA7',      # Emergent consciousness
    
    # Information Species Types
    'binary_bits': '#DDA0DD',        # Binary information
    'quantum_qubits': '#98FB98',     # Quantum information
    'neural_signals': '#F0E68C',     # Neural network data
    'genetic_code': '#FFB6C1',       # Genetic information
    'linguistic_tokens': '#87CEEB',  # Language data
    'sensory_inputs': '#F4A460',     # Sensory information
    'memory_traces': '#D8BFD8',      # Memory patterns
    'emotional_data': '#FFA07A',     # Emotional information
    
    # Information States
    'entropy_high': '#FF4500',       # High entropy (chaos)
    'entropy_med': '#FF8C00',        # Medium entropy
    'entropy_low': '#FFD700',        # Low entropy (order)
    'information_flow': '#00CED1',   # Information transfer
    'mutual_info': '#9370DB',        # Mutual information
    'redundancy': '#DC143C',         # Redundant information
    'compression': '#32CD32',        # Compressed data
    'encryption': '#8B4513',         # Encrypted information
    
    # Ecosystem Elements
    'predator_virus': '#B22222',     # Information viruses
    'symbiont_helper': '#228B22',    # Helpful symbionts
    'neutral_drift': '#708090',      # Neutral evolution
    'adaptive_mutation': '#FF1493',  # Adaptive changes
    'selection_pressure': '#4B0082', # Selection forces
    'fitness_landscape': '#006400',  # Fitness terrain
    'niche_habitat': '#8FBC8F',      # Information niches
    'resource_pool': '#20B2AA',      # Information resources
    
    # Emergent Phenomena
    'swarm_intelligence': '#FF69B4', # Collective intelligence
    'network_topology': '#7B68EE',   # Network structures
    'phase_transition': '#FF6347',   # Critical transitions
    'autocatalysis': '#00FF7F',      # Self-reinforcing loops
    'emergence_signal': '#FFD700',   # Emergent properties
    'complexity_growth': '#DA70D6',  # Growing complexity
    'self_organization': '#40E0D0',  # Self-organizing patterns
    'critical_point': '#FF0000',     # Critical phase points
    
    # Quantum Effects
    'entanglement': '#E6E6FA',       # Quantum entanglement
    'superposition': '#F0F8FF',      # Quantum superposition
    'decoherence': '#FFFACD',        # Quantum decoherence
    'measurement': '#F5DEB3',        # Quantum measurement
    'uncertainty': '#E0E0E0',        # Quantum uncertainty
    'tunnel_effect': '#DCDCDC',      # Quantum tunneling
    'field_fluctuation': '#D3D3D3',  # Quantum field fluctuations - FIXED
    
    # Background and Structure
    'void_background': '#000011',     # Deep information void
    'substrate_layer': '#1A1A2E',     # Information substrate
    'boundary_membrane': '#16213E',   # System boundaries
    'connection_pathway': '#0F3460'   # Information pathways
}

@dataclass 
class InfoParticle:
    """Individual information particle with properties and behaviors"""
    
    particle_id: str
    position: np.ndarray
    velocity: np.ndarray
    info_type: str  # Type of information this particle represents
    entropy: float  # Information entropy content
    complexity: float  # Structural complexity
    age: float = 0.0
    energy: float = 1.0
    fitness: float = 1.0
    genome: List[int] = field(default_factory=lambda: [random.randint(0, 1) for _ in range(32)])
    connections: Set[str] = field(default_factory=set)
    reproduction_cooldown: float = 0.0
    mutation_rate: float = 0.01
    size: float = 1.0
    color: str = '#FF6B6B'
    
    def __post_init__(self):
        if self.position.shape[0] != 3:
            self.position = np.random.randn(3) * 50
        if self.velocity.shape[0] != 3:
            self.velocity = np.random.randn(3) * 0.5
        self.color = INFO_PALETTE.get(self.info_type, '#FF6B6B')
    
    def calculate_entropy(self) -> float:
        """Calculate information entropy based on genome and state"""
        # Shannon entropy of genome
        if not self.genome:
            return 0.0
        
        counts = Counter(self.genome)
        total = len(self.genome)
        entropy = 0.0
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p) if p > 0 else 0  # Added safety check
        
        # Add state-based entropy
        state_entropy = self.complexity * 0.1 + np.sin(self.age * 0.1) * 0.2
        
        return max(0, min(10, entropy + state_entropy))
    
    def calculate_mutual_information(self, other: 'InfoParticle') -> float:
        """Calculate mutual information with another particle"""
        if not self.genome or not other.genome:
            return 0.0
        
        # Simplified mutual information calculation
        min_len = min(len(self.genome), len(other.genome))
        if min_len == 0:
            return 0.0
            
        shared_bits = sum(1 for a, b in zip(self.genome[:min_len], other.genome[:min_len]) if a == b)
        
        return (shared_bits / min_len) * np.log2(max(2, min_len))  # Added max to prevent log(0)
    
    def reproduce(self, partner: 'InfoParticle') -> Optional['InfoParticle']:
        """Create offspring through information crossover"""
        if self.reproduction_cooldown > 0 or partner.reproduction_cooldown > 0:
            return None
        
        if not self.genome or not partner.genome:
            return None
            
        # Genetic crossover
        crossover_point = random.randint(1, max(1, min(len(self.genome), len(partner.genome)) - 1))
        child_genome = self.genome[:crossover_point] + partner.genome[crossover_point:]
        
        # Mutations
        for i in range(len(child_genome)):
            if random.random() < self.mutation_rate:
                child_genome[i] = 1 - child_genome[i]  # Bit flip
        
        # Inherit properties
        child_position = (self.position + partner.position) / 2 + np.random.randn(3) * 2
        child_velocity = (self.velocity + partner.velocity) / 2 + np.random.randn(3) * 0.1
        
        # Information type evolution
        parent_types = [self.info_type, partner.info_type]
        if random.random() < 0.1:  # 10% chance of type evolution
            child_type = random.choice(list(INFO_PALETTE.keys()))
        else:
            child_type = random.choice(parent_types)
        
        child = InfoParticle(
            particle_id=f"child_{random.randint(1000, 9999)}",
            position=child_position,
            velocity=child_velocity,
            info_type=child_type,
            entropy=(self.entropy + partner.entropy) / 2,
            complexity=(self.complexity + partner.complexity) / 2,
            genome=child_genome,
            mutation_rate=(self.mutation_rate + partner.mutation_rate) / 2
        )
        
        # Set reproduction cooldown
        self.reproduction_cooldown = 5.0
        partner.reproduction_cooldown = 5.0
        
        return child
    
    def metabolize_information(self, available_info: float) -> float:
        """Process available information for energy"""
        # Information metabolism - convert raw info to energy
        intake_rate = self.complexity * 0.1
        processed = min(available_info, intake_rate)
        
        self.energy += processed * 0.5
        self.entropy = self.calculate_entropy()
        
        return processed
    
    def interact_with(self, other: 'InfoParticle', interaction_type: str) -> bool:
        """Interact with another particle"""
        try:
            distance = np.linalg.norm(self.position - other.position)
            
            if distance > 20:  # Too far to interact
                return False
            
            if interaction_type == 'cooperation':
                # Mutual benefit
                info_transfer = 0.1
                self.energy += info_transfer
                other.energy += info_transfer
                self.connections.add(other.particle_id)
                other.connections.add(self.particle_id)
                return True
                
            elif interaction_type == 'competition':
                # Zero-sum competition
                if self.fitness > other.fitness:
                    energy_stolen = 0.2
                    self.energy += energy_stolen
                    other.energy = max(0, other.energy - energy_stolen)
                    return True
                return False
                
            elif interaction_type == 'predation':
                # Predator-prey dynamics
                if self.info_type == 'predator_virus' and other.info_type != 'predator_virus':
                    if self.size > other.size * 0.5:
                        self.energy += other.energy * 0.7
                        other.energy = 0  # Consumed
                        return True
                return False
        except:
            return False
        
        return False
    
    def evolve(self, dt: float, environment_pressure: Dict[str, float]):
        """Evolve particle properties based on environment"""
        # Age and energy decay
        self.age += dt
        self.energy *= 0.999  # Gradual energy decay
        
        # Reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= dt
        
        # Environmental adaptation
        pressure = environment_pressure.get(self.info_type, 0.5)
        
        # Fitness calculation
        self.fitness = (self.energy * 0.4 + 
                       (10 - self.entropy) * 0.3 + 
                       self.complexity * 0.2 + 
                       len(self.connections) * 0.1) * (1 + pressure)
        
        # Size based on fitness and energy
        self.size = max(0.5, min(3.0, self.fitness * 0.5 + self.energy * 0.1))
        
        # Death condition
        return self.energy > 0.1 and self.age < 100


class InformationEcosystem:
    """Ecosystem managing information particle populations and interactions"""
    
    def __init__(self, carrying_capacity: int = 300):  # Reduced from 500
        self.carrying_capacity = carrying_capacity
        self.particles = []
        self.resource_grid = np.ones((50, 50)) * 10  # Information resource distribution
        self.environment_pressures = defaultdict(lambda: 0.5)
        self.interaction_network = defaultdict(list)
        self.ecosystem_age = 0
        self.total_information = 0
        self.diversity_index = 0
        self.emergence_events = deque(maxlen=100)
        
        # Initialize with diverse particle types
        self._seed_initial_population()
    
    def _seed_initial_population(self):
        """Create initial diverse population"""
        initial_types = [
            'raw_data', 'structured_info', 'knowledge', 'binary_bits',
            'quantum_qubits', 'neural_signals', 'genetic_code', 'linguistic_tokens'
        ]
        
        for _ in range(50):  # Reduced from 100
            particle_type = random.choice(initial_types)
            position = np.random.uniform(-50, 50, 3)
            velocity = np.random.uniform(-1, 1, 3)
            
            particle = InfoParticle(
                particle_id=f"seed_{random.randint(1000, 9999)}",
                position=position,
                velocity=velocity,
                info_type=particle_type,
                entropy=random.uniform(1, 8),
                complexity=random.uniform(0.5, 2.0)
            )
            
            self.particles.append(particle)
    
    def update_environment(self, dt: float):
        """Update environmental conditions and pressures"""
        try:
            self.ecosystem_age += dt
            
            # Resource regeneration with spatial patterns
            resource_growth = 0.1 * dt
            noise = np.random.randn(50, 50) * 0.05
            self.resource_grid += resource_growth + noise
            self.resource_grid = np.clip(self.resource_grid, 0, 15)
            
            # Add resource hotspots
            if random.random() < 0.01:
                x, y = random.randint(5, 44), random.randint(5, 44)
                self.resource_grid[x-2:x+3, y-2:y+3] += 5
            
            # Dynamic selection pressures
            time_factor = self.ecosystem_age * 0.01
            for info_type in list(INFO_PALETTE.keys())[:20]:  # Limit types
                base_pressure = 0.5
                oscillation = 0.3 * np.sin(time_factor + hash(info_type) % 100)
                noise_pressure = random.uniform(-0.1, 0.1)
                
                self.environment_pressures[info_type] = max(0, min(1, 
                    base_pressure + oscillation + noise_pressure))
            
            # Calculate ecosystem metrics
            self._calculate_ecosystem_metrics()
        except Exception as e:
            print(f"Environment update error: {e}")
    
    def _calculate_ecosystem_metrics(self):
        """Calculate ecosystem-wide metrics"""
        try:
            if not self.particles:
                self.total_information = 0
                self.diversity_index = 0
                return
            
            # Total information content
            self.total_information = sum(p.entropy for p in self.particles)
            
            # Diversity index (Shannon diversity)
            type_counts = Counter(p.info_type for p in self.particles)
            total_particles = len(self.particles)
            
            if total_particles > 0:
                diversity = 0
                for count in type_counts.values():
                    if count > 0:
                        p = count / total_particles
                        diversity -= p * np.log(p) if p > 0 else 0
                self.diversity_index = diversity
            
            # Detect emergence events
            if len(self.particles) > 100 and random.random() < 0.002:  # Reduced frequency
                self._trigger_emergence_event()
        except Exception as e:
            print(f"Metrics calculation error: {e}")
    
    def _trigger_emergence_event(self):
        """Trigger emergent phenomena in the ecosystem"""
        try:
            event_types = ['phase_transition', 'swarm_intelligence', 'self_organization', 
                          'autocatalysis', 'network_topology', 'complexity_growth']
            
            event_type = random.choice(event_types)
            event_location = np.random.uniform(-40, 40, 3)
            
            self.emergence_events.append({
                'type': event_type,
                'location': event_location,
                'strength': random.uniform(0.5, 2.0),
                'age': 0,
                'max_age': random.uniform(10, 30)
            })
            
            # Create specialized particles for the event
            for _ in range(random.randint(3, 8)):  # Reduced particle count
                particle = InfoParticle(
                    particle_id=f"emerge_{random.randint(1000, 9999)}",
                    position=event_location + np.random.randn(3) * 5,
                    velocity=np.random.randn(3) * 0.3,
                    info_type=event_type,
                    entropy=random.uniform(0.5, 3.0),
                    complexity=random.uniform(1.5, 4.0)
                )
                if len(self.particles) < self.carrying_capacity:
                    self.particles.append(particle)
        except Exception as e:
            print(f"Emergence event error: {e}")
    
    def simulate_step(self, dt: float):
        """Run one simulation step"""
        try:
            self.update_environment(dt)
            
            # Update all particles
            surviving_particles = []
            
            for particle in self.particles:
                # Particle evolution
                if particle.evolve(dt, self.environment_pressures):
                    # Move particle
                    particle.position += particle.velocity * dt
                    
                    # Boundary conditions (toroidal)
                    particle.position = np.mod(particle.position + 50, 100) - 50
                    
                    # Resource consumption
                    grid_x = int(np.clip((particle.position[0] + 50) / 2, 0, 49))
                    grid_y = int(np.clip((particle.position[1] + 50) / 2, 0, 49))
                    
                    available_resource = self.resource_grid[grid_x, grid_y]
                    consumed = particle.metabolize_information(available_resource)
                    self.resource_grid[grid_x, grid_y] = max(0, available_resource - consumed)
                    
                    surviving_particles.append(particle)
            
            self.particles = surviving_particles
            
            # Interactions between particles (limited)
            if len(self.particles) < 200:  # Only process if not too many
                self._process_interactions()
            
            # Reproduction (limited)
            if len(self.particles) < self.carrying_capacity * 0.8:
                self._process_reproduction()
            
            # Population control
            if len(self.particles) > self.carrying_capacity:
                # Remove least fit particles
                self.particles.sort(key=lambda p: p.fitness, reverse=True)
                self.particles = self.particles[:self.carrying_capacity]
            
            # Update emergence events
            for event in self.emergence_events:
                event['age'] += dt
            
            self.emergence_events = deque([e for e in self.emergence_events 
                                         if e['age'] < e['max_age']], maxlen=100)
        except Exception as e:
            print(f"Simulation step error: {e}")
    
    def _process_interactions(self):
        """Process interactions between nearby particles"""
        try:
            if len(self.particles) < 2:
                return
            
            # Limit interactions to prevent performance issues
            max_interactions = min(50, len(self.particles) * 2)
            interaction_count = 0
            
            for i, particle1 in enumerate(self.particles):
                if interaction_count >= max_interactions:
                    break
                    
                # Only check a few nearby particles
                for j in range(i+1, min(i+5, len(self.particles))):
                    if interaction_count >= max_interactions:
                        break
                        
                    particle2 = self.particles[j]
                    distance = np.linalg.norm(particle1.position - particle2.position)
                    
                    if distance < 15:  # Interaction range
                        interaction_type = self._determine_interaction_type(particle1, particle2)
                        if particle1.interact_with(particle2, interaction_type):
                            interaction_count += 1
        except Exception as e:
            print(f"Interaction processing error: {e}")
    
    def _determine_interaction_type(self, p1: InfoParticle, p2: InfoParticle) -> str:
        """Determine interaction type between two particles"""
        # Predation
        if 'predator' in p1.info_type or 'virus' in p1.info_type:
            return 'predation'
        
        # Cooperation for similar types
        if p1.info_type == p2.info_type:
            return 'cooperation'
        
        # Competition for resources
        if abs(p1.complexity - p2.complexity) < 0.5:
            return 'competition'
        
        # Default cooperation
        return 'cooperation'
    
    def _process_reproduction(self):
        """Handle particle reproduction"""
        try:
            new_particles = []
            max_reproductions = 5  # Limit reproductions per step
            
            for i, particle1 in enumerate(self.particles):
                if len(new_particles) >= max_reproductions:
                    break
                    
                if (particle1.reproduction_cooldown <= 0 and 
                    particle1.energy > 2.0 and 
                    particle1.fitness > 0.8):
                    
                    # Find suitable partner
                    for particle2 in self.particles[i+1:min(i+10, len(self.particles))]:
                        if (particle2.reproduction_cooldown <= 0 and
                            particle2.energy > 2.0 and
                            np.linalg.norm(particle1.position - particle2.position) < 10):
                            
                            # Check compatibility
                            mutual_info = particle1.calculate_mutual_information(particle2)
                            if mutual_info > 0.3:  # Sufficient compatibility
                                child = particle1.reproduce(particle2)
                                if child and len(self.particles) + len(new_particles) < self.carrying_capacity:
                                    new_particles.append(child)
                                    break
            
            self.particles.extend(new_particles)
        except Exception as e:
            print(f"Reproduction error: {e}")


class QuantumInfoEcologyVisualizer:
    """Main visualization system for quantum information ecology"""
    
    def __init__(self, figsize: Tuple[int, int] = (20, 12)):  # Slightly smaller
        # Setup figure
        self.fig = plt.figure(figsize=figsize, facecolor=INFO_PALETTE['void_background'])
        self.fig.suptitle('Quantum Information Ecology - Where Data Lives and Evolves', 
                         fontsize=18, color=INFO_PALETTE['consciousness'], fontweight='bold')
        
        # Create simplified layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main ecosystem view (large central panel)
        self.ax_ecosystem = self.fig.add_subplot(gs[0:2, 0:3], projection='3d')
        
        # Information flow network (top right)
        self.ax_network = self.fig.add_subplot(gs[0, 3])
        
        # Species diversity (middle right)
        self.ax_diversity = self.fig.add_subplot(gs[1, 3])
        
        # Population dynamics (bottom left)
        self.ax_population = self.fig.add_subplot(gs[2, 0])
        
        # Information metabolism (bottom center)
        self.ax_metabolism = self.fig.add_subplot(gs[2, 1])
        
        # Fitness landscape (bottom right)
        self.ax_fitness = self.fig.add_subplot(gs[2, 2])
        
        # Quantum effects (bottom far right)
        self.ax_quantum = self.fig.add_subplot(gs[2, 3], projection='polar')
        
        # Style all axes
        self._style_axes()
        
        # Initialize ecosystem
        self.ecosystem = InformationEcosystem(carrying_capacity=200)
        
        # Animation state
        self.time = 0
        self.frame_count = 0
        
        # Data tracking
        self.population_history = deque(maxlen=100)
        self.diversity_history = deque(maxlen=100)
        self.entropy_history = deque(maxlen=100)
        
    def _style_axes(self):
        """Style all axes for information ecology theme"""
        # Main 3D ecosystem
        self.ax_ecosystem.set_facecolor(INFO_PALETTE['void_background'])
        self.ax_ecosystem.xaxis.pane.fill = False
        self.ax_ecosystem.yaxis.pane.fill = False
        self.ax_ecosystem.zaxis.pane.fill = False
        self.ax_ecosystem.grid(False)
        
        # 2D axes
        for ax in [self.ax_network, self.ax_diversity, 
                   self.ax_population, self.ax_metabolism, self.ax_fitness]:
            ax.set_facecolor(INFO_PALETTE['substrate_layer'])
            for spine in ax.spines.values():
                spine.set_color(INFO_PALETTE['structured_info'])
                spine.set_linewidth(0.8)
            ax.tick_params(colors=INFO_PALETTE['consciousness'], labelsize=8)
        
        # Quantum polar plot
        self.ax_quantum.set_facecolor(INFO_PALETTE['void_background'])
        self.ax_quantum.grid(True, alpha=0.3, color=INFO_PALETTE['entanglement'])
    
    def update_visualization(self, frame: int):
        """Update the entire visualization"""
        try:
            self.frame_count = frame
            self.time = frame * 0.1
            
            # Simulate ecosystem
            self.ecosystem.simulate_step(0.1)
            
            # Track metrics
            self.population_history.append(len(self.ecosystem.particles))
            self.diversity_history.append(self.ecosystem.diversity_index)
            self.entropy_history.append(self.ecosystem.total_information)
            
            # Clear and redraw
            self._clear_axes()
            self._render_complete_ecology()
        except Exception as e:
            print(f"Visualization update error at frame {frame}: {e}")
    
    def _clear_axes(self):
        """Clear all axes"""
        self.ax_ecosystem.clear()
        self.ax_network.clear()
        self.ax_diversity.clear()
        self.ax_population.clear()
        self.ax_metabolism.clear()
        self.ax_fitness.clear()
        self.ax_quantum.clear()
        
        self._style_axes()
    
    def _render_complete_ecology(self):
        """Render the complete information ecology"""
        try:
            self._render_3d_ecosystem()
            self._render_information_network()
            self._render_species_diversity()
            self._render_population_dynamics()
            self._render_information_metabolism()
            self._render_fitness_landscape()
            self._render_quantum_effects()
        except Exception as e:
            print(f"Rendering error: {e}")
    
    def _render_3d_ecosystem(self):
        """Render main 3D information ecosystem"""
        try:
            self.ax_ecosystem.set_title('Living Information Ecosystem', 
                                       color=INFO_PALETTE['consciousness'], fontsize=14, pad=20)
            
            if not self.ecosystem.particles:
                return
                
            # Limit particles to render for performance
            particles_to_render = self.ecosystem.particles[:150]
            
            # Collect data for batch rendering
            positions = []
            colors = []
            sizes = []
            
            for particle in particles_to_render:
                positions.append(particle.position)
                colors.append(particle.color)
                sizes.append(particle.size * 30)
            
            if positions:
                positions = np.array(positions)
                self.ax_ecosystem.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                         s=sizes, c=colors, alpha=0.7,
                                         edgecolors='white', linewidth=0.3)
            
            # Draw a few connections for visual effect
            connection_count = 0
            max_connections = 20
            
            for particle in particles_to_render[:30]:
                if connection_count >= max_connections:
                    break
                    
                for connection_id in list(particle.connections)[:2]:
                    if connection_count >= max_connections:
                        break
                        
                    connected = next((p for p in particles_to_render 
                                    if p.particle_id == connection_id), None)
                    if connected:
                        self.ax_ecosystem.plot([particle.position[0], connected.position[0]],
                                              [particle.position[1], connected.position[1]],
                                              [particle.position[2], connected.position[2]],
                                              color=INFO_PALETTE['information_flow'], 
                                              alpha=0.3, linewidth=0.5)
                        connection_count += 1
            
            # Set 3D limits
            self.ax_ecosystem.set_xlim(-60, 60)
            self.ax_ecosystem.set_ylim(-60, 60)
            self.ax_ecosystem.set_zlim(-50, 50)
            
            # Remove axis labels
            self.ax_ecosystem.set_xticks([])
            self.ax_ecosystem.set_yticks([])
            self.ax_ecosystem.set_zticks([])
        except Exception as e:
            print(f"3D ecosystem rendering error: {e}")
    
    def _render_information_network(self):
        """Render information flow network topology"""
        try:
            self.ax_network.set_title('Info Network', 
                                     color=INFO_PALETTE['consciousness'], fontsize=10)
            
            if len(self.ecosystem.particles) > 0:
                # Sample a few particles
                sample_size = min(10, len(self.ecosystem.particles))
                sample_particles = random.sample(self.ecosystem.particles, sample_size)
                
                # Simple circular layout
                for i, particle in enumerate(sample_particles):
                    angle = 2 * np.pi * i / sample_size
                    x = 0.8 * np.cos(angle)
                    y = 0.8 * np.sin(angle)
                    
                    self.ax_network.scatter(x, y, s=particle.size * 50, c=particle.color,
                                           alpha=0.8, edgecolors='white', linewidth=0.5)
            
            self.ax_network.set_xlim(-1.2, 1.2)
            self.ax_network.set_ylim(-1.2, 1.2)
            self.ax_network.set_aspect('equal')
            self.ax_network.set_xticks([])
            self.ax_network.set_yticks([])
        except Exception as e:
            print(f"Network rendering error: {e}")
    
    def _render_species_diversity(self):
        """Render species diversity metrics"""
        try:
            self.ax_diversity.set_title('Species Diversity', 
                                       color=INFO_PALETTE['consciousness'], fontsize=10)
            
            if len(self.diversity_history) > 1:
                time_axis = list(range(len(self.diversity_history)))
                diversity_values = list(self.diversity_history)
                
                self.ax_diversity.fill_between(time_axis, 0, diversity_values,
                                              color=INFO_PALETTE['complexity_growth'], alpha=0.3)
                self.ax_diversity.plot(time_axis, diversity_values,
                                      color=INFO_PALETTE['complexity_growth'], linewidth=1.5)
            
            self.ax_diversity.set_ylabel('Diversity', color=INFO_PALETTE['consciousness'], fontsize=8)
            self.ax_diversity.set_ylim(0, 3)
        except Exception as e:
            print(f"Diversity rendering error: {e}")
    
    def _render_population_dynamics(self):
        """Render population dynamics over time"""
        try:
            self.ax_population.set_title('Population', 
                                        color=INFO_PALETTE['consciousness'], fontsize=10)
            
            if len(self.population_history) > 1:
                time_axis = list(range(len(self.population_history)))
                population_values = list(self.population_history)
                
                self.ax_population.fill_between(time_axis, 0, population_values,
                                               color=INFO_PALETTE['swarm_intelligence'], alpha=0.3)
                self.ax_population.plot(time_axis, population_values,
                                       color=INFO_PALETTE['swarm_intelligence'], linewidth=1.5)
                
                # Carrying capacity line
                self.ax_population.axhline(y=self.ecosystem.carrying_capacity,
                                          color=INFO_PALETTE['selection_pressure'],
                                          linestyle='--', alpha=0.5)
            
            self.ax_population.set_ylabel('Count', color=INFO_PALETTE['consciousness'], fontsize=8)
            self.ax_population.set_ylim(0, self.ecosystem.carrying_capacity * 1.2)
        except Exception as e:
            print(f"Population rendering error: {e}")
    
    def _render_information_metabolism(self):
        """Render information metabolism rates"""
        try:
            self.ax_metabolism.set_title('Info Metabolism', 
                                        color=INFO_PALETTE['consciousness'], fontsize=10)
            
            if self.ecosystem.particles:
                # Get top 5 information types by count
                type_counts = Counter(p.info_type for p in self.ecosystem.particles)
                top_types = [t for t, _ in type_counts.most_common(5)]
                
                if top_types:
                    # Calculate average metabolism for each type
                    metabolism_by_type = {}
                    for info_type in top_types:
                        particles_of_type = [p for p in self.ecosystem.particles if p.info_type == info_type]
                        if particles_of_type:
                            avg_metabolism = np.mean([p.energy * p.complexity * 0.1 for p in particles_of_type])
                            metabolism_by_type[info_type] = avg_metabolism
                    
                    # Plot bars
                    types = list(metabolism_by_type.keys())
                    values = list(metabolism_by_type.values())
                    colors = [INFO_PALETTE.get(t, '#FFFFFF') for t in types]
                    
                    self.ax_metabolism.bar(range(len(types)), values,
                                         color=colors, alpha=0.7,
                                         edgecolor=INFO_PALETTE['consciousness'], linewidth=0.5)
                    
                    # Simplified labels
                    labels = [t.split('_')[0][:4] for t in types]  # First 4 chars of first word
                    self.ax_metabolism.set_xticks(range(len(types)))
                    self.ax_metabolism.set_xticklabels(labels, fontsize=7)
            
            self.ax_metabolism.set_ylabel('Rate', color=INFO_PALETTE['consciousness'], fontsize=8)
        except Exception as e:
            print(f"Metabolism rendering error: {e}")
    
    def _render_fitness_landscape(self):
        """Render fitness landscape topology"""
        try:
            self.ax_fitness.set_title('Fitness Landscape', 
                                     color=INFO_PALETTE['consciousness'], fontsize=10)
            
            if len(self.ecosystem.particles) > 0:
                # Simple fitness visualization
                fitness_values = [p.fitness for p in self.ecosystem.particles[:50]]
                
                if fitness_values:
                    # Create a simple heatmap-like visualization
                    grid_size = 10
                    fitness_grid = np.random.rand(grid_size, grid_size) * np.mean(fitness_values)
                    
                    im = self.ax_fitness.imshow(fitness_grid, cmap='viridis', alpha=0.7)
                    
                    # Add a few peak markers
                    for _ in range(3):
                        peak_x = random.randint(1, grid_size-2)
                        peak_y = random.randint(1, grid_size-2)
                        self.ax_fitness.scatter(peak_x, peak_y, s=50,
                                               c=INFO_PALETTE['emergence_signal'],
                                               alpha=0.8, marker='*')
            
            self.ax_fitness.set_xticks([])
            self.ax_fitness.set_yticks([])
        except Exception as e:
            print(f"Fitness rendering error: {e}")
    
    def _render_quantum_effects(self):
        """Render quantum information effects"""
        try:
            self.ax_quantum.set_title('Quantum Effects', 
                                     color=INFO_PALETTE['consciousness'], fontsize=10, pad=20)
            
            # Simple quantum visualization
            n_points = 8
            theta = np.linspace(0, 2*np.pi, n_points)
            
            for i in range(n_points):
                radius = 0.5 + 0.3 * np.sin(self.time + i)
                self.ax_quantum.scatter(theta[i], radius, s=30,
                                       c=INFO_PALETTE['quantum_qubits'], alpha=0.6)
            
            # Add some connecting lines
            for i in range(n_points):
                next_i = (i + 1) % n_points
                self.ax_quantum.plot([theta[i], theta[next_i]], 
                                    [0.5 + 0.3 * np.sin(self.time + i),
                                     0.5 + 0.3 * np.sin(self.time + next_i)],
                                    color=INFO_PALETTE['entanglement'], alpha=0.3, linewidth=0.5)
            
            self.ax_quantum.set_ylim(0, 1)
            self.ax_quantum.set_rticks([])
            self.ax_quantum.set_thetagrids([])
        except Exception as e:
            print(f"Quantum rendering error: {e}")
    
    def animate(self):
        """Start the quantum information ecology animation"""
        def update(frame):
            try:
                self.update_visualization(frame)
            except Exception as e:
                print(f"Animation error at frame {frame}: {e}")
            return []
        
        self.anim = animation.FuncAnimation(
            self.fig, update,
            frames=2000,  # Reduced from 5000
            interval=100,  # Increased from 80ms for better performance
            blit=False,
            repeat=True
        )
        
        plt.show()


def run_quantum_info_ecology():
    """Launch the Quantum Information Ecology"""
    print("QUANTUM INFORMATION ECOLOGY 2025")
    print("Where Information Lives, Evolves, and Forms Consciousness")
    print()
    print("Revolutionary Features:")
    print("• Information particles that live, reproduce, and evolve")
    print("• Information food webs and ecosystem dynamics")
    print("• Emergent intelligence from information interactions")
    print("• Quantum information effects and entanglement")
    print("• Information metabolism and energy flows")
    print("• Digital evolution and artificial life principles")
    print("• 40+ unique colors representing information types")
    print("• Real-time emergence of complex information structures")
    print()
    print("Launching optimized version with error handling...")
    
    try:
        ecology = QuantumInfoEcologyVisualizer()
        ecology.animate()
    except Exception as e:
        print(f"Error launching information ecology: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install numpy matplotlib scipy")


if __name__ == "__main__":
    run_quantum_info_ecology()