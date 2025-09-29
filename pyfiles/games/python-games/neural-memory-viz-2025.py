"""
NEURAL MEMORY CRYSTALLIZATION VISUALIZER 2025
Synaptic Garden Architecture with Memory Formation Dynamics
Novel approach: memories as growing crystals in neural substrate
Featuring: Dendrite trees, synaptic fireflies, and thought bubbles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon, Wedge, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.path import Path
import matplotlib.patches as mpatches
from scipy import interpolate, signal, ndimage, spatial
from scipy.special import jv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator, Protocol, TypeVar, Union
from collections import deque, defaultdict, namedtuple
from enum import Enum, auto
import colorsys
import math
import random
import time

# Novel ocean-sunset palette with deeper psychological tones
NEURAL_PALETTE = {
    'deep_cognition': '#4A5D7A',     # Deep blue-gray
    'memory_amber': '#B8956A',       # Warm amber
    'synapse_violet': '#8B6AA7',     # Purple synapse
    'dendrite_teal': '#5A8B7D',      # Teal dendrite
    'thought_coral': '#C67B6F',      # Coral thought
    'dream_indigo': '#6B5B8B',       # Dream indigo
    'focus_gold': '#A89B5B',         # Focus gold
    'recall_sage': '#6B8B6A',        # Recall sage
    'neuron_blush': '#B57B8B',       # Neuron blush
    'axon_slate': '#6A7B8B',         # Axon slate
    'impulse_peach': '#D4A574',      # Impulse peach
    'plasticity_plum': '#9B6B8B'     # Plasticity plum
}

@dataclass
class MemoryCrystal:
    """Memory represented as a growing crystal structure"""
    
    core_position: Tuple[float, float]
    formation_time: float
    strength: float = 0.0
    crystal_type: str = 'declarative'  # declarative, procedural, emotional
    vertices: List[Tuple[float, float]] = field(default_factory=list)
    color: str = '#8B6AA7'
    growth_rate: float = 0.1
    max_size: float = 50.0
    decay_rate: float = 0.001
    connections: List[str] = field(default_factory=list)
    resonance_frequency: float = field(default_factory=lambda: random.uniform(0.1, 2.0))
    
    def grow(self, nutrients: float, time: float):
        """Grow crystal based on available neural nutrients"""
        if self.strength < self.max_size:
            self.strength += nutrients * self.growth_rate * (1 - self.strength/self.max_size)
        
        # Generate crystal vertices using polar coordinates
        n_vertices = int(6 + self.strength / 10)
        self.vertices = []
        
        for i in range(n_vertices):
            angle = 2 * np.pi * i / n_vertices + time * 0.01
            
            # Add fractal-like irregularity
            r = self.strength * (1 + 0.2 * np.sin(5 * angle + time * self.resonance_frequency))
            
            x = self.core_position[0] + r * np.cos(angle)
            y = self.core_position[1] + r * np.sin(angle)
            
            self.vertices.append((x, y))
    
    def decay(self):
        """Natural memory decay over time"""
        self.strength *= (1 - self.decay_rate)
        if self.strength < 0.1:
            self.strength = 0
    
    def resonate_with(self, other: 'MemoryCrystal') -> float:
        """Calculate resonance between memories"""
        freq_diff = abs(self.resonance_frequency - other.resonance_frequency)
        distance = np.sqrt((self.core_position[0] - other.core_position[0])**2 + 
                          (self.core_position[1] - other.core_position[1])**2)
        
        resonance = np.exp(-freq_diff) * np.exp(-distance/100)
        return resonance


class SynapticFirefly:
    """Represents neural impulses as fireflies traveling along dendrites"""
    
    def __init__(self, start_pos: Tuple[float, float], target_pos: Tuple[float, float], 
                 neurotransmitter: str = 'dopamine'):
        self.position = list(start_pos)
        self.target = target_pos
        self.trail = deque(maxlen=20)
        self.neurotransmitter = neurotransmitter
        self.brightness = 1.0
        self.speed = random.uniform(2, 5)
        self.phase = random.uniform(0, 2*np.pi)
        self.alive = True
        
        # Neurotransmitter colors
        self.colors = {
            'dopamine': '#FFB366',     # Orange - reward
            'serotonin': '#66B3FF',    # Blue - mood
            'gaba': '#B366FF',         # Purple - inhibitory
            'glutamate': '#66FFB3',    # Green - excitatory
            'acetylcholine': '#FF66B3' # Pink - attention
        }
        
        self.color = self.colors.get(neurotransmitter, '#FFFFFF')
    
    def update(self, dt: float):
        """Update firefly position with organic movement"""
        if not self.alive:
            return
        
        # Calculate direction to target
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 5:
            self.alive = False
            return
        
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Add wandering behavior
        wander_angle = np.sin(self.phase) * 0.5
        cos_w = np.cos(wander_angle)
        sin_w = np.sin(wander_angle)
        
        new_dx = dx * cos_w - dy * sin_w
        new_dy = dx * sin_w + dy * cos_w
        
        # Update position
        self.position[0] += new_dx * self.speed
        self.position[1] += new_dy * self.speed
        
        # Update trail
        self.trail.append(tuple(self.position))
        
        # Update brightness (pulsing effect)
        self.brightness = 0.5 + 0.5 * np.sin(self.phase * 2)
        self.phase += 0.1
    
    def burst(self) -> List['SynapticFirefly']:
        """Create burst of new fireflies when reaching synapse"""
        if not self.alive:
            burst = []
            for _ in range(random.randint(2, 5)):
                angle = random.uniform(0, 2*np.pi)
                new_target = (
                    self.target[0] + 50 * np.cos(angle),
                    self.target[1] + 50 * np.sin(angle)
                )
                burst.append(SynapticFirefly(self.target, new_target, self.neurotransmitter))
            return burst
        return []


class DendriteTree:
    """Fractal dendrite structure that grows and branches"""
    
    def __init__(self, root: Tuple[float, float], angle: float = -np.pi/2):
        self.root = root
        self.branches = []
        self.angle = angle
        self.max_depth = 5
        self.branch_angle = np.pi/6
        self.length_ratio = 0.75
        self.thickness = 5
        self.growth_stage = 0
        
        # Initialize with seed branch
        self._grow_branch(root, angle, 40, 0)
    
    def _grow_branch(self, start: Tuple[float, float], angle: float, 
                     length: float, depth: int):
        """Recursively grow dendrite branches"""
        if depth > self.max_depth or length < 5:
            return
        
        # Calculate end point
        end = (
            start[0] + length * np.cos(angle),
            start[1] + length * np.sin(angle)
        )
        
        # Store branch
        self.branches.append({
            'start': start,
            'end': end,
            'depth': depth,
            'thickness': self.thickness * (0.7 ** depth),
            'active': random.random() > 0.3
        })
        
        # Grow child branches with probability
        if random.random() > 0.2:
            # Left branch
            self._grow_branch(end, angle - self.branch_angle, 
                            length * self.length_ratio, depth + 1)
        
        if random.random() > 0.2:
            # Right branch
            self._grow_branch(end, angle + self.branch_angle, 
                            length * self.length_ratio, depth + 1)
    
    def stimulate(self, intensity: float):
        """Stimulate dendrite with neural activity"""
        for branch in self.branches:
            if branch['active']:
                branch['thickness'] = min(10, branch['thickness'] * (1 + intensity * 0.1))
    
    def get_endpoints(self) -> List[Tuple[float, float]]:
        """Get all branch endpoints for synapse connections"""
        endpoints = []
        for branch in self.branches:
            if branch['depth'] >= self.max_depth - 1:
                endpoints.append(branch['end'])
        return endpoints


class ThoughtBubble:
    """Represents conscious thoughts as expanding bubbles"""
    
    def __init__(self, origin: Tuple[float, float], thought_type: str = 'random'):
        self.origin = origin
        self.radius = 5
        self.max_radius = random.uniform(30, 60)
        self.opacity = 1.0
        self.thought_type = thought_type
        self.lifetime = 0
        self.max_lifetime = random.uniform(50, 100)
        self.wobble_phase = random.uniform(0, 2*np.pi)
        
        # Thought patterns
        self.patterns = {
            'memory': self._generate_memory_pattern,
            'emotion': self._generate_emotion_pattern,
            'logic': self._generate_logic_pattern,
            'creative': self._generate_creative_pattern
        }
        
        self.inner_pattern = []
        self._generate_pattern()
    
    def _generate_pattern(self):
        """Generate internal pattern based on thought type"""
        if self.thought_type in self.patterns:
            self.inner_pattern = self.patterns[self.thought_type]()
    
    def _generate_memory_pattern(self) -> List[Tuple[float, float]]:
        """Spiral pattern for memories"""
        points = []
        for i in range(20):
            t = i / 20 * 2 * np.pi
            r = self.radius * 0.5 * (1 - i/20)
            x = self.origin[0] + r * np.cos(t * 3)
            y = self.origin[1] + r * np.sin(t * 3)
            points.append((x, y))
        return points
    
    def _generate_emotion_pattern(self) -> List[Tuple[float, float]]:
        """Heart-like pattern for emotions"""
        points = []
        for i in range(20):
            t = i / 20 * 2 * np.pi
            x = self.origin[0] + self.radius * 0.3 * 16 * np.sin(t)**3 / 16
            y = self.origin[1] - self.radius * 0.3 * (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) / 16
            points.append((x, y))
        return points
    
    def _generate_logic_pattern(self) -> List[Tuple[float, float]]:
        """Grid pattern for logical thoughts"""
        points = []
        grid_size = 3
        for i in range(grid_size):
            for j in range(grid_size):
                x = self.origin[0] + (i - grid_size/2) * self.radius * 0.3
                y = self.origin[1] + (j - grid_size/2) * self.radius * 0.3
                points.append((x, y))
        return points
    
    def _generate_creative_pattern(self) -> List[Tuple[float, float]]:
        """Random star burst for creativity"""
        points = []
        for i in range(8):
            angle = i * np.pi / 4
            length = self.radius * 0.4 * random.uniform(0.5, 1.0)
            x = self.origin[0] + length * np.cos(angle)
            y = self.origin[1] + length * np.sin(angle)
            points.append((x, y))
        return points
    
    def update(self, dt: float):
        """Update bubble expansion and fade"""
        self.lifetime += dt
        
        # Expand
        if self.radius < self.max_radius:
            self.radius += 0.5
        
        # Fade
        self.opacity = max(0, 1 - self.lifetime / self.max_lifetime)
        
        # Wobble
        self.wobble_phase += 0.1
        wobble = np.sin(self.wobble_phase) * 2
        self.origin = (self.origin[0] + wobble * 0.1, self.origin[1])
        
        # Update pattern
        self._generate_pattern()
    
    def is_alive(self) -> bool:
        return self.opacity > 0.01


class ConsciousnessField:
    """Background field representing overall consciousness state"""
    
    def __init__(self, width: int = 100, height: int = 80):
        self.width = width
        self.height = height
        self.field = np.zeros((height, width))
        self.phase = 0
        self.wave_sources = []
        
        # Brainwave frequencies (Hz)
        self.frequencies = {
            'delta': 2,      # Deep sleep
            'theta': 6,      # Drowsiness
            'alpha': 10,     # Relaxed
            'beta': 20,      # Alert
            'gamma': 40      # Concentration
        }
        
        self.current_state = 'alpha'
    
    def add_wave_source(self, x: int, y: int, amplitude: float):
        """Add a new consciousness wave source"""
        self.wave_sources.append({'x': x, 'y': y, 'amplitude': amplitude, 'phase': random.uniform(0, 2*np.pi)})
    
    def update(self, dt: float):
        """Update consciousness field with wave interference"""
        self.phase += dt * self.frequencies[self.current_state]
        
        # Clear field
        self.field.fill(0)
        
        # Generate wave interference pattern
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        
        for source in self.wave_sources:
            # Distance from source
            r = np.sqrt((xx - source['x'])**2 + (yy - source['y'])**2)
            
            # Wave equation with damping
            wave = source['amplitude'] * np.sin(2 * np.pi * r / 10 - self.phase + source['phase']) * np.exp(-r / 30)
            self.field += wave
            
            # Update source phase
            source['phase'] += dt * 0.5
        
        # Normalize field
        if np.max(np.abs(self.field)) > 0:
            self.field = self.field / np.max(np.abs(self.field))
    
    def set_state(self, state: str):
        """Change consciousness state"""
        if state in self.frequencies:
            self.current_state = state


class NeuralMemoryVisualizer:
    """Main visualization engine for neural memory system"""
    
    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        # Setup figure
        self.fig = plt.figure(figsize=figsize, facecolor='#0a0a0f')
        self.fig.suptitle('Neural Memory Crystallization System 2025', 
                         fontsize=18, color='#E0E0F0', fontweight='bold')
        
        # Create custom layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.25, wspace=0.25)
        
        self.ax_main = self.fig.add_subplot(gs[0:2, 0:3])      # Main neural garden
        self.ax_memory = self.fig.add_subplot(gs[0, 3])        # Memory strength
        self.ax_waves = self.fig.add_subplot(gs[1, 3])         # Brainwaves
        self.ax_pattern = self.fig.add_subplot(gs[2, 0])       # Pattern recognition
        self.ax_emotion = self.fig.add_subplot(gs[2, 1])       # Emotional state
        self.ax_learning = self.fig.add_subplot(gs[2, 2])      # Learning curve
        self.ax_consciousness = self.fig.add_subplot(gs[2, 3]) # Consciousness field
        
        # Style axes
        for ax in [self.ax_main, self.ax_memory, self.ax_waves, self.ax_pattern,
                   self.ax_emotion, self.ax_learning, self.ax_consciousness]:
            ax.set_facecolor('#15151f')
            ax.tick_params(colors='#808090', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#303040')
                spine.set_linewidth(0.5)
        
        # Initialize components
        self.memories = []
        self.fireflies = []
        self.dendrites = []
        self.thoughts = []
        self.consciousness = ConsciousnessField()
        
        # Data tracking
        self.memory_strength_history = deque(maxlen=100)
        self.emotion_history = deque(maxlen=100)
        self.learning_rate = deque(maxlen=100)
        
        # Animation state
        self.frame = 0
        self.time = 0
        
        # Initialize neural garden
        self._initialize_neural_garden()
        
        # Prebuilt memory patterns
        self.memory_patterns = self._generate_memory_patterns()
    
    def _initialize_neural_garden(self):
        """Setup initial neural network structure"""
        # Create dendrite trees at various positions
        positions = [
            (100, 300), (200, 400), (300, 350),
            (400, 300), (500, 400), (600, 350)
        ]
        
        for pos in positions:
            angle = random.uniform(-np.pi*0.7, -np.pi*0.3)
            self.dendrites.append(DendriteTree(pos, angle))
        
        # Add initial memories
        memory_positions = [
            (150, 250, 'declarative'),
            (350, 280, 'procedural'),
            (550, 320, 'emotional')
        ]
        
        for x, y, mtype in memory_positions:
            memory = MemoryCrystal(
                core_position=(x, y),
                formation_time=self.time,
                crystal_type=mtype,
                color=random.choice(list(NEURAL_PALETTE.values()))
            )
            self.memories.append(memory)
        
        # Initialize consciousness field
        for _ in range(5):
            self.consciousness.add_wave_source(
                random.randint(20, 80),
                random.randint(20, 60),
                random.uniform(0.5, 1.0)
            )
    
    def _generate_memory_patterns(self) -> Dict[str, np.ndarray]:
        """Generate prebuilt memory formation patterns"""
        patterns = {}
        
        # Short-term memory formation (rapid spike and decay)
        t = np.linspace(0, 10, 100)
        patterns['short_term'] = 10 * np.exp(-t/2) * (1 + 0.3*np.sin(5*t))
        
        # Long-term memory consolidation (slow build with plateau)
        patterns['long_term'] = 20 * (1 - np.exp(-t/3)) * (1 + 0.1*np.sin(2*t))
        
        # Working memory (oscillating maintenance)
        patterns['working'] = 5 + 3*np.sin(2*t) + 2*np.sin(3*t)
        
        # Episodic memory (burst pattern)
        patterns['episodic'] = np.zeros_like(t)
        for peak in [2, 5, 8]:
            patterns['episodic'] += 15 * np.exp(-((t-peak)/0.5)**2)
        
        # Semantic memory (gradual accumulation)
        patterns['semantic'] = np.cumsum(0.1 + 0.5*np.random.random(len(t)))
        
        return patterns
    
    def update_frame(self, frame_num: int):
        """Update all visualization components"""
        self.frame = frame_num
        self.time = frame_num * 0.1
        
        # Clear all axes
        for ax in [self.ax_main, self.ax_memory, self.ax_waves, self.ax_pattern,
                   self.ax_emotion, self.ax_learning, self.ax_consciousness]:
            ax.clear()
            ax.set_facecolor('#15151f')
        
        # Update components
        self._update_memories()
        self._update_fireflies()
        self._update_thoughts()
        self.consciousness.update(0.1)
        
        # Draw everything
        self._draw_neural_garden()
        self._draw_memory_strength()
        self._draw_brainwaves()
        self._draw_pattern_recognition()
        self._draw_emotional_state()
        self._draw_learning_curve()
        self._draw_consciousness_field()
        
        # Spawn new elements periodically
        if frame_num % 20 == 0:
            self._spawn_firefly()
        
        if frame_num % 50 == 0:
            self._spawn_thought()
        
        if frame_num % 100 == 0:
            self._trigger_memory_formation()
    
    def _update_memories(self):
        """Update all memory crystals"""
        total_strength = 0
        
        for memory in self.memories:
            # Memories grow when stimulated by nearby fireflies
            nearby_nutrients = 0
            for firefly in self.fireflies:
                dist = np.sqrt((firefly.position[0] - memory.core_position[0])**2 + 
                             (firefly.position[1] - memory.core_position[1])**2)
                if dist < 50:
                    nearby_nutrients += firefly.brightness / (1 + dist/10)
            
            memory.grow(nearby_nutrients, self.time)
            memory.decay()
            total_strength += memory.strength
        
        # Check for memory resonance
        for i, m1 in enumerate(self.memories):
            for m2 in self.memories[i+1:]:
                resonance = m1.resonate_with(m2)
                if resonance > 0.7:
                    # Strong resonance creates new connection
                    if m2 not in m1.connections:
                        m1.connections.append(m2)
                        m2.connections.append(m1)
        
        self.memory_strength_history.append(total_strength)
    
    def _update_fireflies(self):
        """Update synaptic fireflies"""
        new_fireflies = []
        
        for firefly in self.fireflies[:]:
            firefly.update(0.1)
            
            if not firefly.alive:
                # Firefly reached target, might burst
                burst = firefly.burst()
                new_fireflies.extend(burst)
                self.fireflies.remove(firefly)
        
        self.fireflies.extend(new_fireflies)
    
    def _update_thoughts(self):
        """Update thought bubbles"""
        for thought in self.thoughts[:]:
            thought.update(0.1)
            if not thought.is_alive():
                self.thoughts.remove(thought)
    
    def _draw_neural_garden(self):
        """Draw main neural network visualization"""
        self.ax_main.set_xlim(0, 700)
        self.ax_main.set_ylim(0, 500)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('Synaptic Garden', color='#E0E0F0', fontsize=10)
        
        # Draw dendrites
        for dendrite in self.dendrites:
            for branch in dendrite.branches:
                color = NEURAL_PALETTE['dendrite_teal'] if branch['active'] else '#404050'
                alpha = 0.8 if branch['active'] else 0.3
                
                self.ax_main.plot(
                    [branch['start'][0], branch['end'][0]],
                    [branch['start'][1], branch['end'][1]],
                    color=color,
                    linewidth=branch['thickness'],
                    alpha=alpha,
                    solid_capstyle='round'
                )
        
        # Draw memory crystals
        for memory in self.memories:
            if memory.vertices:
                crystal = Polygon(
                    memory.vertices,
                    facecolor=memory.color,
                    edgecolor='white',
                    alpha=0.6 + 0.3 * np.sin(self.time * memory.resonance_frequency),
                    linewidth=1
                )
                self.ax_main.add_patch(crystal)
                
                # Draw connections
                for connected in memory.connections:
                    if isinstance(connected, MemoryCrystal):
                        self.ax_main.plot(
                            [memory.core_position[0], connected.core_position[0]],
                            [memory.core_position[1], connected.core_position[1]],
                            color='white',
                            alpha=0.2,
                            linewidth=0.5,
                            linestyle='--'
                        )
        
        # Draw fireflies
        for firefly in self.fireflies:
            # Draw trail
            if len(firefly.trail) > 1:
                trail_points = list(firefly.trail)
                for i in range(len(trail_points) - 1):
                    alpha = (i / len(trail_points)) * 0.5
                    self.ax_main.plot(
                        [trail_points[i][0], trail_points[i+1][0]],
                        [trail_points[i][1], trail_points[i+1][1]],
                        color=firefly.color,
                        alpha=alpha,
                        linewidth=1
                    )
            
            # Draw firefly
            circle = Circle(
                firefly.position,
                radius=3 * firefly.brightness,
                facecolor=firefly.color,
                edgecolor='white',
                alpha=firefly.brightness,
                linewidth=0.5
            )
            self.ax_main.add_patch(circle)
        
        # Draw thought bubbles
        for thought in self.thoughts:
            # Draw expanding bubble
            bubble = Circle(
                thought.origin,
                radius=thought.radius,
                facecolor='none',
                edgecolor=NEURAL_PALETTE['thought_coral'],
                alpha=thought.opacity * 0.5,
                linewidth=2,
                linestyle='--'
            )
            self.ax_main.add_patch(bubble)
            
            # Draw internal pattern
            if thought.inner_pattern:
                for point in thought.inner_pattern:
                    self.ax_main.plot(
                        point[0], point[1],
                        'o',
                        color=NEURAL_PALETTE['thought_coral'],
                        markersize=2,
                        alpha=thought.opacity
                    )
    
    def _draw_memory_strength(self):
        """Draw memory strength over time"""
        self.ax_memory.set_title('Memory Strength', color='#E0E0F0', fontsize=9)
        self.ax_memory.set_xlabel('Time', color='#808090', fontsize=8)
        self.ax_memory.set_ylabel('Strength', color='#808090', fontsize=8)
        
        if len(self.memory_strength_history) > 1:
            x = range(len(self.memory_strength_history))
            y = list(self.memory_strength_history)
            
            self.ax_memory.fill_between(x, 0, y, 
                                       color=NEURAL_PALETTE['memory_amber'],
                                       alpha=0.3)
            self.ax_memory.plot(x, y,
                               color=NEURAL_PALETTE['memory_amber'],
                               linewidth=2)
            
            # Add moving average
            if len(y) > 10:
                window = np.ones(10) / 10
                ma = np.convolve(y, window, mode='valid')
                self.ax_memory.plot(range(5, len(y)-4), ma,
                                  color='white',
                                  linewidth=1,
                                  alpha=0.5,
                                  linestyle='--')
    
    def _draw_brainwaves(self):
        """Draw brainwave patterns"""
        self.ax_waves.set_title('Brainwave Activity', color='#E0E0F0', fontsize=9)
        self.ax_waves.set_xlabel('Frequency (Hz)', color='#808090', fontsize=8)
        self.ax_waves.set_ylabel('Power', color='#808090', fontsize=8)
        
        # Simulate brainwave spectrum
        freqs = np.linspace(0.5, 50, 100)
        
        # Generate power spectrum based on consciousness state
        power = np.zeros_like(freqs)
        
        for wave_type, freq in self.consciousness.frequencies.items():
            # Add peak at characteristic frequency
            sigma = 2 if wave_type == self.consciousness.current_state else 0.5
            amplitude = 1.0 if wave_type == self.consciousness.current_state else 0.3
            power += amplitude * np.exp(-((freqs - freq) / sigma) ** 2)
        
        # Add noise
        power += 0.1 * np.random.random(len(freqs))
        
        self.ax_waves.fill_between(freqs, 0, power,
                                  color=NEURAL_PALETTE['synapse_violet'],
                                  alpha=0.3)
        self.ax_waves.plot(freqs, power,
                         color=NEURAL_PALETTE['synapse_violet'],
                         linewidth=2)
        
        # Mark frequency bands
        for wave_type, freq in self.consciousness.frequencies.items():
            self.ax_waves.axvline(freq, color='white', alpha=0.2, linestyle=':')
            self.ax_waves.text(freq, max(power)*0.9, wave_type[0].upper(),
                             color='white', fontsize=7, alpha=0.5)
    
    def _draw_pattern_recognition(self):
        """Draw pattern recognition matrix"""
        self.ax_pattern.set_title('Pattern Recognition', color='#E0E0F0', fontsize=9)
        
        # Create pattern matrix
        size = 20
        pattern = np.zeros((size, size))
        
        # Add recognized patterns based on active memories
        for memory in self.memories:
            if memory.strength > 10:
                x = int(memory.core_position[0] / 700 * size)
                y = int(memory.core_position[1] / 500 * size)
                
                # Create pattern around memory location
                for i in range(max(0, x-2), min(size, x+3)):
                    for j in range(max(0, y-2), min(size, y+3)):
                        pattern[j, i] += memory.strength / 50
        
        # Display pattern
        im = self.ax_pattern.imshow(pattern,
                                   cmap='plasma',
                                   interpolation='bicubic',
                                   vmin=0, vmax=1)
        self.ax_pattern.set_xticks([])
        self.ax_pattern.set_yticks([])
    
    def _draw_emotional_state(self):
        """Draw emotional state radar chart"""
        self.ax_emotion.set_title('Emotional State', color='#E0E0F0', fontsize=9)
        
        # Emotional dimensions
        emotions = ['Joy', 'Fear', 'Anger', 'Sadness', 'Surprise', 'Trust']
        
        # Generate emotional values based on neural activity
        values = []
        for i, emotion in enumerate(emotions):
            base = 0.3
            
            # Modulate based on firefly activity
            if emotion == 'Joy' and len(self.fireflies) > 5:
                base += 0.4
            elif emotion == 'Fear' and any(m.crystal_type == 'emotional' for m in self.memories):
                base += 0.3
            elif emotion == 'Trust' and len(self.thoughts) > 2:
                base += 0.3
            
            # Add variation
            value = base + 0.2 * np.sin(self.time * (i + 1) * 0.5)
            values.append(min(1, max(0, value)))
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
        values = values + values[:1]  # Complete the circle
        angles += angles[:1]
        
        self.ax_emotion.plot(angles, values,
                           color=NEURAL_PALETTE['thought_coral'],
                           linewidth=2)
        self.ax_emotion.fill(angles, values,
                           color=NEURAL_PALETTE['thought_coral'],
                           alpha=0.25)
        
        # Add emotion labels
        self.ax_emotion.set_xticks(angles[:-1])
        self.ax_emotion.set_xticklabels(emotions, size=7, color='#808090')
        self.ax_emotion.set_ylim(0, 1)
        self.ax_emotion.set_yticks([0.25, 0.5, 0.75])
        self.ax_emotion.set_yticklabels(['0.25', '0.5', '0.75'], size=7, color='#808090')
        self.ax_emotion.grid(True, alpha=0.2)
        
        self.emotion_history.append(values[0])  # Track joy
    
    def _draw_learning_curve(self):
        """Draw learning progress curve"""
        self.ax_learning.set_title('Learning Curve', color='#E0E0F0', fontsize=9)
        self.ax_learning.set_xlabel('Iterations', color='#808090', fontsize=8)
        self.ax_learning.set_ylabel('Performance', color='#808090', fontsize=8)
        
        # Simulate learning curve
        x = np.linspace(0, 100, 100)
        
        # Different learning patterns
        pattern_idx = (self.frame // 100) % 5
        
        if pattern_idx == 0:
            # Smooth learning
            y = 1 - np.exp(-x/20)
            label = 'Smooth'
        elif pattern_idx == 1:
            # Plateau learning
            y = np.minimum(1, x/30) * (1 - 0.2*np.exp(-(x-50)**2/100))
            label = 'Plateau'
        elif pattern_idx == 2:
            # Breakthrough learning
            y = 1 / (1 + np.exp(-(x-50)/5))
            label = 'Breakthrough'
        elif pattern_idx == 3:
            # Oscillating learning
            y = (1 - np.exp(-x/25)) * (1 + 0.2*np.sin(x/5))
            label = 'Oscillating'
        else:
            # Forgetting curve
            y = np.exp(-x/30) + 0.2
            label = 'Forgetting'
        
        # Add noise
        y += 0.05 * np.random.randn(len(y))
        y = np.clip(y, 0, 1)
        
        self.ax_learning.plot(x, y,
                            color=NEURAL_PALETTE['focus_gold'],
                            linewidth=2,
                            label=label)
        self.ax_learning.fill_between(x, 0, y,
                                     color=NEURAL_PALETTE['focus_gold'],
                                     alpha=0.3)
        
        self.ax_learning.legend(loc='lower right', fontsize=7,
                              framealpha=0.5, facecolor='#15151f')
        self.ax_learning.set_ylim(0, 1.1)
        self.ax_learning.grid(True, alpha=0.2)
        
        self.learning_rate.append(y[-1])
    
    def _draw_consciousness_field(self):
        """Draw consciousness field"""
        self.ax_consciousness.set_title(f'Consciousness Field ({self.consciousness.current_state})', 
                                       color='#E0E0F0', fontsize=9)
        
        # Display field
        field_display = self.consciousness.field
        
        im = self.ax_consciousness.imshow(field_display,
                                         cmap='twilight',
                                         interpolation='bicubic',
                                         vmin=-1, vmax=1,
                                         alpha=0.9)
        
        self.ax_consciousness.set_xticks([])
        self.ax_consciousness.set_yticks([])
        
        # Add state indicator
        states = list(self.consciousness.frequencies.keys())
        current_idx = states.index(self.consciousness.current_state)
        
        # Cycle through states periodically
        if self.frame % 200 == 0:
            next_idx = (current_idx + 1) % len(states)
            self.consciousness.set_state(states[next_idx])
    
    def _spawn_firefly(self):
        """Spawn new synaptic firefly"""
        if self.dendrites:
            # Pick random dendrite endpoint
            dendrite = random.choice(self.dendrites)
            endpoints = dendrite.get_endpoints()
            
            if endpoints:
                start = random.choice(endpoints)
                
                # Pick random target (memory or another dendrite)
                if self.memories and random.random() > 0.5:
                    target_memory = random.choice(self.memories)
                    target = target_memory.core_position
                else:
                    target_dendrite = random.choice(self.dendrites)
                    target_endpoints = target_dendrite.get_endpoints()
                    if target_endpoints:
                        target = random.choice(target_endpoints)
                    else:
                        return
                
                # Random neurotransmitter type
                neurotransmitter = random.choice(['dopamine', 'serotonin', 'gaba', 
                                                 'glutamate', 'acetylcholine'])
                
                firefly = SynapticFirefly(start, target, neurotransmitter)
                self.fireflies.append(firefly)
    
    def _spawn_thought(self):
        """Spawn new thought bubble"""
        # Random position near active area
        if self.memories:
            memory = random.choice(self.memories)
            x = memory.core_position[0] + random.uniform(-50, 50)
            y = memory.core_position[1] + random.uniform(-50, 50)
            
            thought_type = random.choice(['memory', 'emotion', 'logic', 'creative'])
            thought = ThoughtBubble((x, y), thought_type)
            self.thoughts.append(thought)
    
    def _trigger_memory_formation(self):
        """Trigger formation of new memory"""
        # Find area with high neural activity
        if len(self.fireflies) > 3:
            # Calculate center of firefly activity
            x_mean = np.mean([f.position[0] for f in self.fireflies])
            y_mean = np.mean([f.position[1] for f in self.fireflies])
            
            # Create new memory at activity center
            memory = MemoryCrystal(
                core_position=(x_mean, y_mean),
                formation_time=self.time,
                crystal_type=random.choice(['declarative', 'procedural', 'emotional']),
                color=random.choice(list(NEURAL_PALETTE.values()))
            )
            
            self.memories.append(memory)
    
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


def run_neural_visualization():
    """Main entry point"""
    print("Neural Memory Crystallization Visualizer 2025")
    print("Novel Features:")
    print("- Memory crystals that grow and resonate")
    print("- Synaptic fireflies carrying neurotransmitters")
    print("- Fractal dendrite trees")
    print("- Thought bubbles with pattern recognition")
    print("- Consciousness field with brainwave states")
    print("- Emotional state radar")
    print("- Learning curve dynamics")
    print("- 12 custom psychological color palette")
    
    visualizer = NeuralMemoryVisualizer()
    visualizer.animate()


if __name__ == "__main__":
    run_neural_visualization()