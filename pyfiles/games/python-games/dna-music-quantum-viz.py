"""
QUANTUM DNA MUSIC EVOLUTION SYSTEM 
2025
Genetic Symphony Architecture with DNA-to-Sound Synthesis
Cazzy Aporbo, Ms: DNA sequences as musical compositions evolving in quantum space
I tried to represent Chromosome dancers, gene harmonics, and mutation melodies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon, Ellipse, Arc, FancyBboxPatch
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.path import Path
import matplotlib.patches as mpatches
from scipy import interpolate, signal, fft, special
from scipy.spatial import Voronoi, voronoi_plot_2d, distance
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Protocol
from collections import deque, defaultdict, Counter
from enum import Enum, auto
import colorsys
import math
import random
import itertools
import hashlib

# Bioluminescent ocean palette
GENETIC_PALETTE = {
    'adenine_azure': '#4A7B8C',      # DNA base A
    'thymine_teal': '#5C8B7B',       # DNA base T
    'guanine_gold': '#8C7B4A',       # DNA base G
    'cytosine_coral': '#8C5C6B',     # DNA base C
    'helix_violet': '#7B6B8C',       # Double helix
    'mutation_magenta': '#8C4A7B',   # Mutations
    'evolution_emerald': '#4A8C6B',  # Evolution
    'protein_pearl': '#8B8C8A',      # Proteins
    'enzyme_electric': '#6B7B8C',    # Enzymes
    'chromosome_chrome': '#8C8B7B',  # Chromosomes
    'telomere_turquoise': '#4A8B8C', # Telomeres
    'nucleus_navy': '#4A5C7B'        # Cell nucleus
}

# Musical notes mapped to DNA codons
CODON_NOTES = {
    'A': 440.00,   # A4
    'T': 493.88,   # B4
    'G': 523.25,   # C5
    'C': 587.33,   # D5
    'AT': 659.25,  # E5
    'GC': 698.46,  # F5
    'TA': 783.99,  # G5
    'CG': 880.00   # A5
}


@dataclass
class DNAStrand:
    """Single strand of DNA with musical properties"""
    
    sequence: str
    position: Tuple[float, float]
    strand_id: str
    mutation_rate: float = 0.01
    fitness: float = 1.0
    color: str = '#4A7B8C'
    twist_angle: float = 0.0
    expression_level: float = 1.0
    epigenetic_marks: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        self.validate_sequence()
        self.musical_signature = self.generate_musical_signature()
    
    def validate_sequence(self):
        """Ensure sequence contains only valid bases"""
        valid_bases = set('ATGC')
        self.sequence = ''.join([b for b in self.sequence.upper() if b in valid_bases])
        # Ensure sequence is not empty
        if not self.sequence:
            self.sequence = 'ATGC'  # Default sequence
    
    def generate_musical_signature(self) -> List[float]:
        """Convert DNA sequence to musical frequencies"""
        frequencies = []
        
        if not self.sequence:
            return [440.0]  # Default frequency if empty
        
        for i in range(0, len(self.sequence) - 1, 2):
            codon = self.sequence[i:i+2]
            if codon in CODON_NOTES:
                frequencies.append(CODON_NOTES[codon])
            else:
                # Single base frequency
                frequencies.append(CODON_NOTES.get(self.sequence[i], 440))
        
        # Ensure at least one frequency
        if not frequencies:
            frequencies.append(CODON_NOTES.get(self.sequence[0], 440))
        
        return frequencies
    
    def mutate(self) -> bool:
        """Apply random mutation to sequence"""
        if not self.sequence or random.random() >= self.mutation_rate:
            return False
            
        bases = ['A', 'T', 'G', 'C']
        pos = random.randint(0, len(self.sequence) - 1)
        old_base = self.sequence[pos]
        new_base = random.choice([b for b in bases if b != old_base])
        
        self.sequence = self.sequence[:pos] + new_base + self.sequence[pos+1:]
        self.musical_signature = self.generate_musical_signature()
        return True
    
    def crossover(self, other: 'DNAStrand') -> 'DNAStrand':
        """Perform genetic crossover with another strand"""
        # Single point crossover
        if len(self.sequence) > 2 and len(other.sequence) > 2:
            point1 = random.randint(1, len(self.sequence) - 1)
            point2 = random.randint(1, len(other.sequence) - 1)
            
            new_sequence = self.sequence[:point1] + other.sequence[point2:]
            
            return DNAStrand(
                sequence=new_sequence,
                position=((self.position[0] + other.position[0])/2,
                         (self.position[1] + other.position[1])/2),
                strand_id=f"{self.strand_id}x{other.strand_id}",
                mutation_rate=(self.mutation_rate + other.mutation_rate)/2,
                color=self._blend_colors(self.color, other.color)
            )
        return self
    
    def _blend_colors(self, color1: str, color2: str) -> str:
        """Blend two hex colors"""
        try:
            c1 = tuple(int(color1.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            c2 = tuple(int(color2.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            
            blended = tuple((c1[i] + c2[i])/2 for i in range(3))
            return f'#{int(blended[0]*255):02x}{int(blended[1]*255):02x}{int(blended[2]*255):02x}'
        except (ValueError, IndexError):
            return color1  # Return first color if blending fails


class ChromosomeDancer:
    """Animated chromosome that dances to its genetic rhythm"""
    
    def __init__(self, x: float, y: float, chromosome_number: int):
        self.x = x
        self.y = y
        self.initial_x = x  # Store initial position
        self.initial_y = y
        self.chromosome_number = chromosome_number
        self.phase = random.uniform(0, 2*np.pi)
        self.dance_amplitude = random.uniform(10, 30)
        self.rotation = 0
        self.size = 20 + chromosome_number * 2
        self.telomere_length = 100
        self.centromere_position = 0.4 + random.uniform(-0.1, 0.1)
        
        # Sister chromatids
        self.chromatid_separation = 0
        self.is_dividing = False
        
        # Gene expression levels for different regions
        self.gene_regions = self._generate_gene_regions()
    
    def _generate_gene_regions(self) -> List[Dict[str, Any]]:
        """Generate gene expression regions along chromosome"""
        regions = []
        n_regions = random.randint(5, 12)
        
        for i in range(n_regions):
            regions.append({
                'start': i / n_regions,
                'end': (i + 1) / n_regions,
                'expression': random.uniform(0, 1),
                'color': random.choice(list(GENETIC_PALETTE.values())),
                'frequency': random.uniform(200, 800),  # Hz
                'active': random.random() > 0.3
            })
        
        return regions
    
    def dance(self, time: float, music_beat: float):
        """Update chromosome position based on music"""
        # Oscillating movement around initial position
        self.x = self.initial_x + self.dance_amplitude * 0.1 * np.sin(time * music_beat + self.phase)
        self.y = self.initial_y + self.dance_amplitude * 0.05 * np.cos(time * music_beat * 1.3 + self.phase)
        
        # Rotation
        self.rotation += 0.02 * music_beat
        
        # Telomere shortening (slower)
        self.telomere_length = max(10, self.telomere_length * 0.9999)
        
        # Sister chromatid separation during division
        if self.is_dividing:
            self.chromatid_separation = min(50, self.chromatid_separation + 0.5)
        
        # Randomly start division
        if random.random() < 0.001:
            self.is_dividing = not self.is_dividing
            if not self.is_dividing:
                self.chromatid_separation = 0
    
    def get_shape_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get chromosome X-shape points"""
        # Create X-shaped chromosome
        length = self.size
        width = self.size * 0.3
        
        # Rotation matrix
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        
        # Define X shape with centromere
        points1 = []
        points2 = []
        
        # First arm (top-left to bottom-right)
        arm1_x = np.linspace(-length/2, length/2, 20)
        arm1_y = arm1_x * 0.8
        
        # Add centromere bulge
        for i, (x, y) in enumerate(zip(arm1_x, arm1_y)):
            t = (i / len(arm1_x))
            if abs(t - self.centromere_position) < 0.1:
                bulge = width * 2 * (1 - abs(t - self.centromere_position) / 0.1)
                y += bulge
            
            # Apply rotation
            rx = cos_r * x - sin_r * y + self.x
            ry = sin_r * x + cos_r * y + self.y
            points1.append([rx, ry])
        
        # Second arm (top-right to bottom-left)
        arm2_x = np.linspace(length/2, -length/2, 20)
        arm2_y = -arm2_x * 0.8
        
        # Sister chromatid separation
        arm2_x = arm2_x + self.chromatid_separation
        
        for i, (x, y) in enumerate(zip(arm2_x, arm2_y)):
            t = (i / len(arm2_x))
            if abs(t - self.centromere_position) < 0.1:
                bulge = width * 2 * (1 - abs(t - self.centromere_position) / 0.1)
                y += bulge
            
            # Apply rotation
            rx = cos_r * x - sin_r * y + self.x
            ry = sin_r * x + cos_r * y + self.y
            points2.append([rx, ry])
        
        return np.array(points1), np.array(points2)


class GeneHarmonic:
    """Represents a gene as a harmonic oscillator"""
    
    def __init__(self, gene_name: str, base_frequency: float):
        self.gene_name = gene_name
        self.base_frequency = max(100, min(2000, base_frequency))  # Clamp frequency
        self.harmonics = [1, 2, 3, 5, 8]  # Fibonacci harmonics
        self.amplitude = random.uniform(0.5, 1.0)
        self.phase = random.uniform(0, 2*np.pi)
        self.envelope = self._generate_envelope()
        self.expression_pattern = self._generate_expression_pattern()
        self.time_offset = 0
    
    def _generate_envelope(self) -> np.ndarray:
        """ADSR envelope for gene expression"""
        attack = np.linspace(0, 1, 10)
        decay = np.linspace(1, 0.7, 5)
        sustain = np.ones(20) * 0.7
        release = np.linspace(0.7, 0, 15)
        
        return np.concatenate([attack, decay, sustain, release])
    
    def _generate_expression_pattern(self) -> np.ndarray:
        """Generate temporal expression pattern"""
        t = np.linspace(0, 24, 100)  # 24-hour cycle
        
        # Circadian rhythm
        circadian = np.sin(2 * np.pi * t / 24 + self.phase)
        
        # Ultradian rhythms
        ultradian = 0.3 * np.sin(2 * np.pi * t / 4)
        
        # Stochastic bursts
        bursts = np.random.poisson(0.1, len(t))
        
        pattern = self.amplitude * (0.5 + 0.3 * circadian + 0.2 * ultradian) + 0.1 * bursts
        return np.clip(pattern, 0, 1)
    
    def synthesize_sound(self, time: float) -> float:
        """Generate sound wave from gene"""
        sound = 0
        
        for i, harmonic in enumerate(self.harmonics):
            frequency = self.base_frequency * harmonic
            amplitude = self.amplitude / (i + 1)  # Decrease amplitude for higher harmonics
            
            # Prevent audio artifacts
            if frequency > 20000:  # Above human hearing
                continue
                
            sound += amplitude * np.sin(2 * np.pi * frequency * time + self.phase)
        
        # Apply envelope
        envelope_index = int((time + self.time_offset) * 50) % len(self.envelope)
        sound *= self.envelope[envelope_index]
        
        return np.clip(sound, -1, 1)  # Prevent clipping


class QuantumProteinFolder:
    """Simulates protein folding in quantum superposition"""
    
    def __init__(self, amino_acid_sequence: str):
        self.sequence = amino_acid_sequence[:20]  # Limit sequence length for performance
        self.conformations = []
        self.energy_landscape = None
        self.folding_progress = 0
        self.quantum_states = self._initialize_quantum_states()
        self.current_conformation = None
        
        # Amino acid properties
        self.hydrophobicity = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90,
            'C': 0.29, 'Q': -0.85, 'E': -0.74, 'G': 0.48,
            'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50,
            'M': 0.64, 'F': 1.19, 'P': 0.12, 'S': -0.18,
            'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }
    
    def _initialize_quantum_states(self) -> List[np.ndarray]:
        """Initialize quantum superposition of conformations"""
        if not self.sequence:
            return [np.array([[0, 0, 0]])]
            
        n_states = min(5, len(self.sequence))  # Reduce for performance
        states = []
        
        for _ in range(n_states):
            # Random 3D conformation
            conformation = np.random.randn(len(self.sequence), 3)
            
            # Normalize to prevent explosion
            conformation = conformation * 0.1
            
            states.append(conformation)
        
        return states
    
    def fold_step(self, temperature: float = 1.0):
        """Perform one step of quantum-assisted folding"""
        if not self.quantum_states:
            return
            
        self.folding_progress = min(1.0, self.folding_progress + 0.005)  # Slower folding
        
        # Quantum annealing approach
        best_energy = float('inf')
        best_conformation = None
        
        for state in self.quantum_states:
            energy = self._calculate_energy(state, temperature)
            
            if energy < best_energy:
                best_energy = energy
                best_conformation = state
        
        self.current_conformation = best_conformation
        
        # Mutate quantum states (less frequently)
        if random.random() < 0.1:
            self._mutate_states(temperature)
    
    def _calculate_energy(self, conformation: np.ndarray, temperature: float) -> float:
        """Calculate folding energy using simplified force field"""
        if len(conformation) < 2:
            return 0
            
        energy = 0
        
        # Hydrophobic interactions (simplified)
        for i in range(len(self.sequence)):
            for j in range(i + 2, len(self.sequence)):
                if j >= len(conformation):
                    continue
                    
                distance = np.linalg.norm(conformation[i] - conformation[j])
                
                hydro_i = self.hydrophobicity.get(self.sequence[i], 0)
                hydro_j = self.hydrophobicity.get(self.sequence[j], 0)
                
                # Hydrophobic attraction
                if hydro_i > 0 and hydro_j > 0:
                    energy -= (hydro_i * hydro_j) / (1 + distance)
                
                # Steric clash penalty
                if distance < 0.5:
                    energy += 10 / (distance + 0.1)  # Reduced penalty
        
        return energy
    
    def _mutate_states(self, temperature: float):
        """Mutate quantum states for exploration"""
        for state in self.quantum_states:
            if random.random() < 0.3:
                # Random perturbation
                mutation = np.random.randn(*state.shape) * temperature * 0.01  # Smaller mutations
                state += mutation


class MutationWave:
    """Traveling wave of mutations across genetic landscape"""
    
    def __init__(self, origin: Tuple[float, float], wave_type: str = 'point'):
        self.origin = origin
        self.wave_type = wave_type
        self.radius = 0
        self.max_radius = 200
        self.speed = 1  # Slower propagation
        self.amplitude = 1.0
        self.frequency = 0.1
        self.color = GENETIC_PALETTE['mutation_magenta']
        self.active = True
        
        # Wave patterns
        self.patterns = {
            'point': self._point_mutation_pattern,
            'insertion': self._insertion_pattern,
            'deletion': self._deletion_pattern,
            'inversion': self._inversion_pattern,
            'duplication': self._duplication_pattern
        }
    
    def propagate(self, dt: float):
        """Propagate mutation wave"""
        self.radius += self.speed
        self.amplitude *= 0.99  # Slower decay
        
        if self.radius > self.max_radius or self.amplitude < 0.01:
            self.active = False
    
    def get_wave_points(self) -> List[Tuple[float, float, float]]:
        """Get points along wave front with intensity"""
        if not self.active:
            return []
        
        points = []
        n_points = max(20, min(100, int(2 * np.pi * self.radius / 5)))  # Limit points
        
        pattern_func = self.patterns.get(self.wave_type, self._point_mutation_pattern)
        
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            
            # Apply wave pattern
            r = self.radius + pattern_func(angle)
            
            x = self.origin[0] + r * np.cos(angle)
            y = self.origin[1] + r * np.sin(angle)
            intensity = self.amplitude * np.exp(-self.radius / 100)
            
            points.append((x, y, intensity))
        
        return points
    
    def _point_mutation_pattern(self, angle: float) -> float:
        """Simple circular wave"""
        return 5 * np.sin(angle * 10 + self.radius * 0.1)
    
    def _insertion_pattern(self, angle: float) -> float:
        """Expanding spiral pattern"""
        return 10 * np.sin(angle * 3 + self.radius * 0.2)
    
    def _deletion_pattern(self, angle: float) -> float:
        """Contracting pattern"""
        return -5 * np.cos(angle * 4 - self.radius * 0.15)
    
    def _inversion_pattern(self, angle: float) -> float:
        """Flipping pattern"""
        return 8 * np.sign(np.sin(angle * 2)) * np.cos(self.radius * 0.1)
    
    def _duplication_pattern(self, angle: float) -> float:
        """Double wave pattern"""
        return 5 * (np.sin(angle * 6) + np.sin(angle * 3 + np.pi/4))


class EvolutionLandscape:
    """Fitness landscape for genetic evolution"""
    
    def __init__(self, resolution: int = 50):  # Reduced resolution for performance
        self.resolution = resolution
        self.landscape = self._generate_landscape()
        self.populations = []
        self.time = 0
        
    def _generate_landscape(self) -> np.ndarray:
        """Generate complex fitness landscape with peaks and valleys"""
        x = np.linspace(-5, 5, self.resolution)
        y = np.linspace(-5, 5, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Multiple fitness peaks (adaptive landscape)
        landscape = np.zeros_like(X)
        
        # Add Gaussian peaks
        peaks = [
            (0, 0, 1.0, 1.0),      # Central peak
            (2, 2, 0.8, 0.7),      # Secondary peaks
            (-2, 1, 0.7, 0.8),
            (1, -2, 0.6, 0.9),
            (-1, -1, 0.5, 0.6)
        ]
        
        for px, py, height, width in peaks:
            landscape += height * np.exp(-((X - px)**2 + (Y - py)**2) / (2 * width**2))
        
        # Add valleys (deleterious regions)
        valleys = [
            (1, 1, -0.5, 0.5),
            (-1, 2, -0.3, 0.7)
        ]
        
        for vx, vy, depth, width in valleys:
            landscape += depth * np.exp(-((X - vx)**2 + (Y - vy)**2) / (2 * width**2))
        
        # Add noise for ruggedness
        landscape += 0.05 * np.random.randn(self.resolution, self.resolution)  # Less noise
        
        # Normalize to [0, 1]
        landscape = (landscape - landscape.min()) / (landscape.max() - landscape.min())
        
        return landscape
    
    def add_population(self, x: float, y: float, size: int = 100):
        """Add new population to landscape"""
        self.populations.append({
            'x': x,
            'y': y,
            'size': size,
            'fitness': self.get_fitness(x, y),
            'color': random.choice(list(GENETIC_PALETTE.values())),
            'velocity': [random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)]  # Slower movement
        })
    
    def get_fitness(self, x: float, y: float) -> float:
        """Get fitness value at position"""
        # Convert to grid coordinates
        i = int((x + 5) / 10 * self.resolution)
        j = int((y + 5) / 10 * self.resolution)
        
        i = np.clip(i, 0, self.resolution - 1)
        j = np.clip(j, 0, self.resolution - 1)
        
        return self.landscape[j, i]
    
    def evolve_populations(self, dt: float):
        """Evolve populations on landscape"""
        for pop in self.populations:
            # Calculate gradient (hill climbing)
            dx = 0.01
            fitness_x_plus = self.get_fitness(pop['x'] + dx, pop['y'])
            fitness_x_minus = self.get_fitness(pop['x'] - dx, pop['y'])
            fitness_y_plus = self.get_fitness(pop['x'], pop['y'] + dx)
            fitness_y_minus = self.get_fitness(pop['x'], pop['y'] - dx)
            
            grad_x = (fitness_x_plus - fitness_x_minus) / (2 * dx)
            grad_y = (fitness_y_plus - fitness_y_minus) / (2 * dx)
            
            # Update velocity (with momentum)
            pop['velocity'][0] = 0.95 * pop['velocity'][0] + 0.05 * grad_x  # More momentum
            pop['velocity'][1] = 0.95 * pop['velocity'][1] + 0.05 * grad_y
            
            # Add random drift (reduced)
            pop['velocity'][0] += random.gauss(0, 0.01)
            pop['velocity'][1] += random.gauss(0, 0.01)
            
            # Update position
            pop['x'] += pop['velocity'][0] * dt
            pop['y'] += pop['velocity'][1] * dt
            
            # Boundary conditions
            pop['x'] = np.clip(pop['x'], -4.9, 4.9)
            pop['y'] = np.clip(pop['y'], -4.9, 4.9)
            
            # Update fitness
            pop['fitness'] = self.get_fitness(pop['x'], pop['y'])
            
            # Population size changes based on fitness (slower)
            growth_rate = 0.05 * (pop['fitness'] - 0.5)
            pop['size'] = max(10, min(500, pop['size'] * (1 + growth_rate * dt)))


class GeneticSymphonyVisualizer:
    """Main visualization engine for quantum genetic music system"""
    
    def __init__(self, figsize: Tuple[int, int] = (16, 10)):  # Smaller figure
        # Setup figure
        self.fig = plt.figure(figsize=figsize, facecolor='#0a0a1a')
        self.fig.suptitle('Quantum DNA Music Evolution System 2025', 
                         fontsize=16, color='#E5E5F5', fontweight='bold')
        
        # Create layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        self.ax_main = self.fig.add_subplot(gs[0:2, 0:2])        # DNA helix dance floor
        self.ax_chromosome = self.fig.add_subplot(gs[0, 2])      # Chromosome dancers
        self.ax_landscape = self.fig.add_subplot(gs[1, 2])       # Evolution landscape
        self.ax_protein = self.fig.add_subplot(gs[0, 3])         # Protein folding
        self.ax_music = self.fig.add_subplot(gs[1, 3])           # Music waveform
        self.ax_spectrum = self.fig.add_subplot(gs[2, 0])        # Frequency spectrum
        self.ax_mutations = self.fig.add_subplot(gs[2, 1])       # Mutation tracker
        self.ax_expression = self.fig.add_subplot(gs[2, 2])      # Gene expression
        self.ax_phylogeny = self.fig.add_subplot(gs[2, 3])       # Phylogenetic tree
        
        # Style axes
        for ax in self.fig.axes:
            ax.set_facecolor('#1a1a2a')
            ax.tick_params(colors='#909095', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#353545')
                spine.set_linewidth(0.5)
        
        # Initialize components
        self.dna_strands = []
        self.chromosomes = []
        self.mutations = []
        self.proteins = []
        self.harmonics = []
        
        # Evolution landscape
        self.landscape = EvolutionLandscape(resolution=30)  # Further reduced
        
        # Music synthesis
        self.audio_buffer = deque(maxlen=500)  # Smaller buffer
        self.frequency_buffer = deque(maxlen=50)
        
        # Animation state
        self.frame = 0
        self.beat_phase = 0
        
        # Initialize genetic orchestra
        self._initialize_genetic_orchestra()
    
    def _initialize_genetic_orchestra(self):
        """Setup initial genetic elements"""
        # Create DNA strands (fewer for performance)
        sequences = [
            'ATGCATGCTAGCTAGCATGC',
            'GCTAGCTAGCTAGCTAGCTA',
            'TATATATGCGCGCATATAT'
        ]
        
        colors = list(GENETIC_PALETTE.values())
        
        for i, seq in enumerate(sequences):
            strand = DNAStrand(
                sequence=seq,
                position=(200 + i * 80, 250 + np.sin(i) * 50),
                strand_id=f'strand_{i}',
                color=colors[i % len(colors)]
            )
            self.dna_strands.append(strand)
        
        # Create chromosome dancers (fewer)
        for i in range(3):
            chromosome = ChromosomeDancer(
                x=100 + i * 50,
                y=100 + i * 30,
                chromosome_number=i + 1
            )
            self.chromosomes.append(chromosome)
        
        # Initialize protein folder
        self.proteins.append(
            QuantumProteinFolder('ARNDCQEGHILKMF')  # Shorter sequence
        )
        
        # Create gene harmonics (fewer)
        genes = ['p53', 'BRCA1', 'MYC']
        for gene in genes:
            harmonic = GeneHarmonic(gene, random.uniform(200, 800))
            self.harmonics.append(harmonic)
        
        # Add initial populations to landscape (fewer)
        for _ in range(2):
            self.landscape.add_population(
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.randint(50, 150)
            )
    
    def update_frame(self, frame_num: int):
        """Update all components for animation frame"""
        self.frame = frame_num
        time = frame_num * 0.05  # Slower time progression
        
        # Calculate music beat
        self.beat_phase += 0.05
        music_beat = 1 + 0.3 * np.sin(self.beat_phase)
        
        # Clear axes
        for ax in self.fig.axes:
            ax.clear()
            ax.set_facecolor('#1a1a2a')
        
        # Update all components
        self._update_genetics(time, music_beat)
        
        # Draw all visualizations
        self._draw_dna_helix_dance()
        self._draw_chromosome_dancers()
        self._draw_evolution_landscape()
        self._draw_protein_folding()
        self._draw_music_waveform()
        self._draw_frequency_spectrum()
        self._draw_mutation_tracker()
        self._draw_gene_expression()
        self._draw_phylogenetic_tree()
        
        # Trigger events (less frequently)
        if frame_num % 60 == 0:
            self._trigger_mutation_wave()
        
        if frame_num % 100 == 0:
            self._perform_crossover()
    
    def _update_genetics(self, time: float, music_beat: float):
        """Update all genetic components"""
        # Update DNA strands
        for strand in self.dna_strands:
            # Twist animation
            strand.twist_angle += 0.03 * music_beat
            
            # Random mutations (less frequent)
            if random.random() < 0.01 and strand.mutate():
                self.mutations.append(MutationWave(strand.position, 'point'))
            
            # Update position with music (smaller movements)
            strand.position = (
                strand.position[0] + 1 * np.sin(time * 0.3),
                strand.position[1] + 1 * np.cos(time * 0.2)
            )
        
        # Update chromosomes
        for chromosome in self.chromosomes:
            chromosome.dance(time, music_beat)
        
        # Update mutation waves
        for mutation in self.mutations[:]:
            mutation.propagate(0.05)
            if not mutation.active:
                self.mutations.remove(mutation)
        
        # Limit number of mutations
        if len(self.mutations) > 5:
            self.mutations = self.mutations[-5:]
        
        # Update protein folding
        for protein in self.proteins:
            protein.fold_step(temperature=1.5 - self.frame / 1000)
        
        # Update evolution landscape
        self.landscape.evolve_populations(0.05)
        
        # Synthesize music
        self._synthesize_genetic_music(time)
    
    def _draw_dna_helix_dance(self):
        """Draw animated DNA double helix"""
        self.ax_main.set_xlim(0, 600)
        self.ax_main.set_ylim(0, 400)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('DNA Helix Dance Floor', color='#E5E5F5', fontsize=10)
        
        # Draw DNA strands as double helixes
        for strand in self.dna_strands:
            # Generate helix points (fewer points)
            t = np.linspace(0, 4*np.pi, 50)
            
            # First helix strand
            x1 = strand.position[0] + 30 * np.cos(t + strand.twist_angle)
            y1 = strand.position[1] + t * 8 - 20
            
            # Second helix strand (180 degrees offset)
            x2 = strand.position[0] + 30 * np.cos(t + strand.twist_angle + np.pi)
            y2 = strand.position[1] + t * 8 - 20
            
            # Draw phosphate backbone
            self.ax_main.plot(x1, y1, color=strand.color, linewidth=2, alpha=0.8)
            self.ax_main.plot(x2, y2, color=strand.color, linewidth=2, alpha=0.8)
            
            # Draw base pairs (fewer)
            for i in range(0, len(t), 10):
                if i < len(strand.sequence) * 5:
                    base_idx = (i // 5) % len(strand.sequence)
                    base = strand.sequence[base_idx]
                    base_color = {
                        'A': GENETIC_PALETTE['adenine_azure'],
                        'T': GENETIC_PALETTE['thymine_teal'],
                        'G': GENETIC_PALETTE['guanine_gold'],
                        'C': GENETIC_PALETTE['cytosine_coral']
                    }.get(base, '#FFFFFF')
                    
                    if i < len(x1) and i < len(x2):
                        self.ax_main.plot([x1[i], x2[i]], [y1[i], y2[i]],
                                        color=base_color, linewidth=1, alpha=0.6)
        
        # Draw mutation waves
        for mutation in self.mutations:
            points = mutation.get_wave_points()
            if points:
                x_points = [p[0] for p in points if 0 <= p[0] <= 600 and 0 <= p[1] <= 400]
                y_points = [p[1] for p in points if 0 <= p[0] <= 600 and 0 <= p[1] <= 400]
                intensities = [p[2] for p in points if 0 <= p[0] <= 600 and 0 <= p[1] <= 400]
                
                if x_points and y_points:
                    # Close the circle
                    x_points.append(x_points[0])
                    y_points.append(y_points[0])
                    
                    max_intensity = max(intensities) if intensities else 0.1
                    self.ax_main.plot(x_points, y_points,
                                    color=mutation.color,
                                    linewidth=2 * max_intensity,
                                    alpha=max_intensity)
    
    def _draw_chromosome_dancers(self):
        """Draw dancing chromosomes"""
        self.ax_chromosome.set_xlim(0, 300)
        self.ax_chromosome.set_ylim(0, 200)
        self.ax_chromosome.set_aspect('equal')
        self.ax_chromosome.set_title('Chromosome Ballet', color='#E5E5F5', fontsize=10)
        
        for chromosome in self.chromosomes:
            arm1, arm2 = chromosome.get_shape_points()
            
            # Draw chromosome arms
            self.ax_chromosome.plot(arm1[:, 0], arm1[:, 1],
                                  color=GENETIC_PALETTE['chromosome_chrome'],
                                  linewidth=3, solid_capstyle='round')
            self.ax_chromosome.plot(arm2[:, 0], arm2[:, 1],
                                  color=GENETIC_PALETTE['chromosome_chrome'],
                                  linewidth=3, solid_capstyle='round')
            
            # Draw gene expression regions (fewer)
            active_regions = [r for r in chromosome.gene_regions if r['active']][:3]
            for region in active_regions:
                start_idx = int(region['start'] * len(arm1))
                end_idx = int(region['end'] * len(arm1))
                
                if start_idx < len(arm1) and end_idx <= len(arm1):
                    self.ax_chromosome.plot(
                        arm1[start_idx:end_idx, 0],
                        arm1[start_idx:end_idx, 1],
                        color=region['color'],
                        linewidth=5,
                        alpha=region['expression']
                    )
            
            # Draw telomeres
            telomere_size = max(1, chromosome.telomere_length / 20)
            for arm_end in [arm1[0], arm1[-1], arm2[0], arm2[-1]]:
                circle = Circle(arm_end, telomere_size,
                              color=GENETIC_PALETTE['telomere_turquoise'],
                              alpha=0.7)
                self.ax_chromosome.add_patch(circle)
    
    def _draw_evolution_landscape(self):
        """Draw fitness landscape with populations"""
        self.ax_landscape.set_title('Evolution Landscape', color='#E5E5F5', fontsize=10)
        
        # Display landscape
        im = self.ax_landscape.imshow(self.landscape.landscape,
                                     cmap='terrain',
                                     extent=[-5, 5, -5, 5],
                                     origin='lower',
                                     alpha=0.7)
        
        # Draw populations
        for pop in self.landscape.populations:
            circle = Circle((pop['x'], pop['y']),
                          radius=max(0.1, np.sqrt(pop['size']) / 30),
                          color=pop['color'],
                          alpha=0.6 + 0.4 * pop['fitness'])
            self.ax_landscape.add_patch(circle)
            
            # Draw velocity vector
            vel_scale = 5
            self.ax_landscape.arrow(pop['x'], pop['y'],
                                   pop['velocity'][0] * vel_scale, pop['velocity'][1] * vel_scale,
                                   color='white', alpha=0.3,
                                   head_width=0.1, head_length=0.05)
        
        self.ax_landscape.set_xlim(-5, 5)
        self.ax_landscape.set_ylim(-5, 5)
        self.ax_landscape.set_xlabel('Trait 1', color='#909095', fontsize=8)
        self.ax_landscape.set_ylabel('Trait 2', color='#909095', fontsize=8)
    
    def _draw_protein_folding(self):
        """Draw protein folding visualization"""
        self.ax_protein.set_title('Quantum Protein Folding', color='#E5E5F5', fontsize=10)
        
        if self.proteins and self.proteins[0].current_conformation is not None:
            protein = self.proteins[0]
            conf = protein.current_conformation
            
            # Project 3D to 2D
            x = conf[:, 0]
            y = conf[:, 1]
            
            # Draw backbone
            self.ax_protein.plot(x, y, 'o-',
                               color=GENETIC_PALETTE['protein_pearl'],
                               linewidth=2, markersize=4, alpha=0.8)
            
            # Color by hydrophobicity (sample points)
            for i in range(0, len(protein.sequence), 2):
                if i < len(x):
                    aa = protein.sequence[i]
                    hydro = protein.hydrophobicity.get(aa, 0)
                    if hydro > 0:
                        color = GENETIC_PALETTE['enzyme_electric']
                    else:
                        color = GENETIC_PALETTE['evolution_emerald']
                    
                    self.ax_protein.plot(x[i], y[i], 'o',
                                       color=color, markersize=6,
                                       alpha=min(1.0, abs(hydro)))
            
            # Draw folding progress
            self.ax_protein.text(0.02, 0.98,
                               f'Folding: {protein.folding_progress:.1%}',
                               transform=self.ax_protein.transAxes,
                               color='white', fontsize=8,
                               verticalalignment='top')
        
        self.ax_protein.set_xlim(-1, 1)
        self.ax_protein.set_ylim(-1, 1)
        self.ax_protein.set_aspect('equal')
    
    def _draw_music_waveform(self):
        """Draw genetic music waveform"""
        self.ax_music.set_title('Genetic Symphony', color='#E5E5F5', fontsize=10)
        
        if len(self.audio_buffer) > 1:
            time_axis = np.arange(len(self.audio_buffer))
            waveform = list(self.audio_buffer)
            
            self.ax_music.fill_between(time_axis, 0, waveform,
                                      color=GENETIC_PALETTE['helix_violet'],
                                      alpha=0.3)
            self.ax_music.plot(time_axis, waveform,
                             color=GENETIC_PALETTE['helix_violet'],
                             linewidth=1)
            
            self.ax_music.set_ylim(-1, 1)
            self.ax_music.set_xlabel('Time', color='#909095', fontsize=8)
            self.ax_music.set_ylabel('Amplitude', color='#909095', fontsize=8)
        else:
            self.ax_music.text(0.5, 0.5, 'Generating Music...', 
                             transform=self.ax_music.transAxes,
                             color='white', ha='center', va='center')
    
    def _draw_frequency_spectrum(self):
        """Draw frequency spectrum of genetic music"""
        self.ax_spectrum.set_title('Frequency Spectrum', color='#E5E5F5', fontsize=10)
        
        if len(self.audio_buffer) > 64:
            # Compute FFT (smaller window)
            signal_array = np.array(list(self.audio_buffer)[-128:])
            
            # Apply window function
            windowed_signal = signal_array * np.hanning(len(signal_array))
            fft_vals = np.fft.rfft(windowed_signal)
            freqs = np.fft.rfftfreq(len(signal_array), 1/1000)  # Lower sample rate
            
            # Plot spectrum (fewer points)
            n_plot = min(50, len(fft_vals))
            self.ax_spectrum.semilogy(freqs[:n_plot], np.abs(fft_vals[:n_plot]) + 1e-10,
                                    color=GENETIC_PALETTE['mutation_magenta'],
                                    linewidth=1.5)
            
            self.ax_spectrum.fill_between(freqs[:n_plot], 1e-10, np.abs(fft_vals[:n_plot]) + 1e-10,
                                         color=GENETIC_PALETTE['mutation_magenta'],
                                         alpha=0.3)
            
            self.ax_spectrum.set_xlabel('Frequency (Hz)', color='#909095', fontsize=8)
            self.ax_spectrum.set_ylabel('Power', color='#909095', fontsize=8)
            self.ax_spectrum.set_xlim(0, 500)
        else:
            self.ax_spectrum.text(0.5, 0.5, 'Building Spectrum...', 
                                transform=self.ax_spectrum.transAxes,
                                color='white', ha='center', va='center')
    
    def _draw_mutation_tracker(self):
        """Track mutation types and frequencies"""
        self.ax_mutations.set_title('Mutation Monitor', color='#E5E5F5', fontsize=10)
        
        # Count mutations by type
        mutation_types = ['Point', 'Insert', 'Delete']  # Fewer types
        counts = [len([m for m in self.mutations if m.wave_type == t.lower()]) 
                 for t in ['point', 'insertion', 'deletion']]
        
        colors = [
            GENETIC_PALETTE['adenine_azure'],
            GENETIC_PALETTE['thymine_teal'],
            GENETIC_PALETTE['guanine_gold']
        ]
        
        bars = self.ax_mutations.bar(range(len(mutation_types)), counts,
                                    color=colors, alpha=0.7)
        
        self.ax_mutations.set_xticks(range(len(mutation_types)))
        self.ax_mutations.set_xticklabels(mutation_types, rotation=45,
                                         fontsize=7, color='#909095')
        self.ax_mutations.set_ylabel('Count', color='#909095', fontsize=8)
        self.ax_mutations.set_ylim(0, max(5, max(counts) + 1))
    
    def _draw_gene_expression(self):
        """Draw gene expression heatmap"""
        self.ax_expression.set_title('Gene Expression', color='#E5E5F5', fontsize=10)
        
        # Create expression matrix
        n_genes = len(self.harmonics)
        n_samples = 8  # Fewer samples
        
        if n_genes > 0:
            expression_matrix = np.zeros((n_genes, n_samples))
            
            for i, harmonic in enumerate(self.harmonics):
                # Sample expression pattern
                pattern_len = len(harmonic.expression_pattern)
                sample_points = np.linspace(0, pattern_len-1, n_samples)
                for j, point in enumerate(sample_points):
                    idx = int(point) % pattern_len
                    expression_matrix[i, j] = harmonic.expression_pattern[idx]
            
            # Draw heatmap
            im = self.ax_expression.imshow(expression_matrix,
                                          cmap='YlOrRd',
                                          aspect='auto',
                                          interpolation='bicubic',
                                          vmin=0, vmax=1)
            
            self.ax_expression.set_yticks(range(n_genes))
            self.ax_expression.set_yticklabels([h.gene_name for h in self.harmonics],
                                              fontsize=7, color='#909095')
            self.ax_expression.set_xlabel('Time Points', color='#909095', fontsize=8)
        else:
            self.ax_expression.text(0.5, 0.5, 'No Genes', 
                                  transform=self.ax_expression.transAxes,
                                  color='white', ha='center', va='center')
    
    def _draw_phylogenetic_tree(self):
        """Draw evolutionary tree"""
        self.ax_phylogeny.set_title('Phylogenetic Tree', color='#E5E5F5', fontsize=10)
        
        # Simple tree structure
        def add_branches(x, y, level, max_level=3):  # Fewer levels
            if level >= max_level:
                # Draw leaf node
                self.ax_phylogeny.plot(x, y, 'o',
                                     color=GENETIC_PALETTE['evolution_emerald'],
                                     markersize=6)
                return
            
            # Branch left and right
            offset = 0.2 / (level + 1)
            height = 0.25
            
            # Left branch
            new_x = x - offset
            new_y = y + height
            self.ax_phylogeny.plot([x, new_x], [y, new_y],
                                  color=GENETIC_PALETTE['evolution_emerald'],
                                  linewidth=2, alpha=0.8)
            add_branches(new_x, new_y, level + 1, max_level)
            
            # Right branch
            new_x = x + offset
            new_y = y + height
            self.ax_phylogeny.plot([x, new_x], [y, new_y],
                                  color=GENETIC_PALETTE['evolution_emerald'],
                                  linewidth=2, alpha=0.8)
            add_branches(new_x, new_y, level + 1, max_level)
            
            # Draw internal node
            self.ax_phylogeny.plot(x, y, 'o',
                                 color='white',
                                 markersize=4)
        
        # Start tree
        add_branches(0.5, 0.1, 0)
        
        self.ax_phylogeny.set_xlim(0, 1)
        self.ax_phylogeny.set_ylim(0, 1)
        self.ax_phylogeny.set_xticks([])
        self.ax_phylogeny.set_yticks([])
    
    def _synthesize_genetic_music(self, time: float):
        """Synthesize music from genetic sequences"""
        sample = 0
        
        # Combine DNA strand frequencies
        for strand in self.dna_strands:
            if strand.musical_signature:
                freq_idx = int(time * 5) % len(strand.musical_signature)  # Slower progression
                frequency = strand.musical_signature[freq_idx]
                
                # Generate tone with envelope
                envelope = np.exp(-((time % 1) * 2))  # Decay envelope
                sample += 0.1 * envelope * np.sin(2 * np.pi * frequency * time * 0.01)
        
        # Add gene harmonics
        for harmonic in self.harmonics:
            sample += 0.03 * harmonic.synthesize_sound(time)
        
        # Clip and add to buffer
        sample = np.clip(sample, -1, 1)
        self.audio_buffer.append(sample)
    
    def _trigger_mutation_wave(self):
        """Trigger new mutation wave"""
        if self.dna_strands:
            strand = random.choice(self.dna_strands)
            wave_type = random.choice(['point', 'insertion', 'deletion'])
            
            mutation = MutationWave(strand.position, wave_type)
            self.mutations.append(mutation)
    
    def _perform_crossover(self):
        """Perform genetic crossover between strands"""
        if len(self.dna_strands) >= 2:
            parent1, parent2 = random.sample(self.dna_strands, 2)
            offspring = parent1.crossover(parent2)
            
            # Replace random strand instead of weakest
            idx = random.randint(0, len(self.dna_strands) - 1)
            self.dna_strands[idx] = offspring
    
    def animate(self):
        """Start animation"""
        def update(frame):
            try:
                self.update_frame(frame)
            except Exception as e:
                print(f"Animation error at frame {frame}: {e}")
            return []
        
        self.anim = animation.FuncAnimation(
            self.fig, update,
            frames=2000,
            interval=100,  # Slower animation
            blit=False,
            repeat=True
        )
        
        plt.show()


def run_genetic_symphony():
    """Main entry point"""
    print("Quantum DNA Music Evolution System 2025")
    print("Novel Features:")
    print("- DNA sequences converted to musical frequencies")
    print("- Dancing X-shaped chromosomes with telomeres")
    print("- Quantum protein folding visualization")
    print("- Evolution on fitness landscapes")
    print("- Mutation waves with different patterns")
    print("- Gene expression heatmaps")
    print("- Real-time genetic music synthesis")
    print("- Phylogenetic tree generation")
    print("- 12 bioluminescent ocean colors")
    print("\nOptimized for performance and stability!")
    
    try:
        visualizer = GeneticSymphonyVisualizer()
        visualizer.animate()
    except Exception as e:
        print(f"Error starting visualization: {e}")
        print("Please check your Python environment and dependencies")


if __name__ == "__main__":
    run_genetic_symphony()