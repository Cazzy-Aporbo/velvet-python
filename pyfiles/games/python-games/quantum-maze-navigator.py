"""
QUANTUM MAZE NAVIGATOR - Multi-Dimensional Puzzle Game
A comprehensive quantum computing educational game that teaches quantum mechanics
through navigating mazes that exist in superposition states. Players manipulate
qubits, apply quantum gates, and use entanglement to solve increasingly complex
multi-dimensional mazes.

This implementation showcases advanced Python 3.8+ features including:
- Walrus operator (:=)
- Type hints and Protocol classes
- Dataclasses with field validators
- AsyncIO for parallel quantum state calculations
- Functools cache and lru_cache
- Pattern matching (structural for Python 3.8 compatibility)
- Context managers and decorators
- Generator expressions and comprehensions
- Abstract base classes and protocols

Author: Advanced Quantum Game Systems
Version: 2.0.0
Python Requirements: 3.8+
Dependencies: pygame, numpy, asyncio
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property, lru_cache
from typing import (
    Final,
    Protocol,
    TypeVar,
)

import numpy as np
import pygame

# Initialize Pygame
pygame.init()

# Type variables for generic programming
T = TypeVar('T')
QuantumState = TypeVar('QuantumState', bound='BaseQuantumState')

# Constants using Final (Python 3.8+)
PLANCK_CONSTANT: Final[float] = 6.62607015e-34
SPEED_OF_LIGHT: Final[int] = 299792458


class QuantumGateType(Enum):
    """Enumeration of quantum gates with their matrix representations"""
    HADAMARD = auto()
    PAULI_X = auto()  # NOT gate
    PAULI_Y = auto()
    PAULI_Z = auto()
    CNOT = auto()     # Controlled NOT
    PHASE = auto()
    TOFFOLI = auto()  # Controlled-controlled NOT
    SWAP = auto()

    @cached_property
    def matrix(self) -> np.ndarray:
        """Return the unitary matrix for this gate using cached property"""
        matrices = {
            QuantumGateType.HADAMARD: np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            QuantumGateType.PAULI_X: np.array([[0, 1], [1, 0]]),
            QuantumGateType.PAULI_Y: np.array([[0, -1j], [1j, 0]]),
            QuantumGateType.PAULI_Z: np.array([[1, 0], [0, -1]]),
            QuantumGateType.PHASE: np.array([[1, 0], [0, 1j]]),
        }
        return matrices.get(self, np.eye(2))


class MeasurementBasis(Enum):
    """Different measurement bases for quantum states"""
    COMPUTATIONAL = auto()  # |0⟩, |1⟩
    HADAMARD = auto()      # |+⟩, |-⟩
    CIRCULAR = auto()      # |R⟩, |L⟩


@dataclass(frozen=True)
class QubitState:
    """
    Immutable qubit state with validation using dataclass features
    Demonstrates Python 3.8+ dataclass field validators and frozen classes
    """
    alpha: complex = field(default=1.0+0j)
    beta: complex = field(default=0.0+0j)
    phase: float = field(default=0.0)

    def __post_init__(self):
        """Validate that the qubit state is normalized"""
        norm = abs(self.alpha)**2 + abs(self.beta)**2
        if not np.isclose(norm, 1.0, rtol=1e-5):
            # Using object.__setattr__ because dataclass is frozen
            normalized_alpha = self.alpha / np.sqrt(norm)
            normalized_beta = self.beta / np.sqrt(norm)
            object.__setattr__(self, 'alpha', normalized_alpha)
            object.__setattr__(self, 'beta', normalized_beta)

    @property
    def probability_zero(self) -> float:
        """Probability of measuring |0⟩"""
        return abs(self.alpha) ** 2

    @property
    def probability_one(self) -> float:
        """Probability of measuring |1⟩"""
        return abs(self.beta) ** 2

    def apply_gate(self, gate: QuantumGateType) -> QubitState:
        """Apply a quantum gate and return new state"""
        state_vector = np.array([self.alpha, self.beta])
        new_state = gate.matrix @ state_vector
        return QubitState(alpha=new_state[0], beta=new_state[1], phase=self.phase)

    @lru_cache(maxsize=128)
    def get_bloch_coordinates(self) -> tuple[float, float, float]:
        """Get Bloch sphere coordinates (cached for performance)"""
        theta = 2 * np.arccos(abs(self.alpha))
        phi = np.angle(self.beta) - np.angle(self.alpha)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return (float(x), float(y), float(z))


class QuantumProtocol(Protocol):
    """Protocol defining quantum object interface (Python 3.8+ Protocol)"""

    def measure(self, basis: MeasurementBasis) -> int:
        """Measure the quantum state in given basis"""
        ...

    def entangle(self, other: QuantumProtocol) -> None:
        """Create entanglement with another quantum object"""
        ...

    def get_amplitude(self) -> complex:
        """Get the probability amplitude"""
        ...


@dataclass
class QuantumMazeCell:
    """
    Represents a cell in the quantum maze that can exist in superposition
    Demonstrates complex dataclass with mutable default factory
    """
    x: int
    y: int
    z: int  # Dimension layer
    states: list[QubitState] = field(default_factory=list)
    is_wall: bool = False
    is_goal: bool = False
    is_start: bool = False
    entangled_with: set[tuple[int, int, int]] | None = field(default_factory=set)
    measurement_history: list[int] = field(default_factory=list)
    superposition_amplitude: complex = 1.0 + 0j

    def __hash__(self):
        """Make cell hashable for use in sets"""
        return hash((self.x, self.y, self.z))

    def add_qubit_state(self, state: QubitState) -> None:
        """Add a qubit state to this cell's superposition"""
        self.states.append(state)
        # Update superposition amplitude using walrus operator (Python 3.8+)
        if (num_states := len(self.states)) > 1:
            self.superposition_amplitude = complex(1/np.sqrt(num_states), 0)

    def collapse(self) -> bool:
        """Collapse superposition and return if cell is passable"""
        if not self.states:
            return not self.is_wall

        # Weighted random collapse based on probability amplitudes
        probabilities = [abs(state.alpha)**2 for state in self.states]
        probabilities = np.array(probabilities) / sum(probabilities)

        collapsed_index = np.random.choice(len(self.states), p=probabilities)
        self.measurement_history.append(collapsed_index)

        # Cell is passable if collapsed state has high probability_zero
        return self.states[collapsed_index].probability_zero > 0.5


class BaseQuantumState(ABC):
    """Abstract base class for quantum states"""

    @abstractmethod
    def evolve(self, time_step: float) -> None:
        """Evolve the quantum state over time"""
        pass

    @abstractmethod
    def measure(self) -> any:
        """Perform measurement on the state"""
        pass

    @abstractmethod
    async def calculate_interference(self) -> float:
        """Asynchronously calculate quantum interference"""
        pass


class QuantumMazeState(BaseQuantumState):
    """
    The complete quantum state of the maze
    Demonstrates async methods and complex state management
    """

    def __init__(self, dimensions: tuple[int, int, int]):
        self.dimensions = dimensions
        self.cells: dict[tuple[int, int, int], QuantumMazeCell] = {}
        self.player_superposition: list[tuple[int, int, int]] = []
        self.entanglement_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
        self.wave_function: np.ndarray | None = None
        self.decoherence_rate: float = 0.01
        self.time_evolution_unitary: np.ndarray | None = None

    def evolve(self, time_step: float) -> None:
        """Implement Schrödinger equation evolution"""
        if self.wave_function is None:
            self._initialize_wave_function()

        # Apply time evolution operator
        if self.time_evolution_unitary is None:
            self._create_time_evolution_operator(time_step)

        self.wave_function = self.time_evolution_unitary @ self.wave_function

        # Apply decoherence
        self._apply_decoherence()

    def measure(self) -> tuple[int, int, int]:
        """Collapse player position from superposition"""
        if not self.player_superposition:
            return (0, 0, 0)

        # Calculate probabilities from wave function
        probabilities = self._calculate_position_probabilities()

        # Collapse to single position
        index = np.random.choice(len(self.player_superposition), p=probabilities)
        collapsed_position = self.player_superposition[index]

        # Update wave function after measurement
        self._collapse_wave_function(collapsed_position)

        return collapsed_position

    async def calculate_interference(self) -> float:
        """
        Asynchronously calculate quantum interference patterns
        Uses asyncio for parallel computation
        """
        async def calculate_path_amplitude(path: list[tuple[int, int, int]]) -> complex:
            """Calculate amplitude for a specific path"""
            amplitude = complex(1, 0)
            for i, pos in enumerate(path[:-1]):
                next_pos = path[i + 1]
                # Simulate quantum phase accumulation
                phase = self._calculate_phase_shift(pos, next_pos)
                amplitude *= np.exp(1j * phase)
                await asyncio.sleep(0)  # Yield control for true async
            return amplitude

        # Find all possible paths in superposition
        paths = self._find_all_quantum_paths()

        # Calculate amplitudes in parallel
        tasks = [calculate_path_amplitude(path) for path in paths]
        amplitudes = await asyncio.gather(*tasks)

        # Calculate interference pattern
        total_amplitude = sum(amplitudes)
        interference = abs(total_amplitude) ** 2

        return interference

    def _initialize_wave_function(self) -> None:
        """Initialize the wave function for the maze"""
        size = np.prod(self.dimensions)
        self.wave_function = np.zeros(size, dtype=complex)

        # Set initial superposition
        for pos in self.player_superposition:
            index = self._position_to_index(pos)
            self.wave_function[index] = 1.0 / np.sqrt(len(self.player_superposition))

    def _create_time_evolution_operator(self, dt: float) -> None:
        """Create the time evolution unitary operator"""
        size = np.prod(self.dimensions)
        # Simplified Hamiltonian for demonstration
        H = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        H = (H + H.conj().T) / 2  # Make Hermitian

        # U = exp(-iHt/ℏ)
        self.time_evolution_unitary = np.exp(-1j * H * dt)

    def _apply_decoherence(self) -> None:
        """Apply environmental decoherence to the wave function"""
        # Random phase decoherence
        phase_noise = np.exp(1j * np.random.normal(0, self.decoherence_rate,
                                                   len(self.wave_function)))
        self.wave_function *= phase_noise

        # Renormalize
        norm = np.linalg.norm(self.wave_function)
        if norm > 0:
            self.wave_function /= norm

    def _calculate_position_probabilities(self) -> np.ndarray:
        """Calculate probability distribution for player positions"""
        probabilities = []
        for pos in self.player_superposition:
            index = self._position_to_index(pos)
            prob = abs(self.wave_function[index]) ** 2
            probabilities.append(prob)

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum()

    def _collapse_wave_function(self, position: tuple[int, int, int]) -> None:
        """Collapse wave function to specific position"""
        self.wave_function.fill(0)
        index = self._position_to_index(position)
        self.wave_function[index] = 1.0
        self.player_superposition = [position]

    def _position_to_index(self, pos: tuple[int, int, int]) -> int:
        """Convert 3D position to 1D index"""
        x, y, z = pos
        return x + y * self.dimensions[0] + z * self.dimensions[0] * self.dimensions[1]

    def _calculate_phase_shift(self, pos1: tuple[int, int, int],
                              pos2: tuple[int, int, int]) -> float:
        """Calculate quantum phase shift between positions"""
        distance = np.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(pos1, pos2, strict=False)))
        return 2 * np.pi * distance / 10  # Arbitrary scaling

    def _find_all_quantum_paths(self) -> list[list[tuple[int, int, int]]]:
        """Find all possible quantum paths through superposition"""
        # Simplified: return some example paths
        if len(self.player_superposition) < 2:
            return [self.player_superposition]

        paths = []
        for i in range(min(10, len(self.player_superposition))):
            path = random.sample(self.player_superposition,
                               min(5, len(self.player_superposition)))
            paths.append(path)
        return paths


class QuantumMazeGenerator:
    """
    Generates quantum mazes with entangled paths and superposition states
    Demonstrates generator functions and complex algorithms
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    @lru_cache(maxsize=32)
    def _calculate_entanglement_strength(distance: float) -> float:
        """Calculate entanglement strength based on distance (cached)"""
        # Bell inequality violation strength decreases with distance
        return np.exp(-distance / 10) * (2 * np.sqrt(2) - 2)

    def generate_quantum_maze(self, width: int, height: int,
                            dimensions: int = 3) -> QuantumMazeState:
        """
        Generate a quantum maze with multiple dimensions
        Uses advanced generation techniques
        """
        maze_state = QuantumMazeState((width, height, dimensions))

        # Generate base maze structure for each dimension
        for z in range(dimensions):
            self._generate_dimension_layer(maze_state, z)

        # Create quantum entanglements between dimensions
        self._create_interdimensional_entanglements(maze_state)

        # Add superposition states to cells
        self._add_superposition_states(maze_state)

        # Set start and goal with quantum properties
        self._set_quantum_objectives(maze_state)

        return maze_state

    def _generate_dimension_layer(self, maze_state: QuantumMazeState, z: int) -> None:
        """Generate a single dimension layer using quantum-inspired algorithm"""
        width, height, _ = maze_state.dimensions

        # Use quantum random walk for maze generation
        for x in range(width):
            for y in range(height):
                cell = QuantumMazeCell(x=x, y=y, z=z)

                # Create walls based on quantum probability
                if self._quantum_wall_probability(x, y, z) > 0.7:
                    cell.is_wall = True

                # Initialize with random qubit state
                random_state = QubitState(
                    alpha=complex(random.random(), random.random()),
                    beta=complex(random.random(), random.random())
                )
                cell.add_qubit_state(random_state)

                maze_state.cells[(x, y, z)] = cell

    def _quantum_wall_probability(self, x: int, y: int, z: int) -> float:
        """Calculate probability of wall using quantum interference pattern"""
        # Create interference pattern
        k1 = 2 * np.pi / 10  # Wave vector 1
        k2 = 2 * np.pi / 7   # Wave vector 2

        # Two-slit interference pattern
        psi1 = np.sin(k1 * x) * np.cos(k1 * y)
        psi2 = np.cos(k2 * x) * np.sin(k2 * y)

        # Quantum interference
        interference = abs(psi1 + psi2 * np.exp(1j * z * np.pi / 4)) ** 2

        return interference / 4  # Normalize to [0, 1]

    def _create_interdimensional_entanglements(self, maze_state: QuantumMazeState) -> None:
        """Create Bell pair entanglements between cells in different dimensions"""
        width, height, dimensions = maze_state.dimensions

        # Create entangled pairs using Bell states
        for _ in range(width * height // 4):  # Entangle 25% of cells
            # Choose random cells from different dimensions
            z1, z2 = random.sample(range(dimensions), 2)
            x, y = random.randint(0, width-1), random.randint(0, height-1)

            cell1 = maze_state.cells.get((x, y, z1))
            cell2 = maze_state.cells.get((x, y, z2))

            if cell1 and cell2:
                # Create Bell state entanglement
                cell1.entangled_with.add((x, y, z2))
                cell2.entangled_with.add((x, y, z1))
                maze_state.entanglement_pairs.add(((x, y, z1), (x, y, z2)))

                # Set entangled qubit states (Bell state |Φ+⟩)
                bell_state1 = QubitState(alpha=1/np.sqrt(2), beta=0)
                bell_state2 = QubitState(alpha=0, beta=1/np.sqrt(2))
                cell1.add_qubit_state(bell_state1)
                cell2.add_qubit_state(bell_state2)

    def _add_superposition_states(self, maze_state: QuantumMazeState) -> None:
        """Add quantum superposition states to cells"""
        for cell in maze_state.cells.values():
            if not cell.is_wall:
                # Add multiple quantum states for superposition
                num_states = random.randint(2, 4)
                for _ in range(num_states - len(cell.states)):
                    # Create random quantum state
                    theta = random.random() * np.pi
                    phi = random.random() * 2 * np.pi

                    alpha = np.cos(theta/2)
                    beta = np.sin(theta/2) * np.exp(1j * phi)

                    state = QubitState(alpha=alpha, beta=beta, phase=phi)
                    cell.add_qubit_state(state)

    def _set_quantum_objectives(self, maze_state: QuantumMazeState) -> None:
        """Set start and goal positions with quantum properties"""
        width, height, dimensions = maze_state.dimensions

        # Set multiple start positions in superposition
        start_positions = [
            (0, 0, z) for z in range(min(3, dimensions))
        ]

        for pos in start_positions:
            if pos in maze_state.cells:
                maze_state.cells[pos].is_start = True
                maze_state.cells[pos].is_wall = False
                maze_state.player_superposition.append(pos)

        # Set goal with quantum tunneling possibility
        goal_pos = (width-1, height-1, random.randint(0, dimensions-1))
        if goal_pos in maze_state.cells:
            maze_state.cells[goal_pos].is_goal = True
            maze_state.cells[goal_pos].is_wall = False

            # Add special quantum state for goal
            goal_state = QubitState(
                alpha=1/np.sqrt(2) + 1j/np.sqrt(2),
                beta=0
            )
            maze_state.cells[goal_pos].add_qubit_state(goal_state)


class QuantumPlayer:
    """
    Player that can exist in quantum superposition
    Demonstrates complex state management and quantum mechanics
    """

    def __init__(self, initial_positions: list[tuple[int, int, int]]):
        self.superposition_positions = initial_positions
        self.collapsed_position: tuple[int, int, int] | None = None
        self.quantum_gates_inventory: dict[QuantumGateType, int] = dict.fromkeys(QuantumGateType, 3)
        self.entanglement_links: set[tuple[int, int, int]] = set()
        self.measurement_count = 0
        self.coherence_time = 100.0
        self.current_coherence = 100.0
        self.quantum_score = 0

    def apply_quantum_gate(self, gate: QuantumGateType,
                          target_cell: QuantumMazeCell) -> bool:
        """Apply a quantum gate to a target cell"""
        if self.quantum_gates_inventory.get(gate, 0) <= 0:
            return False

        # Apply gate to all qubit states in the cell
        new_states = []
        for state in target_cell.states:
            new_state = state.apply_gate(gate)
            new_states.append(new_state)

        target_cell.states = new_states
        self.quantum_gates_inventory[gate] -= 1

        # Applying gates costs coherence
        self.current_coherence -= 5.0

        return True

    def move_in_superposition(self, direction: tuple[int, int, int]) -> None:
        """Move all superposition states simultaneously"""
        new_positions = []

        for pos in self.superposition_positions:
            new_pos = tuple(p + d for p, d in zip(pos, direction, strict=False))
            new_positions.append(new_pos)

        self.superposition_positions = new_positions

        # Movement causes decoherence
        self.current_coherence -= 2.0

    def measure_position(self) -> tuple[int, int, int]:
        """Collapse superposition to single position"""
        if self.collapsed_position:
            return self.collapsed_position

        # Weighted random selection based on quantum amplitudes
        weights = [1.0 / len(self.superposition_positions)] * len(self.superposition_positions)

        # Add bias based on entanglement
        for i, pos in enumerate(self.superposition_positions):
            if pos in self.entanglement_links:
                weights[i] *= 2.0

        # Normalize weights
        total = sum(weights)
        weights = [w/total for w in weights]

        # Collapse
        index = np.random.choice(len(self.superposition_positions), p=weights)
        self.collapsed_position = self.superposition_positions[index]

        self.measurement_count += 1
        self.current_coherence = 0  # Complete decoherence after measurement

        return self.collapsed_position

    def quantum_tunnel(self, target_pos: tuple[int, int, int],
                      barrier_height: float) -> bool:
        """Attempt quantum tunneling through barriers"""
        # Calculate tunneling probability using WKB approximation
        distance = np.sqrt(sum((t - c) ** 2 for t, c in
                             zip(target_pos, self.collapsed_position or (0, 0, 0), strict=False)))

        # Simplified tunneling probability
        tunneling_prob = np.exp(-2 * barrier_height * distance)

        if random.random() < tunneling_prob:
            if self.collapsed_position:
                self.collapsed_position = target_pos
            else:
                self.superposition_positions = [target_pos]

            self.quantum_score += 100  # Bonus for successful tunneling
            return True

        return False

    def entangle_with_cell(self, cell: QuantumMazeCell) -> None:
        """Create quantum entanglement with a cell"""
        position = (cell.x, cell.y, cell.z)
        self.entanglement_links.add(position)
        cell.entangled_with.add(self.collapsed_position or self.superposition_positions[0])

        # Entanglement provides quantum advantage
        self.quantum_score += 50

    def calculate_quantum_score(self) -> int:
        """Calculate score based on quantum mechanics usage"""
        score = self.quantum_score

        # Bonus for maintaining coherence
        score += int(self.current_coherence * 2)

        # Bonus for superposition states
        score += len(self.superposition_positions) * 10

        # Bonus for entanglements
        score += len(self.entanglement_links) * 25

        # Penalty for measurements (encourages quantum strategies)
        score -= self.measurement_count * 5

        return max(0, score)


class QuantumMazeRenderer:
    """
    Renders the quantum maze with visual effects for quantum states
    Demonstrates advanced rendering techniques and visual feedback
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.cell_size = 30
        self.dimension_offset = 250
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Visual effect parameters
        self.wave_animation_phase = 0
        self.entanglement_particles = []
        self.quantum_field_particles = []
        self.superposition_ghosts = []

        # Color schemes for different quantum states
        self.quantum_colors = {
            'superposition': (100, 100, 255, 128),
            'entangled': (255, 100, 255, 200),
            'collapsed': (255, 255, 255, 255),
            'tunneling': (100, 255, 100, 150),
            'interference': (255, 255, 100, 100)
        }

    def render_quantum_maze(self, maze_state: QuantumMazeState,
                           player: QuantumPlayer,
                           current_dimension: int) -> None:
        """Render the complete quantum maze with effects"""
        # Clear screen with quantum field background
        self._draw_quantum_field_background()

        # Draw maze cells for current dimension
        self._draw_maze_layer(maze_state, current_dimension)

        # Draw quantum entanglements
        self._draw_entanglements(maze_state, current_dimension)

        # Draw player in superposition or collapsed state
        self._draw_player(player, current_dimension)

        # Draw wave function visualization
        self._draw_wave_function(maze_state)

        # Draw UI and quantum information
        self._draw_quantum_ui(player, maze_state)

        # Update animation parameters
        self.wave_animation_phase += 0.1
        self._update_particle_effects()

    def _draw_quantum_field_background(self) -> None:
        """Draw animated quantum field background"""
        # Create gradient with quantum fluctuations
        for y in range(self.screen.get_height()):
            # Quantum vacuum fluctuations
            fluctuation = np.sin(self.wave_animation_phase + y * 0.01) * 10
            color_value = int(20 + fluctuation)
            color = (color_value, color_value, color_value + 10)
            pygame.draw.line(self.screen, color, (0, y), (self.screen.get_width(), y))

        # Draw quantum field particles
        for particle in self.quantum_field_particles:
            alpha = int(255 * particle['lifetime'])
            color = (*particle['color'][:3], alpha)

            # Create surface for transparency
            surf = pygame.Surface((particle['size'] * 2, particle['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (particle['size'], particle['size']), particle['size'])

            self.screen.blit(surf, (particle['x'] - particle['size'],
                                   particle['y'] - particle['size']))

    def _draw_maze_layer(self, maze_state: QuantumMazeState, z: int) -> None:
        """Draw a single dimension layer of the maze"""
        width, height, _ = maze_state.dimensions

        # Calculate offset for 3D effect
        offset_x = 50 + z * 20
        offset_y = 50 + z * 20

        for x in range(width):
            for y in range(height):
                cell = maze_state.cells.get((x, y, z))
                if not cell:
                    continue

                # Calculate screen position
                screen_x = offset_x + x * self.cell_size
                screen_y = offset_y + y * self.cell_size

                # Draw cell based on quantum state
                self._draw_quantum_cell(cell, screen_x, screen_y)

    def _draw_quantum_cell(self, cell: QuantumMazeCell, x: int, y: int) -> None:
        """Draw a single cell with quantum state visualization"""
        # Base cell color depends on state
        if cell.is_wall:
            base_color = (50, 50, 50)
        elif cell.is_goal:
            base_color = (100, 255, 100)
        elif cell.is_start:
            base_color = (100, 100, 255)
        else:
            base_color = (100, 100, 100)

        # Draw base cell
        pygame.draw.rect(self.screen, base_color,
                        (x, y, self.cell_size - 2, self.cell_size - 2))

        # Draw superposition states
        if len(cell.states) > 1:
            # Visual representation of superposition
            for i, state in enumerate(cell.states):
                alpha = int(abs(state.alpha) ** 2 * 255)
                overlay = pygame.Surface((self.cell_size - 2, self.cell_size - 2),
                                        pygame.SRCALPHA)

                # Create interference pattern
                pattern_offset = i * (self.cell_size // len(cell.states))
                color = (*self.quantum_colors['superposition'][:3], alpha // 2)

                for line in range(0, self.cell_size - 2, 4):
                    pygame.draw.line(overlay, color,
                                   (pattern_offset, line),
                                   (pattern_offset + 3, line), 2)

                self.screen.blit(overlay, (x, y))

        # Draw entanglement indicator
        if cell.entangled_with:
            pygame.draw.circle(self.screen,
                             self.quantum_colors['entangled'][:3],
                             (x + self.cell_size // 2, y + self.cell_size // 2),
                             5, 1)

        # Draw quantum amplitude as brightness
        if cell.superposition_amplitude != 1.0:
            brightness = int(abs(cell.superposition_amplitude) * 255)
            surf = pygame.Surface((self.cell_size - 2, self.cell_size - 2), pygame.SRCALPHA)
            surf.fill((brightness, brightness, brightness, 50))
            self.screen.blit(surf, (x, y))

    def _draw_entanglements(self, maze_state: QuantumMazeState, current_z: int) -> None:
        """Draw quantum entanglement connections"""
        for (pos1, pos2) in maze_state.entanglement_pairs:
            # Only draw if at least one endpoint is in current dimension
            if pos1[2] != current_z and pos2[2] != current_z:
                continue

            # Calculate screen positions
            offset1_x = 50 + pos1[2] * 20
            offset1_y = 50 + pos1[2] * 20
            screen1_x = offset1_x + pos1[0] * self.cell_size + self.cell_size // 2
            screen1_y = offset1_y + pos1[1] * self.cell_size + self.cell_size // 2

            offset2_x = 50 + pos2[2] * 20
            offset2_y = 50 + pos2[2] * 20
            screen2_x = offset2_x + pos2[0] * self.cell_size + self.cell_size // 2
            screen2_y = offset2_y + pos2[1] * self.cell_size + self.cell_size // 2

            # Draw entanglement as animated wavy line
            self._draw_entanglement_line(screen1_x, screen1_y, screen2_x, screen2_y)

    def _draw_entanglement_line(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Draw animated entanglement connection"""
        num_points = 20

        for i in range(num_points):
            t = i / num_points

            # Linear interpolation with wave perturbation
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t

            # Add quantum fluctuation
            perpendicular_x = -(y2 - y1) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            perpendicular_y = (x2 - x1) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            wave = np.sin(self.wave_animation_phase + t * 10) * 5
            x += perpendicular_x * wave
            y += perpendicular_y * wave

            # Draw point
            color = (
                int(128 + 127 * np.sin(self.wave_animation_phase + t * 5)),
                100,
                int(128 + 127 * np.cos(self.wave_animation_phase + t * 5))
            )

            pygame.draw.circle(self.screen, color, (int(x), int(y)), 2)

    def _draw_player(self, player: QuantumPlayer, current_z: int) -> None:
        """Draw player in superposition or collapsed state"""
        if player.collapsed_position:
            # Draw collapsed player
            if player.collapsed_position[2] == current_z:
                self._draw_collapsed_player(player.collapsed_position)
        else:
            # Draw superposition ghosts
            for pos in player.superposition_positions:
                if pos[2] == current_z:
                    self._draw_superposition_ghost(pos)

    def _draw_collapsed_player(self, position: tuple[int, int, int]) -> None:
        """Draw player in collapsed state"""
        offset_x = 50 + position[2] * 20
        offset_y = 50 + position[2] * 20
        screen_x = offset_x + position[0] * self.cell_size + self.cell_size // 2
        screen_y = offset_y + position[1] * self.cell_size + self.cell_size // 2

        # Draw player as quantum particle
        pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), 8)

        # Draw coherence indicator
        coherence_radius = int(10 + np.sin(self.wave_animation_phase) * 3)
        pygame.draw.circle(self.screen, (100, 200, 255),
                         (screen_x, screen_y), coherence_radius, 2)

    def _draw_superposition_ghost(self, position: tuple[int, int, int]) -> None:
        """Draw ghostly superposition state"""
        offset_x = 50 + position[2] * 20
        offset_y = 50 + position[2] * 20
        screen_x = offset_x + position[0] * self.cell_size + self.cell_size // 2
        screen_y = offset_y + position[1] * self.cell_size + self.cell_size // 2

        # Create transparent surface
        surf = pygame.Surface((20, 20), pygame.SRCALPHA)

        # Draw semi-transparent player
        alpha = int(100 + 50 * np.sin(self.wave_animation_phase))
        pygame.draw.circle(surf, (200, 200, 255, alpha), (10, 10), 6)

        self.screen.blit(surf, (screen_x - 10, screen_y - 10))

    def _draw_wave_function(self, maze_state: QuantumMazeState) -> None:
        """Visualize the wave function at bottom of screen"""
        if maze_state.wave_function is None:
            return

        # Draw wave function amplitude
        wave_height = 100
        wave_y_base = self.screen.get_height() - wave_height - 50

        # Sample wave function for visualization
        num_samples = min(200, len(maze_state.wave_function))

        points = []
        for i in range(num_samples):
            x = int(i * self.screen.get_width() / num_samples)

            # Get amplitude and phase
            amplitude = abs(maze_state.wave_function[i])
            phase = np.angle(maze_state.wave_function[i])

            # Convert to y coordinate
            y = wave_y_base - int(amplitude * wave_height * 10)

            points.append((x, y))

            # Draw phase as color
            phase_color = (
                int(128 + 127 * np.cos(phase)),
                int(128 + 127 * np.sin(phase)),
                128
            )

            pygame.draw.circle(self.screen, phase_color, (x, y), 2)

        # Draw connecting lines
        if len(points) > 1:
            pygame.draw.lines(self.screen, (100, 200, 100), False, points, 1)

    def _draw_quantum_ui(self, player: QuantumPlayer, maze_state: QuantumMazeState) -> None:
        """Draw UI elements showing quantum information"""
        ui_x = self.screen.get_width() - 250
        ui_y = 20

        # Background panel
        panel = pygame.Surface((230, 400), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        self.screen.blit(panel, (ui_x - 10, ui_y - 10))

        # Title
        title = self.font_large.render("QUANTUM STATE", True, (100, 200, 255))
        self.screen.blit(title, (ui_x, ui_y))

        ui_y += 50

        # Quantum Score
        score_text = self.font_medium.render(
            f"Q-Score: {player.calculate_quantum_score()}",
            True, (255, 255, 255)
        )
        self.screen.blit(score_text, (ui_x, ui_y))

        ui_y += 30

        # Coherence meter
        coherence_text = self.font_small.render("Coherence:", True, (255, 255, 255))
        self.screen.blit(coherence_text, (ui_x, ui_y))

        ui_y += 20

        # Draw coherence bar
        bar_width = 200
        bar_height = 20
        coherence_ratio = player.current_coherence / player.coherence_time

        pygame.draw.rect(self.screen, (50, 50, 50),
                        (ui_x, ui_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (100, 200, 255),
                        (ui_x, ui_y, int(bar_width * coherence_ratio), bar_height))

        ui_y += 30

        # Superposition states
        super_text = self.font_small.render(
            f"Superposition: {len(player.superposition_positions)} states",
            True, (255, 255, 255)
        )
        self.screen.blit(super_text, (ui_x, ui_y))

        ui_y += 25

        # Entanglements
        entangle_text = self.font_small.render(
            f"Entanglements: {len(player.entanglement_links)}",
            True, (255, 100, 255)
        )
        self.screen.blit(entangle_text, (ui_x, ui_y))

        ui_y += 25

        # Measurements
        measure_text = self.font_small.render(
            f"Measurements: {player.measurement_count}",
            True, (255, 255, 100)
        )
        self.screen.blit(measure_text, (ui_x, ui_y))

        ui_y += 35

        # Quantum gates inventory
        gates_title = self.font_medium.render("QUANTUM GATES", True, (200, 200, 100))
        self.screen.blit(gates_title, (ui_x, ui_y))

        ui_y += 30

        for gate, count in player.quantum_gates_inventory.items():
            gate_text = self.font_small.render(
                f"{gate.name}: {count}",
                True, (200, 200, 200)
            )
            self.screen.blit(gate_text, (ui_x, ui_y))
            ui_y += 20

    def _update_particle_effects(self) -> None:
        """Update visual particle effects"""
        # Update quantum field particles
        for particle in self.quantum_field_particles[:]:
            particle['lifetime'] -= 0.02
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']

            if particle['lifetime'] <= 0:
                self.quantum_field_particles.remove(particle)

        # Add new particles occasionally
        if random.random() < 0.1:
            self.quantum_field_particles.append({
                'x': random.randint(0, self.screen.get_width()),
                'y': random.randint(0, self.screen.get_height()),
                'vx': random.uniform(-1, 1),
                'vy': random.uniform(-1, 1),
                'size': random.randint(1, 3),
                'color': (100, 100, 200),
                'lifetime': 1.0
            })


class QuantumMazeNavigator:
    """
    Main game class managing quantum maze navigation
    Demonstrates integration of all components and game loop
    """

    def __init__(self):
        # Initialize display
        self.screen_width = 1280
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Quantum Maze Navigator - Navigate Reality in Superposition")

        # Initialize game components
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

        # Quantum maze components
        self.maze_generator = QuantumMazeGenerator(seed=None)
        self.current_level = 1
        self.current_dimension = 0

        # Generate initial maze
        self.maze_state: QuantumMazeState = self._generate_level_maze()

        # Initialize player
        initial_positions = [(0, 0, z) for z in range(min(3, self.maze_state.dimensions[2]))]
        self.player = QuantumPlayer(initial_positions)

        # Initialize renderer
        self.renderer = QuantumMazeRenderer(self.screen)

        # Game state
        self.selected_gate = QuantumGateType.HADAMARD
        self.measurement_cooldown = 0
        self.victory = False

        # Tutorial state
        self.show_tutorial = True
        self.tutorial_page = 0

        # Async event loop for quantum calculations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def _generate_level_maze(self) -> QuantumMazeState:
        """Generate maze for current level"""
        # Increase complexity with level
        width = 10 + self.current_level * 2
        height = 10 + self.current_level * 2
        dimensions = min(3 + self.current_level // 3, 5)

        return self.maze_generator.generate_quantum_maze(width, height, dimensions)

    def run(self) -> None:
        """Main game loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS

            self._handle_events()

            if not self.paused and not self.victory:
                self._update(dt)

            self._render()

            # Handle async quantum calculations
            self._process_quantum_calculations()

        pygame.quit()
        self.loop.close()

    def _handle_events(self) -> None:
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_TAB:
                    self.show_tutorial = not self.show_tutorial

                # Movement in superposition (arrow keys)
                elif event.key == pygame.K_UP:
                    self._move_player((0, -1, 0))
                elif event.key == pygame.K_DOWN:
                    self._move_player((0, 1, 0))
                elif event.key == pygame.K_LEFT:
                    self._move_player((-1, 0, 0))
                elif event.key == pygame.K_RIGHT:
                    self._move_player((1, 0, 0))

                # Dimension navigation
                elif event.key == pygame.K_PAGEUP:
                    self.current_dimension = max(0, self.current_dimension - 1)
                elif event.key == pygame.K_PAGEDOWN:
                    max_dim = self.maze_state.dimensions[2] - 1
                    self.current_dimension = min(max_dim, self.current_dimension + 1)

                # Quantum gate selection (number keys)
                elif event.key >= pygame.K_1 and event.key <= pygame.K_8:
                    gate_index = event.key - pygame.K_1
                    gates = list(QuantumGateType)
                    if gate_index < len(gates):
                        self.selected_gate = gates[gate_index]

                # Measure position (collapse superposition)
                elif event.key == pygame.K_m:
                    if self.measurement_cooldown <= 0:
                        self._measure_player_position()
                        self.measurement_cooldown = 2.0  # 2 second cooldown

                # Quantum tunneling attempt
                elif event.key == pygame.K_t:
                    self._attempt_quantum_tunnel()

                # Apply quantum gate
                elif event.key == pygame.K_g:
                    self._apply_gate_to_cell()

                # Create entanglement
                elif event.key == pygame.K_e:
                    self._create_entanglement()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_cell_click(event.pos)

    def _move_player(self, direction: tuple[int, int, int]) -> None:
        """Move player in superposition"""
        # Check if movement is valid for all superposition states
        new_positions = []

        for pos in self.player.superposition_positions:
            new_pos = tuple(p + d for p, d in zip(pos, direction, strict=False))

            # Check bounds
            if (0 <= new_pos[0] < self.maze_state.dimensions[0] and
                0 <= new_pos[1] < self.maze_state.dimensions[1] and
                0 <= new_pos[2] < self.maze_state.dimensions[2]):

                cell = self.maze_state.cells.get(new_pos)
                if cell and not cell.is_wall:
                    new_positions.append(new_pos)
                else:
                    new_positions.append(pos)  # Stay in place if blocked
            else:
                new_positions.append(pos)

        self.player.superposition_positions = new_positions

        # Check for goal
        self._check_goal_reached()

    def _measure_player_position(self) -> None:
        """Collapse player's superposition"""
        collapsed_pos = self.player.measure_position()

        # Apply measurement to maze state
        self.maze_state.player_superposition = [collapsed_pos]

    def _attempt_quantum_tunnel(self) -> None:
        """Try to tunnel through barriers"""
        if not self.player.collapsed_position:
            return

        # Find nearest wall
        x, y, z = self.player.collapsed_position

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            wall_pos = (x + dx, y + dy, z)
            target_pos = (x + 2*dx, y + 2*dy, z)

            wall_cell = self.maze_state.cells.get(wall_pos)
            target_cell = self.maze_state.cells.get(target_pos)

            if wall_cell and wall_cell.is_wall and target_cell and not target_cell.is_wall:
                # Calculate barrier height based on wall properties
                barrier_height = len(wall_cell.states) * 0.5

                if self.player.quantum_tunnel(target_pos, barrier_height):
                    break

    def _apply_gate_to_cell(self) -> None:
        """Apply selected quantum gate to current cell"""
        if self.player.collapsed_position:
            pos = self.player.collapsed_position
        else:
            pos = self.player.superposition_positions[0]

        cell = self.maze_state.cells.get(pos)
        if cell:
            self.player.apply_quantum_gate(self.selected_gate, cell)

    def _create_entanglement(self) -> None:
        """Create entanglement with current cell"""
        if self.player.collapsed_position:
            pos = self.player.collapsed_position
        else:
            pos = self.player.superposition_positions[0]

        cell = self.maze_state.cells.get(pos)
        if cell:
            self.player.entangle_with_cell(cell)

    def _handle_cell_click(self, mouse_pos: tuple[int, int]) -> None:
        """Handle clicking on maze cells"""
        # Convert mouse position to cell coordinates
        # (Simplified - would need proper coordinate transformation)
        cell_x = (mouse_pos[0] - 50) // self.renderer.cell_size
        cell_y = (mouse_pos[1] - 50) // self.renderer.cell_size

        cell_pos = (cell_x, cell_y, self.current_dimension)
        cell = self.maze_state.cells.get(cell_pos)

        if cell:
            # Apply quantum gate to clicked cell
            self.player.apply_quantum_gate(self.selected_gate, cell)

    def _check_goal_reached(self) -> None:
        """Check if player reached goal"""
        for pos in self.player.superposition_positions:
            cell = self.maze_state.cells.get(pos)
            if cell and cell.is_goal:
                self.victory = True
                self._handle_victory()
                break

    def _handle_victory(self) -> None:
        """Handle level completion"""
        # Calculate final score
        final_score = self.player.calculate_quantum_score()

        # Bonus for maintaining superposition
        if not self.player.collapsed_position:
            final_score += 500

        print(f"Level {self.current_level} completed! Score: {final_score}")

        # Advance to next level
        self.current_level += 1
        self.maze_state = self._generate_level_maze()

        # Reset player
        initial_positions = [(0, 0, z) for z in range(min(3, self.maze_state.dimensions[2]))]
        self.player = QuantumPlayer(initial_positions)

        self.victory = False

    def _update(self, dt: float) -> None:
        """Update game state"""
        # Update maze quantum state
        self.maze_state.evolve(dt)

        # Update player coherence
        if self.player.current_coherence > 0:
            self.player.current_coherence -= dt * 10  # Decoherence over time
            self.player.current_coherence = max(0, self.player.current_coherence)

        # Update cooldowns
        if self.measurement_cooldown > 0:
            self.measurement_cooldown -= dt

        # Random quantum events
        if random.random() < 0.01:  # 1% chance per frame
            self._trigger_quantum_event()

    def _trigger_quantum_event(self) -> None:
        """Trigger random quantum mechanical events"""
        event_type = random.choice(['decoherence', 'entanglement', 'interference'])

        if event_type == 'decoherence':
            # Random decoherence in maze
            self.maze_state.decoherence_rate *= 1.1

        elif event_type == 'entanglement':
            # Create random entanglement
            positions = list(self.maze_state.cells.keys())
            if len(positions) >= 2:
                pos1, pos2 = random.sample(positions, 2)
                self.maze_state.entanglement_pairs.add((pos1, pos2))

        elif event_type == 'interference':
            # Quantum interference affects movement
            if self.player.superposition_positions:
                # Add slight random phase to positions
                for i in range(len(self.player.superposition_positions)):
                    if random.random() < 0.3:
                        pos = list(self.player.superposition_positions[i])
                        pos[random.randint(0, 2)] += random.choice([-1, 1])
                        self.player.superposition_positions[i] = tuple(pos)

    def _render(self) -> None:
        """Render the game"""
        # Clear screen
        self.screen.fill((0, 0, 0))

        # Render quantum maze
        self.renderer.render_quantum_maze(
            self.maze_state,
            self.player,
            self.current_dimension
        )

        # Render tutorial if active
        if self.show_tutorial:
            self._render_tutorial()

        # Render victory screen if won
        if self.victory:
            self._render_victory()

        pygame.display.flip()

    def _render_tutorial(self) -> None:
        """Render tutorial overlay"""
        # Semi-transparent background
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))

        # Tutorial content
        font = pygame.font.Font(None, 24)
        title_font = pygame.font.Font(None, 36)

        title = title_font.render("QUANTUM MAZE NAVIGATOR", True, (100, 200, 255))
        title_rect = title.get_rect(center=(self.screen_width // 2, 50))
        self.screen.blit(title, title_rect)

        tutorial_texts = [
            "",
            "Navigate through quantum mazes existing in superposition!",
            "",
            "QUANTUM CONCEPTS:",
            "- You exist in SUPERPOSITION (multiple positions at once)",
            "- Cells can be ENTANGLED across dimensions",
            "- Measurement COLLAPSES your superposition",
            "- Use QUANTUM TUNNELING to pass through walls",
            "",
            "CONTROLS:",
            "Arrow Keys - Move in superposition",
            "M - Measure (collapse) your position",
            "G - Apply quantum gate to current cell",
            "E - Create entanglement",
            "T - Attempt quantum tunneling",
            "PageUp/Down - Switch dimensions",
            "1-8 - Select quantum gates",
            "",
            "GOAL: Reach the green goal cell using quantum mechanics!",
            "Maintain superposition for bonus points!",
            "",
            "Press TAB to close tutorial"
        ]

        y = 100
        for text in tutorial_texts:
            if text:
                rendered = font.render(text, True, (255, 255, 255))
                text_rect = rendered.get_rect(center=(self.screen_width // 2, y))
                self.screen.blit(rendered, text_rect)
            y += 30

    def _render_victory(self) -> None:
        """Render victory screen"""
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        font = pygame.font.Font(None, 48)
        text = font.render(f"QUANTUM LEVEL {self.current_level - 1} COMPLETE!",
                          True, (100, 255, 100))
        text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text, text_rect)

    def _process_quantum_calculations(self) -> None:
        """Process async quantum calculations"""
        # Run pending async tasks
        try:
            # Use walrus operator for efficient checking (Python 3.8+)
            if pending := asyncio.all_tasks(self.loop):
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except:
            pass  # Handle async exceptions gracefully


# Context manager for game resources
@contextmanager
def game_context():
    """Context manager for proper game initialization and cleanup"""
    pygame.init()
    pygame.font.init()

    try:
        yield
    finally:
        pygame.quit()


# Main execution
def main():
    """Main entry point demonstrating context manager usage"""
    with game_context():
        game = QuantumMazeNavigator()
        game.run()


if __name__ == "__main__":
    main()
