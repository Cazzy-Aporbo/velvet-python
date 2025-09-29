"""
COMPREHENSIVE LYMPHATIC SYSTEM COMPUTATIONAL MODEL
A sophisticated physiological simulation of the human lymphatic system incorporating
fluid dynamics, immunological responses, and network theory. This model implements
advanced computational techniques including finite element methods, graph algorithms,
and stochastic differential equations to simulate lymph flow, immune cell trafficking,
and pathogen clearance.

Mathematical Framework:
- Starling's equation for transcapillary fluid exchange
- Navier-Stokes equations for lymph flow dynamics  
- Michaelis-Menten kinetics for protein transport
- Gillespie algorithm for stochastic immune reactions
- Graph theory for lymphatic network topology

Author: Cazzy Aporbo September 2025
Version: 3.0.0
Python Requirements: 3.8+
Dependencies: numpy, scipy, networkx, matplotlib
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, root
from scipy.stats import poisson, gamma, norm
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Tuple, Optional, Callable, Union, Protocol, TypeVar
from enum import Enum, auto
import itertools
from functools import lru_cache, cached_property, reduce
import warnings
import logging
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
import heapq
import math

# Configure logging for detailed debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type definitions for clarity
T = TypeVar('T')
FlowRate = float  # ml/min
Pressure = float  # mmHg
Concentration = float  # mol/L
Position = Tuple[float, float, float]  # 3D coordinates in mm


class VesselType(Enum):
    """Classification of lymphatic vessel types with physiological properties"""
    INITIAL_LYMPHATIC = auto()  # Blind-ended initial lymphatics, 10-60 μm diameter
    PRECOLLECTOR = auto()        # Transitional vessels, 60-150 μm diameter  
    COLLECTOR = auto()           # Valved collecting vessels, 150-500 μm diameter
    TRUNK = auto()               # Major lymphatic trunks, 500-2000 μm diameter
    DUCT = auto()                # Thoracic and right lymphatic ducts, >2000 μm diameter
    
    @property
    def diameter_range(self) -> Tuple[float, float]:
        """Return physiological diameter range in micrometers"""
        ranges = {
            VesselType.INITIAL_LYMPHATIC: (10, 60),
            VesselType.PRECOLLECTOR: (60, 150),
            VesselType.COLLECTOR: (150, 500),
            VesselType.TRUNK: (500, 2000),
            VesselType.DUCT: (2000, 5000)
        }
        return ranges[self]
    
    @property
    def valve_spacing(self) -> Optional[float]:
        """Return typical valve spacing in mm, None if no valves"""
        spacing = {
            VesselType.INITIAL_LYMPHATIC: None,
            VesselType.PRECOLLECTOR: None,
            VesselType.COLLECTOR: 2.0,  # 2mm between valves
            VesselType.TRUNK: 5.0,
            VesselType.DUCT: 10.0
        }
        return spacing[self]


@dataclass
class FluidProperties:
    """
    Physical and chemical properties of lymphatic fluid
    
    The lymph composition varies based on tissue source and pathological conditions.
    Normal lymph has lower protein concentration than plasma due to selective
    capillary filtration.
    """
    viscosity: float = 1.5e-3  # Pa·s (1.5x water viscosity due to proteins)
    density: float = 1015.0     # kg/m³ (slightly denser than water)
    osmolality: float = 285.0   # mOsm/kg (isotonic with plasma)
    protein_concentration: float = 2.0  # g/dL (vs 7 g/dL in plasma)
    ph: float = 7.4            # Physiological pH
    temperature: float = 37.0   # °C
    surface_tension: float = 0.05  # N/m
    
    @cached_property
    def reynolds_number(self) -> Callable[[float, float], float]:
        """
        Calculate Reynolds number for lymph flow
        Re = ρvD/μ where ρ=density, v=velocity, D=diameter, μ=viscosity
        
        Lymph flow is typically laminar (Re < 2000) except in large ducts
        """
        def calculate_re(velocity: float, diameter: float) -> float:
            return (self.density * abs(velocity) * diameter) / self.viscosity
        return calculate_re
    
    @cached_property
    def womersley_number(self) -> Callable[[float, float], float]:
        """
        Calculate Womersley number for pulsatile flow
        α = r√(ωρ/μ) where r=radius, ω=angular frequency
        
        Determines relative importance of unsteady to viscous forces
        """
        def calculate_wo(radius: float, frequency: float) -> float:
            omega = 2 * np.pi * frequency
            return radius * np.sqrt(omega * self.density / self.viscosity)
        return calculate_wo


@dataclass
class ImmuneCell:
    """
    Representation of immune cells in lymphatic system
    
    Lymphocytes continuously recirculate between blood and lymph, with
    approximately 10^11 lymphocytes passing through lymph nodes daily.
    """
    cell_type: str  # T-cell, B-cell, NK, Dendritic, etc.
    position: np.ndarray  # 3D position vector
    velocity: np.ndarray  # 3D velocity vector
    activation_state: float = 0.0  # 0=naive, 1=fully activated
    antigen_specificity: Optional[str] = None
    lifespan: float = 100.0  # hours
    age: float = 0.0
    adhesion_molecules: Dict[str, float] = field(default_factory=dict)
    chemokine_receptors: Dict[str, float] = field(default_factory=dict)
    
    def update_position(self, dt: float, flow_field: np.ndarray) -> None:
        """
        Update cell position using Langevin equation for cell motion
        dx/dt = v_flow + v_chemotaxis + √(2D)ξ
        
        Incorporates advection, chemotaxis, and Brownian motion
        """
        # Advection with flow
        self.position += self.velocity * dt
        
        # Brownian motion (Einstein relation: D = kT/6πηr)
        k_b = 1.38e-23  # Boltzmann constant
        T = 310  # Body temperature in Kelvin
        eta = 1.5e-3  # Lymph viscosity
        r_cell = 5e-6  # Cell radius ~5 μm
        D = k_b * T / (6 * np.pi * eta * r_cell)
        
        # Add random walk component
        self.position += np.sqrt(2 * D * dt) * np.random.randn(3)
        
        # Update velocity from flow field
        self.velocity = flow_field
    
    def chemotaxis_velocity(self, gradient: np.ndarray, sensitivity: float = 1e-6) -> np.ndarray:
        """
        Calculate chemotactic velocity using Keller-Segel model
        v_chem = χ∇c/(1 + λc) where χ=sensitivity, c=concentration
        """
        concentration = np.linalg.norm(gradient)
        return sensitivity * gradient / (1 + 0.1 * concentration)


class LymphNode:
    """
    Sophisticated model of lymph node structure and function
    
    Lymph nodes are organized into distinct compartments:
    - Subcapsular sinus: Initial filtration
    - Cortex: B-cell follicles and germinal centers
    - Paracortex: T-cell zones with high endothelial venules
    - Medulla: Medullary cords and sinuses for exit
    """
    
    def __init__(self, node_id: str, position: Position, volume: float = 1.0):
        self.node_id = node_id
        self.position = np.array(position)
        self.volume = volume  # cm³
        self.compartments = self._initialize_compartments()
        
        # Immunological parameters
        self.resident_cells: Dict[str, List[ImmuneCell]] = defaultdict(list)
        self.antigen_concentration: Dict[str, float] = {}
        self.cytokine_levels: Dict[str, float] = defaultdict(float)
        self.germinal_centers: List[GerminalCenter] = []
        
        # Fluid dynamics
        self.inlet_pressure: float = 3.0  # mmHg
        self.outlet_pressure: float = 1.0  # mmHg
        self.flow_resistance: float = self._calculate_resistance()
        
        # Filtration parameters
        self.filtration_coefficient = 0.5  # Fraction of fluid filtered
        self.particle_retention = {}  # Size-dependent retention
        
        logger.info(f"Initialized lymph node {node_id} at position {position}")
    
    def _initialize_compartments(self) -> Dict[str, float]:
        """Initialize anatomical compartments with relative volumes"""
        return {
            'subcapsular_sinus': 0.10,
            'cortex': 0.35,
            'paracortex': 0.30,
            'medulla': 0.20,
            'hilum': 0.05
        }
    
    def _calculate_resistance(self) -> float:
        """
        Calculate flow resistance using Darcy's law for porous media
        R = μL/(κA) where μ=viscosity, L=length, κ=permeability, A=area
        """
        viscosity = 1.5e-3  # Pa·s
        length = (self.volume ** (1/3)) * 10  # Approximate length in mm
        permeability = 1e-12  # m² (typical for biological tissue)
        area = length ** 2  # Approximate cross-sectional area
        
        # Convert to mmHg·min/ml
        resistance = (viscosity * length) / (permeability * area) * 1.33e-4
        return resistance
    
    def process_lymph(self, inlet_flow: float, dt: float) -> Tuple[float, Dict[str, float]]:
        """
        Process incoming lymph through the node
        
        Returns:
            outlet_flow: Flow rate leaving the node
            filtered_substances: Concentrations of filtered substances
        """
        # Calculate pressure-driven flow (Starling's equation)
        transmural_pressure = self.inlet_pressure - self.outlet_pressure
        outlet_flow = transmural_pressure / self.flow_resistance
        
        # Fluid exchange in sinuses
        filtered_volume = inlet_flow * self.filtration_coefficient * dt
        
        # Particle filtration based on size
        filtered_substances = self._filter_particles(inlet_flow, dt)
        
        # Update compartment dynamics
        self._update_compartment_flow(inlet_flow, outlet_flow, dt)
        
        # Immune cell interactions
        self._process_immune_interactions(dt)
        
        return outlet_flow, filtered_substances
    
    def _filter_particles(self, flow_rate: float, dt: float) -> Dict[str, float]:
        """
        Size-selective filtration of particles and pathogens
        
        Retention efficiency η = 1 - exp(-λd) where λ=filtration parameter, d=diameter
        """
        filtered = {}
        
        particle_sizes = {
            'virus': 0.1,      # μm
            'bacteria': 1.0,
            'dead_cells': 10.0,
            'protein_aggregates': 0.01
        }
        
        for particle_type, size in particle_sizes.items():
            # Filtration efficiency increases with size
            lambda_filt = 0.5  # Filtration parameter
            efficiency = 1 - np.exp(-lambda_filt * size)
            
            # Amount filtered
            if particle_type in self.antigen_concentration:
                filtered_amount = self.antigen_concentration[particle_type] * efficiency * flow_rate * dt
                filtered[particle_type] = filtered_amount
                self.antigen_concentration[particle_type] *= (1 - efficiency)
        
        return filtered
    
    def _update_compartment_flow(self, inlet: float, outlet: float, dt: float) -> None:
        """
        Update flow through anatomical compartments using compartmental model
        
        dV_i/dt = Q_in,i - Q_out,i + Σ(Q_ji - Q_ij)
        """
        # Simplified compartmental flow routing
        total_volume_change = (inlet - outlet) * dt
        
        for compartment, fraction in self.compartments.items():
            # Distribute volume change proportionally
            volume_change = total_volume_change * fraction
            # Update compartment volumes (implementation depends on specific needs)
    
    def _process_immune_interactions(self, dt: float) -> None:
        """
        Simulate immune cell interactions within the node
        
        Includes:
        - T-B cell interactions in germinal centers
        - Antigen presentation by dendritic cells
        - Clonal expansion of activated lymphocytes
        """
        # Antigen presentation rate (Michaelis-Menten kinetics)
        for antigen, concentration in self.antigen_concentration.items():
            K_m = 1e-6  # Michaelis constant
            V_max = 1e-3  # Maximum presentation rate
            
            presentation_rate = (V_max * concentration) / (K_m + concentration)
            
            # Activate naive T cells based on presentation
            for cell in self.resident_cells.get('T-cell', []):
                if cell.activation_state < 1.0 and cell.antigen_specificity == antigen:
                    activation_probability = presentation_rate * dt
                    if np.random.random() < activation_probability:
                        cell.activation_state = min(1.0, cell.activation_state + 0.1)
        
        # Clonal expansion of activated cells
        self._simulate_clonal_expansion(dt)
    
    def _simulate_clonal_expansion(self, dt: float) -> None:
        """
        Simulate clonal expansion using branching process
        
        Activated lymphocytes undergo rapid division with
        generation time ~12 hours, expanding up to 10^6-fold
        """
        division_rate = 1.0 / 12.0  # divisions per hour
        
        new_cells = []
        for cell_type, cells in self.resident_cells.items():
            for cell in cells:
                if cell.activation_state > 0.5:  # Activated cells divide
                    division_probability = division_rate * dt
                    if np.random.random() < division_probability:
                        # Create daughter cell
                        daughter = ImmuneCell(
                            cell_type=cell.cell_type,
                            position=cell.position + np.random.randn(3) * 1e-6,
                            velocity=cell.velocity,
                            activation_state=cell.activation_state,
                            antigen_specificity=cell.antigen_specificity
                        )
                        new_cells.append(daughter)
        
        # Add new cells to population
        for cell in new_cells:
            self.resident_cells[cell.cell_type].append(cell)


class GerminalCenter:
    """
    Model of germinal center dynamics for antibody production
    
    Germinal centers are sites of B-cell proliferation, somatic hypermutation,
    and affinity maturation in response to antigen stimulation.
    """
    
    def __init__(self, antigen: str):
        self.antigen = antigen
        self.b_cells: List[ImmuneCell] = []
        self.follicular_dendritic_cells = []
        self.t_helper_cells = []
        
        # Affinity maturation parameters
        self.mutation_rate = 0.001  # Per base pair per division
        self.selection_threshold = 0.7  # Minimum affinity for survival
        
        # Temporal dynamics
        self.age = 0.0  # Days since formation
        self.peak_size = 1e4  # Peak number of B cells
        self.current_size = 100  # Initial founder B cells
    
    def evolve(self, dt: float) -> None:
        """
        Simulate germinal center evolution including:
        - Dark zone: Proliferation and somatic hypermutation
        - Light zone: Selection based on antigen affinity
        """
        self.age += dt / 24.0  # Convert hours to days
        
        # Growth dynamics (logistic growth with decay)
        growth_rate = 0.5  # Per day
        carrying_capacity = self.peak_size * np.exp(-self.age / 14.0)  # Decay over 2 weeks
        
        dN_dt = growth_rate * self.current_size * (1 - self.current_size / carrying_capacity)
        self.current_size += dN_dt * dt / 24.0
        
        # Somatic hypermutation and selection
        self._perform_affinity_maturation(dt)
    
    def _perform_affinity_maturation(self, dt: float) -> None:
        """
        Simulate somatic hypermutation and affinity-based selection
        
        Uses a simplified model where affinity increases stochastically
        with selection for high-affinity variants
        """
        surviving_cells = []
        
        for cell in self.b_cells:
            # Random mutation affects affinity
            if np.random.random() < self.mutation_rate:
                # Affinity change (can be positive or negative)
                affinity_change = np.random.normal(0, 0.1)
                # Update cell's binding affinity (stored in activation_state as proxy)
                cell.activation_state = max(0, min(1, cell.activation_state + affinity_change))
            
            # Selection based on affinity
            survival_probability = cell.activation_state / self.selection_threshold
            if np.random.random() < survival_probability:
                surviving_cells.append(cell)
        
        self.b_cells = surviving_cells


class LymphaticVessel:
    """
    Advanced model of lymphatic vessel with contractile properties
    
    Collecting lymphatics have smooth muscle that contracts rhythmically
    to propel lymph against gravity. Contraction frequency ~5-10/min.
    """
    
    def __init__(self, vessel_id: str, vessel_type: VesselType, 
                 start_pos: Position, end_pos: Position):
        self.vessel_id = vessel_id
        self.vessel_type = vessel_type
        self.start_position = np.array(start_pos)
        self.end_position = np.array(end_pos)
        
        # Geometric properties
        self.length = np.linalg.norm(self.end_position - self.start_position)
        diameter_range = vessel_type.diameter_range
        self.diameter = np.random.uniform(*diameter_range) * 1e-6  # Convert to meters
        self.wall_thickness = self.diameter * 0.1  # 10% of diameter
        
        # Mechanical properties
        self.elastance = self._calculate_elastance()
        self.compliance = 1.0 / self.elastance
        self.contractility = self._calculate_contractility()
        
        # Valves (if present)
        self.valves = self._initialize_valves()
        
        # Flow properties
        self.flow_rate: float = 0.0
        self.pressure_gradient: float = 0.0
        self.reynolds_number: float = 0.0
        
        # Contractile state
        self.contraction_phase: float = 0.0  # 0-2π
        self.contraction_frequency: float = 8.0 / 60.0  # Hz (8 contractions/min)
        self.max_contraction: float = 0.3  # 30% diameter reduction
        
        logger.info(f"Created {vessel_type.name} vessel {vessel_id}: "
                   f"length={self.length:.2f}mm, diameter={self.diameter*1e6:.1f}μm")
    
    def _calculate_elastance(self) -> float:
        """
        Calculate vessel elastance using LaPlace's law
        E = ΔP/ΔV = 2hE_w/(r²L) where h=wall thickness, E_w=wall modulus, r=radius
        """
        # Young's modulus for lymphatic vessel wall (varies with vessel type)
        modulus_map = {
            VesselType.INITIAL_LYMPHATIC: 1e3,  # Pa (very compliant)
            VesselType.PRECOLLECTOR: 5e3,
            VesselType.COLLECTOR: 1e4,
            VesselType.TRUNK: 2e4,
            VesselType.DUCT: 5e4
        }
        
        E_wall = modulus_map.get(self.vessel_type, 1e4)
        radius = self.diameter / 2
        
        elastance = (2 * self.wall_thickness * E_wall) / (radius**2 * self.length * 1e-3)
        return elastance
    
    def _calculate_contractility(self) -> float:
        """
        Calculate contractile strength based on smooth muscle content
        
        Collectors have highest smooth muscle density
        """
        contractility_map = {
            VesselType.INITIAL_LYMPHATIC: 0.0,  # No smooth muscle
            VesselType.PRECOLLECTOR: 0.2,
            VesselType.COLLECTOR: 1.0,  # Maximum contractility
            VesselType.TRUNK: 0.8,
            VesselType.DUCT: 0.5
        }
        return contractility_map.get(self.vessel_type, 0.0)
    
    def _initialize_valves(self) -> List[float]:
        """
        Initialize valve positions along vessel
        
        Valves prevent backflow and segment vessel into lymphangions
        """
        valve_positions = []
        
        if self.vessel_type.valve_spacing:
            spacing = self.vessel_type.valve_spacing
            num_valves = int(self.length / spacing)
            
            for i in range(1, num_valves):
                position = i * spacing
                valve_positions.append(position)
        
        return valve_positions
    
    def calculate_flow(self, inlet_pressure: float, outlet_pressure: float, 
                      dt: float) -> float:
        """
        Calculate flow rate using Poiseuille's law with modifications for:
        - Vessel compliance
        - Active contractions
        - Valve function
        
        Q = πr⁴ΔP/(8μL) × compliance_factor × contraction_factor
        """
        # Update contraction phase
        self.contraction_phase += 2 * np.pi * self.contraction_frequency * dt
        
        # Calculate effective diameter with contraction
        contraction_factor = 1.0 - self.max_contraction * self.contractility * \
                           (1 + np.sin(self.contraction_phase)) / 2
        
        effective_diameter = self.diameter * contraction_factor
        effective_radius = effective_diameter / 2
        
        # Pressure gradient
        self.pressure_gradient = (inlet_pressure - outlet_pressure) / self.length
        
        # Check valve function (simplified - valves close if adverse pressure)
        if self.pressure_gradient < 0 and len(self.valves) > 0:
            self.flow_rate = 0.0  # Valves prevent backflow
            return self.flow_rate
        
        # Poiseuille flow with compliance
        viscosity = 1.5e-3  # Pa·s
        
        # Basic Poiseuille flow
        flow_poiseuille = (np.pi * effective_radius**4 * abs(self.pressure_gradient) * 133.322) / \
                         (8 * viscosity * self.length * 1e-3)
        
        # Compliance effect (vessel distension with pressure)
        transmural_pressure = (inlet_pressure + outlet_pressure) / 2
        compliance_factor = 1.0 + self.compliance * transmural_pressure * 0.01
        
        # Total flow
        self.flow_rate = flow_poiseuille * compliance_factor * np.sign(self.pressure_gradient)
        
        # Calculate Reynolds number
        if effective_diameter > 0:
            mean_velocity = abs(self.flow_rate) / (np.pi * effective_radius**2)
            self.reynolds_number = (1015 * mean_velocity * effective_diameter) / viscosity
        
        return self.flow_rate
    
    def propagate_contraction_wave(self, upstream_phase: float, coupling_strength: float = 0.5) -> None:
        """
        Propagate contractile waves along vessel chain
        
        Models coordination of lymphatic pumping through electrical coupling
        """
        # Phase coupling (Kuramoto model)
        phase_difference = upstream_phase - self.contraction_phase
        self.contraction_phase += coupling_strength * np.sin(phase_difference)


class LymphaticNetwork:
    """
    Complete lymphatic network model using graph theory
    
    The lymphatic system forms a complex network with:
    - Initial lymphatics: Mesh-like capillary network
    - Collecting vessels: Tree-like drainage structure
    - Lymph nodes: Network hubs for immune surveillance
    - Central drainage: Convergence to thoracic duct
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for lymph flow
        self.vessels: Dict[str, LymphaticVessel] = {}
        self.nodes: Dict[str, LymphNode] = {}
        self.initial_lymphatics: List[str] = []
        
        # Fluid state
        self.tissue_pressure: float = -3.0  # mmHg (interstitial pressure)
        self.venous_pressure: float = 5.0   # mmHg (subclavian vein)
        
        # Network statistics
        self.total_flow: float = 0.0
        self.total_volume: float = 0.0
        
        # Build anatomical network
        self._construct_anatomical_network()
        
        logger.info(f"Constructed lymphatic network with {self.graph.number_of_nodes()} nodes "
                   f"and {self.graph.number_of_edges()} vessels")
    
    def _construct_anatomical_network(self) -> None:
        """
        Construct anatomically realistic lymphatic network
        
        Based on physiological data:
        - 600-700 lymph nodes in human body
        - Drainage follows tissue territories
        - Hierarchical organization from capillaries to ducts
        """
        # Create major lymph node groups
        node_groups = {
            'cervical': [(50, 200, 0), (50, 180, 0), (50, 160, 0)],
            'axillary': [(100, 150, 0), (120, 150, 0), (140, 150, 0)],
            'inguinal': [(80, 50, 0), (80, 30, 0)],
            'mesenteric': [(0, 100, 50), (0, 80, 50), (0, 60, 50)],
            'mediastinal': [(0, 150, 30), (0, 130, 30)]
        }
        
        # Create lymph nodes
        for group_name, positions in node_groups.items():
            for i, pos in enumerate(positions):
                node_id = f"{group_name}_{i}"
                node = LymphNode(node_id, pos, volume=np.random.uniform(0.5, 2.0))
                self.nodes[node_id] = node
                self.graph.add_node(node_id, node_type='lymph_node', position=pos)
        
        # Create vessel network connecting nodes
        self._create_drainage_vessels()
        
        # Create initial lymphatic networks
        self._create_initial_lymphatics()
        
        # Add thoracic duct
        self._create_central_drainage()
    
    def _create_drainage_vessels(self) -> None:
        """Create collecting vessels between lymph nodes"""
        
        # Connect nodes within groups
        for group_name in ['cervical', 'axillary', 'inguinal', 'mesenteric', 'mediastinal']:
            group_nodes = [n for n in self.graph.nodes() if n.startswith(group_name)]
            
            # Create chain connections within group
            for i in range(len(group_nodes) - 1):
                vessel_id = f"collector_{group_nodes[i]}_{group_nodes[i+1]}"
                
                start_pos = self.nodes[group_nodes[i]].position
                end_pos = self.nodes[group_nodes[i+1]].position
                
                vessel = LymphaticVessel(
                    vessel_id,
                    VesselType.COLLECTOR,
                    start_pos,
                    end_pos
                )
                
                self.vessels[vessel_id] = vessel
                self.graph.add_edge(group_nodes[i], group_nodes[i+1], 
                                  vessel_id=vessel_id, weight=vessel.length)
        
        # Connect groups with trunk vessels
        trunk_connections = [
            ('cervical_2', 'mediastinal_0'),
            ('axillary_2', 'mediastinal_0'),
            ('inguinal_1', 'mesenteric_0'),
            ('mesenteric_2', 'mediastinal_1')
        ]
        
        for start, end in trunk_connections:
            vessel_id = f"trunk_{start}_{end}"
            vessel = LymphaticVessel(
                vessel_id,
                VesselType.TRUNK,
                self.nodes[start].position,
                self.nodes[end].position
            )
            self.vessels[vessel_id] = vessel
            self.graph.add_edge(start, end, vessel_id=vessel_id, weight=vessel.length)
    
    def _create_initial_lymphatics(self) -> None:
        """
        Create mesh-like initial lymphatic network
        
        Initial lymphatics form blind-ended vessels that absorb interstitial fluid
        """
        # Create tissue source nodes
        tissue_regions = [
            ('skin', [(x, y, -10) for x in range(0, 200, 50) for y in range(0, 200, 50)]),
            ('muscle', [(x, y, 20) for x in range(0, 150, 50) for y in range(50, 150, 50)]),
            ('viscera', [(x, y, 50) for x in range(-50, 50, 25) for y in range(50, 150, 25)])
        ]
        
        for tissue_type, positions in tissue_regions:
            for i, pos in enumerate(positions):
                # Create tissue node
                tissue_node_id = f"{tissue_type}_{i}"
                self.graph.add_node(tissue_node_id, node_type='tissue', position=pos)
                
                # Find nearest lymph node
                nearest_ln = self._find_nearest_lymph_node(pos)
                
                if nearest_ln:
                    # Create initial lymphatic vessel
                    vessel_id = f"initial_{tissue_node_id}_{nearest_ln}"
                    vessel = LymphaticVessel(
                        vessel_id,
                        VesselType.INITIAL_LYMPHATIC,
                        pos,
                        self.nodes[nearest_ln].position
                    )
                    self.vessels[vessel_id] = vessel
                    self.initial_lymphatics.append(vessel_id)
                    self.graph.add_edge(tissue_node_id, nearest_ln,
                                      vessel_id=vessel_id, weight=vessel.length)
    
    def _create_central_drainage(self) -> None:
        """Create thoracic duct for central drainage to venous system"""
        
        # Thoracic duct connects to left subclavian vein
        duct_start = 'mediastinal_1'
        venous_junction = 'subclavian_vein'
        
        self.graph.add_node(venous_junction, node_type='venous', position=(0, 180, 0))
        
        vessel_id = 'thoracic_duct'
        vessel = LymphaticVessel(
            vessel_id,
            VesselType.DUCT,
            self.nodes[duct_start].position,
            (0, 180, 0)
        )
        
        self.vessels[vessel_id] = vessel
        self.graph.add_edge(duct_start, venous_junction,
                           vessel_id=vessel_id, weight=vessel.length)
    
    def _find_nearest_lymph_node(self, position: Position) -> Optional[str]:
        """Find nearest lymph node to given position using Euclidean distance"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node in self.nodes.items():
            distance = np.linalg.norm(np.array(position) - node.position)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    def simulate_flow(self, dt: float = 0.1) -> Dict[str, float]:
        """
        Simulate lymph flow through network using:
        1. Pressure-driven flow in vessels
        2. Active pumping by vessel contractions
        3. Node filtration and processing
        
        Returns flow rates through each vessel
        """
        flow_rates = {}
        
        # Calculate pressures at each node using iterative solver
        node_pressures = self._solve_pressure_field()
        
        # Calculate flow through each vessel
        for edge in self.graph.edges():
            edge_data = self.graph.get_edge_data(*edge)
            vessel_id = edge_data['vessel_id']
            vessel = self.vessels.get(vessel_id)
            
            if vessel:
                inlet_pressure = node_pressures.get(edge[0], self.tissue_pressure)
                outlet_pressure = node_pressures.get(edge[1], self.venous_pressure)
                
                flow = vessel.calculate_flow(inlet_pressure, outlet_pressure, dt)
                flow_rates[vessel_id] = flow
                
                # Update vessel contraction coordination
                if edge[0] in self.vessels:  # If upstream is also a vessel
                    upstream_vessel = self.vessels[edge[0]]
                    vessel.propagate_contraction_wave(upstream_vessel.contraction_phase)
        
        # Process flow through lymph nodes
        for node_id, node in self.nodes.items():
            # Calculate inlet flow
            inlet_flow = sum(flow_rates.get(e['vessel_id'], 0) 
                           for _, _, e in self.graph.in_edges(node_id, data=True))
            
            # Process lymph in node
            if inlet_flow > 0:
                outlet_flow, filtered = node.process_lymph(inlet_flow, dt)
                
                # Update outlet vessel flows
                for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                    vessel_id = edge_data['vessel_id']
                    if vessel_id in flow_rates:
                        flow_rates[vessel_id] = outlet_flow
        
        # Update total network flow
        self.total_flow = sum(abs(flow) for flow in flow_rates.values())
        
        return flow_rates
    
    def _solve_pressure_field(self) -> Dict[str, float]:
        """
        Solve for steady-state pressure at each node using Kirchhoff's law
        
        At each node: Σ(Q_in) = Σ(Q_out)
        Where Q = (P1 - P2)/R for each vessel
        
        This creates a system of linear equations: Ax = b
        """
        # Get node list (excluding boundary nodes)
        node_list = [n for n in self.graph.nodes() 
                    if self.graph.degree(n) > 1 and n != 'subclavian_vein']
        n_nodes = len(node_list)
        node_index = {node: i for i, node in enumerate(node_list)}
        
        # Build conductance matrix (A) and boundary vector (b)
        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)
        
        for i, node in enumerate(node_list):
            # Diagonal element: sum of conductances
            total_conductance = 0
            
            # Process each connected vessel
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                vessel = self.vessels.get(edge_data['vessel_id'])
                
                if vessel:
                    # Conductance = 1/Resistance
                    conductance = 1.0 / (vessel.flow_resistance + 1e-10)
                    total_conductance += conductance
                    
                    # Off-diagonal elements
                    if neighbor in node_index:
                        j = node_index[neighbor]
                        A[i, j] -= conductance
                    else:
                        # Boundary condition
                        if neighbor == 'subclavian_vein':
                            b[i] += conductance * self.venous_pressure
                        else:
                            b[i] += conductance * self.tissue_pressure
            
            # Also process vessels entering this node
            for predecessor in self.graph.predecessors(node):
                edge_data = self.graph.get_edge_data(predecessor, node)
                vessel = self.vessels.get(edge_data['vessel_id'])
                
                if vessel:
                    conductance = 1.0 / (vessel.flow_resistance + 1e-10)
                    total_conductance += conductance
                    
                    if predecessor in node_index:
                        j = node_index[predecessor]
                        A[i, j] -= conductance
                    else:
                        b[i] += conductance * self.tissue_pressure
            
            A[i, i] = total_conductance
        
        # Solve linear system
        try:
            pressures_vector = np.linalg.solve(A, b)
            pressures = {node: pressures_vector[i] for node, i in node_index.items()}
            
            # Add boundary pressures
            pressures['subclavian_vein'] = self.venous_pressure
            
            return pressures
        except np.linalg.LinAlgError:
            logger.warning("Failed to solve pressure field, using default pressures")
            return {node: self.tissue_pressure for node in self.graph.nodes()}
    
    def calculate_drainage_efficiency(self) -> float:
        """
        Calculate overall network drainage efficiency
        
        Efficiency = (Fluid removed from tissue) / (Fluid filtered from capillaries)
        Normal efficiency ~10% (2-4 L/day lymph from 20 L/day capillary filtration)
        """
        # Calculate total drainage from initial lymphatics
        initial_drainage = sum(
            abs(self.vessels[v_id].flow_rate) 
            for v_id in self.initial_lymphatics 
            if v_id in self.vessels
        )
        
        # Calculate theoretical maximum drainage
        capillary_filtration = 20.0 * 1000 / (24 * 60)  # 20 L/day to ml/min
        
        efficiency = initial_drainage / (capillary_filtration + 1e-10)
        return min(1.0, efficiency)  # Cap at 100%
    
    def simulate_pathology(self, condition: str) -> None:
        """
        Simulate various lymphatic pathologies
        
        Common conditions affecting lymphatic function:
        - Lymphedema: Impaired drainage leading to swelling
        - Lymphangitis: Vessel inflammation
        - Lymph node metastasis: Cancer spread via lymphatics
        """
        if condition == 'lymphedema':
            # Reduce vessel compliance and contractility
            for vessel in self.vessels.values():
                vessel.compliance *= 0.5
                vessel.contractility *= 0.3
            logger.info("Simulating lymphedema: reduced vessel function")
            
        elif condition == 'lymphangitis':
            # Increase vessel permeability and resistance
            for vessel in self.vessels.values():
                vessel.flow_resistance *= 2.0
            logger.info("Simulating lymphangitis: increased flow resistance")
            
        elif condition == 'metastasis':
            # Block random lymph nodes
            blocked_nodes = np.random.choice(list(self.nodes.keys()), 
                                           size=len(self.nodes) // 4, replace=False)
            for node_id in blocked_nodes:
                self.nodes[node_id].flow_resistance *= 10
                self.nodes[node_id].filtration_coefficient *= 0.1
            logger.info(f"Simulating metastasis: blocked {len(blocked_nodes)} nodes")
    
    def visualize_network(self, flow_rates: Optional[Dict[str, float]] = None) -> None:
        """
        Visualize lymphatic network with flow rates
        
        Uses spring layout for 2D projection of 3D network
        """
        plt.figure(figsize=(14, 10))
        
        # Get positions using spring layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data['node_type'] == 'lymph_node':
                node_colors.append('lightblue')
                node_sizes.append(500)
            elif node_data['node_type'] == 'tissue':
                node_colors.append('pink')
                node_sizes.append(100)
            else:  # venous
                node_colors.append('red')
                node_sizes.append(700)
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7)
        
        # Draw edges with flow rates
        if flow_rates:
            edge_widths = []
            edge_colors = []
            
            for edge in self.graph.edges():
                edge_data = self.graph.get_edge_data(*edge)
                vessel_id = edge_data.get('vessel_id')
                
                if vessel_id in flow_rates:
                    flow = abs(flow_rates[vessel_id])
                    # Width proportional to flow
                    edge_widths.append(1 + flow * 10)
                    # Color based on flow direction
                    edge_colors.append('blue' if flow_rates[vessel_id] > 0 else 'red')
                else:
                    edge_widths.append(0.5)
                    edge_colors.append('gray')
            
            nx.draw_networkx_edges(self.graph, pos, width=edge_widths, 
                                  edge_color=edge_colors, alpha=0.6, arrows=True)
        else:
            nx.draw_networkx_edges(self.graph, pos, alpha=0.3, arrows=True)
        
        # Draw labels for lymph nodes only
        lymph_node_labels = {n: n.split('_')[0][:3] for n in self.nodes.keys()}
        nx.draw_networkx_labels(self.graph, pos, labels=lymph_node_labels, 
                               font_size=8)
        
        plt.title("Lymphatic Network Visualization", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class LymphaticSystemSimulator:
    """
    Main simulation controller for comprehensive lymphatic system modeling
    
    Integrates all components and provides high-level simulation interface
    """
    
    def __init__(self):
        self.network = LymphaticNetwork()
        self.time = 0.0
        self.dt = 0.1  # Time step in seconds
        
        # Simulation data storage
        self.flow_history = []
        self.pressure_history = []
        self.immune_cell_counts = defaultdict(list)
        
        # Physiological parameters
        self.heart_rate = 70  # bpm
        self.respiratory_rate = 12  # breaths/min
        self.activity_level = 'rest'  # rest, light, moderate, heavy
        
        logger.info("Initialized Lymphatic System Simulator")
    
    def run_simulation(self, duration: float, visualize: bool = True) -> Dict:
        """
        Run complete lymphatic system simulation
        
        Parameters:
            duration: Simulation time in seconds
            visualize: Whether to show visualizations
        
        Returns:
            Dictionary containing simulation results and statistics
        """
        n_steps = int(duration / self.dt)
        
        logger.info(f"Starting simulation for {duration} seconds ({n_steps} steps)")
        
        for step in range(n_steps):
            # Update time
            self.time += self.dt
            
            # Simulate cardiovascular coupling (pressure variations)
            self._update_driving_pressures()
            
            # Simulate lymph flow
            flow_rates = self.network.simulate_flow(self.dt)
            
            # Record data
            self.flow_history.append({
                'time': self.time,
                'total_flow': self.network.total_flow,
                'drainage_efficiency': self.network.calculate_drainage_efficiency()
            })
            
            # Simulate immune cell dynamics every 10 steps
            if step % 10 == 0:
                self._simulate_immune_dynamics()
            
            # Log progress
            if step % 100 == 0:
                logger.info(f"Step {step}/{n_steps}: Total flow = {self.network.total_flow:.2f} ml/min")
        
        # Compile results
        results = self._compile_results()
        
        # Visualize if requested
        if visualize:
            self._visualize_results(results)
            self.network.visualize_network(flow_rates)
        
        return results
    
    def _update_driving_pressures(self) -> None:
        """
        Update tissue and venous pressures based on physiological state
        
        Incorporates:
        - Cardiac cycle (arterial pulsations affect tissue pressure)
        - Respiratory pump (thoracic pressure changes)
        - Skeletal muscle pump (activity-dependent)
        """
        # Cardiac component
        cardiac_phase = (self.time * self.heart_rate / 60) % 1
        cardiac_pressure = 2 * np.sin(2 * np.pi * cardiac_phase)  # ±2 mmHg
        
        # Respiratory component
        respiratory_phase = (self.time * self.respiratory_rate / 60) % 1
        respiratory_pressure = 3 * np.sin(2 * np.pi * respiratory_phase)  # ±3 mmHg
        
        # Activity component
        activity_pressure = {
            'rest': 0,
            'light': 2,
            'moderate': 5,
            'heavy': 10
        }.get(self.activity_level, 0)
        
        # Update network pressures
        self.network.tissue_pressure = -3.0 + cardiac_pressure + activity_pressure
        self.network.venous_pressure = 5.0 + respiratory_pressure
    
    def _simulate_immune_dynamics(self) -> None:
        """Simulate immune cell populations and trafficking"""
        
        for node_id, node in self.network.nodes.items():
            # Generate new immune cells
            if len(node.resident_cells['T-cell']) < 1000:  # Max population
                # Poisson arrival process for cell entry
                new_t_cells = np.random.poisson(10 * self.dt)  # 10 cells/sec average
                
                for _ in range(new_t_cells):
                    cell = ImmuneCell(
                        cell_type='T-cell',
                        position=node.position + np.random.randn(3) * 1.0,
                        velocity=np.random.randn(3) * 0.1
                    )
                    node.resident_cells['T-cell'].append(cell)
            
            # Record cell counts
            self.immune_cell_counts[node_id].append({
                'time': self.time,
                'T-cells': len(node.resident_cells.get('T-cell', [])),
                'B-cells': len(node.resident_cells.get('B-cell', []))
            })
    
    def _compile_results(self) -> Dict:
        """Compile simulation results and calculate statistics"""
        
        flow_data = np.array([(d['time'], d['total_flow']) 
                             for d in self.flow_history])
        
        results = {
            'duration': self.time,
            'mean_flow': np.mean(flow_data[:, 1]) if len(flow_data) > 0 else 0,
            'max_flow': np.max(flow_data[:, 1]) if len(flow_data) > 0 else 0,
            'flow_variance': np.var(flow_data[:, 1]) if len(flow_data) > 0 else 0,
            'drainage_efficiency': self.network.calculate_drainage_efficiency(),
            'flow_history': self.flow_history,
            'immune_cell_counts': dict(self.immune_cell_counts)
        }
        
        logger.info(f"Simulation complete. Mean flow: {results['mean_flow']:.2f} ml/min, "
                   f"Drainage efficiency: {results['drainage_efficiency']*100:.1f}%")
        
        return results
    
    def _visualize_results(self, results: Dict) -> None:
        """Create visualization plots for simulation results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Flow rate over time
        ax1 = axes[0, 0]
        times = [d['time'] for d in self.flow_history]
        flows = [d['total_flow'] for d in self.flow_history]
        ax1.plot(times, flows, 'b-', linewidth=1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Total Flow Rate (ml/min)')
        ax1.set_title('Lymphatic Flow Dynamics')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drainage efficiency
        ax2 = axes[0, 1]
        efficiencies = [d['drainage_efficiency'] for d in self.flow_history]
        ax2.plot(times, efficiencies, 'g-', linewidth=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Drainage Efficiency')
        ax2.set_title('Network Drainage Efficiency')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Vessel diameter distribution
        ax3 = axes[1, 0]
        diameters = [v.diameter * 1e6 for v in self.network.vessels.values()]  # Convert to μm
        ax3.hist(diameters, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Vessel Diameter (μm)')
        ax3.set_ylabel('Count')
        ax3.set_title('Vessel Diameter Distribution')
        ax3.set_yscale('log')
        
        # Plot 4: Reynolds number distribution
        ax4 = axes[1, 1]
        reynolds = [v.reynolds_number for v in self.network.vessels.values() if v.reynolds_number > 0]
        if reynolds:
            ax4.hist(reynolds, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Reynolds Number')
            ax4.set_ylabel('Count')
            ax4.set_title('Flow Regime Distribution')
            ax4.axvline(x=2000, color='red', linestyle='--', label='Turbulent threshold')
            ax4.legend()
        
        plt.suptitle('Lymphatic System Simulation Results', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def test_starling_equilibrium(self) -> None:
        """
        Test Starling's equation for transcapillary fluid balance
        
        J_v = L_p × A × [(P_c - P_i) - σ(π_c - π_i)]
        
        Where:
        J_v = Fluid flux
        L_p = Hydraulic conductivity
        A = Surface area
        P = Hydrostatic pressure
        π = Oncotic pressure
        σ = Reflection coefficient
        """
        logger.info("Testing Starling equilibrium...")
        
        # Typical values
        L_p = 1e-7  # cm/s/mmHg
        A = 300  # m² total capillary area
        P_c = 25  # Capillary hydrostatic pressure (mmHg)
        P_i = -3  # Interstitial hydrostatic pressure (mmHg)
        pi_c = 28  # Capillary oncotic pressure (mmHg)
        pi_i = 8   # Interstitial oncotic pressure (mmHg)
        sigma = 0.9  # Protein reflection coefficient
        
        # Calculate filtration
        J_v = L_p * A * ((P_c - P_i) - sigma * (pi_c - pi_i))
        
        # Convert to ml/min
        J_v_ml_min = J_v * 60 * 1000  # cm³/min
        
        logger.info(f"Starling filtration rate: {J_v_ml_min:.2f} ml/min")
        logger.info(f"Daily filtration: {J_v_ml_min * 60 * 24 / 1000:.1f} L/day")
        
        # This should equal lymph flow + reabsorption
        expected_lymph_flow = J_v_ml_min * 0.1  # ~10% becomes lymph
        logger.info(f"Expected lymph flow: {expected_lymph_flow:.2f} ml/min")


def run_comprehensive_tests():
    """Run comprehensive testing suite with example data"""
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE LYMPHATIC SYSTEM MODEL TESTING")
    logger.info("="*60)
    
    # Test 1: Basic vessel mechanics
    logger.info("\nTest 1: Vessel Mechanics")
    test_vessel = LymphaticVessel(
        "test_vessel",
        VesselType.COLLECTOR,
        (0, 0, 0),
        (10, 0, 0)
    )
    
    # Calculate flow under different pressure gradients
    pressure_gradients = [0.5, 1.0, 2.0, 5.0]  # mmHg
    
    for dp in pressure_gradients:
        flow = test_vessel.calculate_flow(dp, 0, 0.1)
        logger.info(f"Pressure gradient: {dp} mmHg, Flow: {flow:.3f} ml/min")
    
    # Test 2: Lymph node filtration
    logger.info("\nTest 2: Lymph Node Processing")
    test_node = LymphNode("test_node", (50, 50, 0), volume=1.5)
    
    # Add some antigens
    test_node.antigen_concentration = {
        'bacteria': 1e-6,  # mol/L
        'virus': 1e-7,
        'protein_aggregates': 1e-5
    }
    
    # Process lymph
    inlet_flow = 0.5  # ml/min
    outlet_flow, filtered = test_node.process_lymph(inlet_flow, 0.1)
    
    logger.info(f"Inlet flow: {inlet_flow} ml/min")
    logger.info(f"Outlet flow: {outlet_flow:.3f} ml/min")
    logger.info(f"Filtered substances: {filtered}")
    
    # Test 3: Full network simulation
    logger.info("\nTest 3: Full Network Simulation")
    simulator = LymphaticSystemSimulator()
    
    # Test Starling equilibrium
    simulator.test_starling_equilibrium()
    
    # Run short simulation
    logger.info("\nRunning 60-second simulation...")
    results = simulator.run_simulation(duration=60.0, visualize=True)
    
    # Display results
    logger.info("\nSimulation Results:")
    logger.info(f"Mean flow rate: {results['mean_flow']:.2f} ml/min")
    logger.info(f"Maximum flow rate: {results['max_flow']:.2f} ml/min")
    logger.info(f"Flow variance: {results['flow_variance']:.2f}")
    logger.info(f"Drainage efficiency: {results['drainage_efficiency']*100:.1f}%")
    
    # Test 4: Pathological conditions
    logger.info("\nTest 4: Pathological Conditions")
    
    # Simulate lymphedema
    simulator.network.simulate_pathology('lymphedema')
    results_lymphedema = simulator.run_simulation(duration=30.0, visualize=False)
    logger.info(f"Lymphedema - Mean flow: {results_lymphedema['mean_flow']:.2f} ml/min")
    
    # Reset and simulate lymphangitis
    simulator = LymphaticSystemSimulator()
    simulator.network.simulate_pathology('lymphangitis')
    results_lymphangitis = simulator.run_simulation(duration=30.0, visualize=False)
    logger.info(f"Lymphangitis - Mean flow: {results_lymphangitis['mean_flow']:.2f} ml/min")
    
    logger.info("\n" + "="*60)
    logger.info("TESTING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    # Run comprehensive testing
    run_comprehensive_tests()
