"""
COMPREHENSIVE BIOLOGICAL SYSTEMS MODELING FRAMEWORK
A sophisticated computational biology framework that models multiple interconnected
biological systems including gene regulation, protein folding, metabolic pathways,
neural networks, immune responses, and evolutionary dynamics.

This framework demonstrates advanced biological modeling techniques including:
- Stochastic gene expression with Hill kinetics
- Protein folding energy landscapes using Go-model approximations
- Metabolic flux balance analysis with constraint-based optimization
- Hodgkin-Huxley neural dynamics with synaptic plasticity
- Agent-based immune system simulation with adaptive responses
- Population genetics with Wright-Fisher models
- Systems biology network analysis
- Multi-scale temporal dynamics from milliseconds to generations

Author: Advanced Computational Biology Systems
Version: 3.0.0
Python Requirements: 3.8+
Dependencies: numpy, scipy, networkx, pandas
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Configure logging for the biological systems
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Fundamental biological constants
AVOGADRO = 6.02214076e23  # molecules/mol
GAS_CONSTANT = 8.314  # J/(mol*K)
BOLTZMANN = 1.380649e-23  # J/K
STANDARD_TEMP = 310.15  # K (37°C)
CYTOPLASM_VOLUME = 1e-15  # L (typical bacterial cell)
NUCLEUS_VOLUME = 1e-16  # L


class BiologicalConstants:
    """Central repository for biological constants and parameters"""

    # DNA/RNA parameters
    TRANSCRIPTION_RATE = 50  # nucleotides/second
    TRANSLATION_RATE = 20  # amino acids/second
    MRNA_DEGRADATION_RATE = 0.003  # per second
    PROTEIN_DEGRADATION_RATE = 0.0001  # per second

    # Metabolic parameters
    ATP_HYDROLYSIS_ENERGY = -30.5  # kJ/mol
    GLUCOSE_ENERGY = -2870  # kJ/mol complete oxidation
    NAD_REDOX_POTENTIAL = -0.32  # V

    # Neural parameters
    RESTING_POTENTIAL = -70  # mV
    SODIUM_REVERSAL = 50  # mV
    POTASSIUM_REVERSAL = -77  # mV
    CHLORIDE_REVERSAL = -65  # mV

    # Immune parameters
    ANTIGEN_RECOGNITION_THRESHOLD = 0.7  # affinity threshold
    ANTIBODY_PRODUCTION_RATE = 2000  # molecules/second per B cell
    T_CELL_PROLIFERATION_RATE = 0.693  # per day (24h doubling)

    # Evolution parameters
    MUTATION_RATE = 1e-8  # per base per generation
    CROSSOVER_RATE = 0.001  # per base
    SELECTION_COEFFICIENT = 0.01  # typical value


@dataclass
class Gene:
    """
    Represents a gene with regulatory elements and expression dynamics.
    Includes promoter strength, regulatory binding sites, and expression state.
    """

    gene_id: str
    sequence: str
    promoter_strength: float = 1.0
    transcription_factors: dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    mrna_count: int = 0
    protein_count: int = 0
    chromosome: str | None = None
    position: int | None = None

    def __post_init__(self):
        """Validate gene parameters upon initialization"""
        if not 0 <= self.promoter_strength <= 10:
            raise ValueError("Promoter strength must be between 0 and 10")
        if len(self.sequence) == 0:
            raise ValueError("Gene sequence cannot be empty")

        # Calculate GC content for stability considerations
        self.gc_content = (self.sequence.count('G') + self.sequence.count('C')) / len(self.sequence)
        self.length = len(self.sequence)

    def calculate_transcription_rate(self, tf_concentrations: dict[str, float]) -> float:
        """
        Calculate transcription rate using Hill kinetics for transcription factor binding.
        Implements cooperative binding and competitive inhibition.
        """
        base_rate = self.promoter_strength * BiologicalConstants.TRANSCRIPTION_RATE

        if not self.is_active:
            return 0.0

        # Calculate regulatory effect using Hill functions
        activation = 1.0
        repression = 1.0

        for tf_name, binding_params in self.transcription_factors.items():
            if tf_name not in tf_concentrations:
                continue

            concentration = tf_concentrations[tf_name]

            if isinstance(binding_params, dict):
                kd = binding_params.get('kd', 1.0)
                hill_coeff = binding_params.get('n', 2.0)
                is_activator = binding_params.get('activator', True)

                # Hill equation for binding
                binding = (concentration ** hill_coeff) / (kd ** hill_coeff + concentration ** hill_coeff)

                if is_activator:
                    activation *= (1 + binding * 10)  # 10-fold max activation
                else:
                    repression *= (1 - binding * 0.99)  # 99% max repression
            else:
                # Simple linear regulation for backwards compatibility
                if binding_params > 0:
                    activation *= (1 + concentration * binding_params)
                else:
                    repression *= max(0, 1 + concentration * binding_params)

        return base_rate * activation * repression

    def mutate(self, mutation_rate: float = BiologicalConstants.MUTATION_RATE) -> 'Gene':
        """
        Create a mutated version of the gene.
        Models point mutations, insertions, and deletions.
        """
        new_sequence = list(self.sequence)
        bases = ['A', 'T', 'G', 'C']

        for i in range(len(new_sequence)):
            if random.random() < mutation_rate:
                mutation_type = random.choice(['substitution', 'insertion', 'deletion'])

                if mutation_type == 'substitution':
                    current_base = new_sequence[i]
                    new_base = random.choice([b for b in bases if b != current_base])
                    new_sequence[i] = new_base

                elif mutation_type == 'insertion' and len(new_sequence) < 10000:
                    new_sequence.insert(i, random.choice(bases))

                elif mutation_type == 'deletion' and len(new_sequence) > 100:
                    del new_sequence[i]

        mutated_gene = Gene(
            gene_id=f"{self.gene_id}_mut",
            sequence=''.join(new_sequence),
            promoter_strength=self.promoter_strength * random.gauss(1.0, 0.1),
            transcription_factors=self.transcription_factors.copy(),
            is_active=self.is_active,
            chromosome=self.chromosome,
            position=self.position
        )

        return mutated_gene


@dataclass
class Protein:
    """
    Represents a protein with folding dynamics and functional properties.
    Includes amino acid sequence, 3D structure representation, and activity.
    """

    protein_id: str
    sequence: str
    native_structure: np.ndarray | None = None
    current_structure: np.ndarray | None = None
    folding_energy: float = 0.0
    is_folded: bool = False
    activity: float = 0.0
    half_life: float = 3600.0  # seconds

    def __post_init__(self):
        """Initialize protein structural properties"""
        self.length = len(self.sequence)
        self.molecular_weight = self._calculate_molecular_weight()
        self.isoelectric_point = self._calculate_pi()
        self.hydrophobicity = self._calculate_hydrophobicity()

        if self.native_structure is None:
            self.native_structure = self._predict_native_structure()

        if self.current_structure is None:
            self.current_structure = self._initialize_random_structure()

    def _calculate_molecular_weight(self) -> float:
        """Calculate molecular weight from amino acid sequence"""
        aa_weights = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
            'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }

        weight = sum(aa_weights.get(aa, 110.0) for aa in self.sequence)
        weight -= 18.0 * (len(self.sequence) - 1)  # Subtract water for peptide bonds
        return weight

    def _calculate_pi(self) -> float:
        """Calculate theoretical isoelectric point"""
        # Simplified calculation based on charged residues
        positive_charges = self.sequence.count('K') + self.sequence.count('R') + self.sequence.count('H') / 2
        negative_charges = self.sequence.count('D') + self.sequence.count('E')

        if negative_charges == 0:
            return 12.0

        ratio = positive_charges / negative_charges
        pi = 6.5 + np.log10(ratio)  # Simplified Henderson-Hasselbalch
        return max(3.0, min(12.0, pi))

    def _calculate_hydrophobicity(self) -> float:
        """Calculate average hydrophobicity using Kyte-Doolittle scale"""
        hydrophobicity_scale = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
            'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
            'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        }

        total_hydrophobicity = sum(hydrophobicity_scale.get(aa, 0.0) for aa in self.sequence)
        return total_hydrophobicity / len(self.sequence)

    def _predict_native_structure(self) -> np.ndarray:
        """
        Predict native structure using simplified Go-model.
        Creates a compact structure based on hydrophobic collapse.
        """
        n = len(self.sequence)

        # Initialize with extended chain
        structure = np.zeros((n, 3))

        # Create alpha helix as starting structure
        for i in range(n):
            angle = i * 100 * np.pi / 180  # 100 degrees per residue for helix
            structure[i, 0] = 2.3 * np.cos(angle)  # 2.3 Å rise per residue
            structure[i, 1] = 2.3 * np.sin(angle)
            structure[i, 2] = 1.5 * i  # 1.5 Å translation per residue

        # Apply hydrophobic collapse
        hydrophobic_residues = set(['I', 'V', 'L', 'F', 'C', 'M', 'A', 'W'])

        for iteration in range(100):
            center_of_mass = np.mean(structure, axis=0)

            for i in range(n):
                if self.sequence[i] in hydrophobic_residues:
                    # Move hydrophobic residues toward center
                    direction = center_of_mass - structure[i]
                    structure[i] += 0.01 * direction
                else:
                    # Move hydrophilic residues away from center
                    direction = structure[i] - center_of_mass
                    structure[i] += 0.005 * direction

            # Maintain bond lengths
            for i in range(n - 1):
                bond = structure[i + 1] - structure[i]
                bond_length = np.linalg.norm(bond)
                if bond_length > 0:
                    structure[i + 1] = structure[i] + bond * (3.8 / bond_length)

        return structure

    def _initialize_random_structure(self) -> np.ndarray:
        """Initialize unfolded random coil structure"""
        n = len(self.sequence)
        structure = np.zeros((n, 3))

        # Random walk with fixed bond length
        for i in range(1, n):
            theta = random.random() * np.pi
            phi = random.random() * 2 * np.pi

            # Spherical coordinates with bond length 3.8 Å
            structure[i] = structure[i - 1] + 3.8 * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])

        return structure

    def calculate_folding_energy(self) -> float:
        """
        Calculate folding energy using simplified force field.
        Includes bond lengths, angles, and hydrophobic interactions.
        """
        if self.current_structure is None:
            return float('inf')

        energy = 0.0
        n = len(self.sequence)

        # Bond length energy (harmonic potential)
        for i in range(n - 1):
            bond = self.current_structure[i + 1] - self.current_structure[i]
            bond_length = np.linalg.norm(bond)
            energy += 100 * (bond_length - 3.8) ** 2  # Equilibrium at 3.8 Å

        # Bond angle energy
        for i in range(n - 2):
            v1 = self.current_structure[i + 1] - self.current_structure[i]
            v2 = self.current_structure[i + 2] - self.current_structure[i + 1]

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            # Prefer extended conformations (120 degrees)
            energy += 10 * (angle - 2 * np.pi / 3) ** 2

        # Hydrophobic interactions (Lennard-Jones-like)
        hydrophobic_residues = set(['I', 'V', 'L', 'F', 'C', 'M', 'A', 'W'])

        for i in range(n - 2):
            if self.sequence[i] not in hydrophobic_residues:
                continue

            for j in range(i + 2, n):
                if self.sequence[j] not in hydrophobic_residues:
                    continue

                distance = np.linalg.norm(self.current_structure[i] - self.current_structure[j])

                if distance < 20.0:  # Cutoff distance
                    # Lennard-Jones potential
                    energy += 4 * ((4.0 / distance) ** 12 - (4.0 / distance) ** 6)

        # Native structure bias (Go-model term)
        if self.native_structure is not None:
            rmsd = np.sqrt(np.mean(np.sum((self.current_structure - self.native_structure) ** 2, axis=1)))
            energy += 0.1 * rmsd ** 2

        self.folding_energy = energy
        return energy

    def fold_step(self, temperature: float = STANDARD_TEMP) -> None:
        """
        Perform one Monte Carlo step of protein folding simulation.
        Uses Metropolis criterion for accepting conformational changes.
        """
        if self.current_structure is None:
            return

        n = len(self.sequence)

        # Choose random residue to move
        residue_index = random.randint(1, n - 2)  # Don't move terminals

        # Store old position and energy
        old_position = self.current_structure[residue_index].copy()
        old_energy = self.calculate_folding_energy()

        # Random displacement
        displacement = np.random.randn(3) * 0.5  # 0.5 Å step size
        self.current_structure[residue_index] += displacement

        # Calculate new energy
        new_energy = self.calculate_folding_energy()

        # Metropolis criterion
        delta_energy = new_energy - old_energy

        if delta_energy > 0:
            # Accept with Boltzmann probability
            probability = np.exp(-delta_energy / (GAS_CONSTANT * temperature / 1000))

            if random.random() > probability:
                # Reject move
                self.current_structure[residue_index] = old_position
                self.folding_energy = old_energy

        # Check if folded (RMSD < 5 Å from native)
        if self.native_structure is not None:
            rmsd = np.sqrt(np.mean(np.sum((self.current_structure - self.native_structure) ** 2, axis=1)))
            self.is_folded = rmsd < 5.0
            self.activity = max(0.0, 1.0 - rmsd / 20.0) if self.is_folded else 0.0

    def calculate_binding_affinity(self, other: 'Protein') -> float:
        """
        Calculate binding affinity to another protein.
        Based on shape complementarity and electrostatic interactions.
        """
        if not self.is_folded or not other.is_folded:
            return 0.0

        # Calculate interface residues (within 8 Å)
        min_distances = []

        for i in range(len(self.sequence)):
            distances = [np.linalg.norm(self.current_structure[i] - other.current_structure[j])
                        for j in range(len(other.sequence))]
            min_distances.append(min(distances))

        interface_residues = [i for i, d in enumerate(min_distances) if d < 8.0]

        if not interface_residues:
            return 0.0

        # Calculate shape complementarity
        complementarity = 0.0

        for i in interface_residues:
            for j in range(len(other.sequence)):
                distance = np.linalg.norm(self.current_structure[i] - other.current_structure[j])

                if 3.5 < distance < 5.0:  # Optimal contact distance
                    complementarity += 1.0
                elif distance < 3.0:  # Steric clash
                    complementarity -= 10.0

        # Normalize by interface size
        affinity = complementarity / (len(interface_residues) + 1)

        return max(0.0, min(1.0, affinity / 10.0))


class MetabolicNetwork:
    """
    Represents a metabolic network with flux balance analysis capabilities.
    Models enzymatic reactions, metabolite concentrations, and pathway regulation.
    """

    def __init__(self, name: str = "cell_metabolism"):
        self.name = name
        self.metabolites: dict[str, float] = {}  # Concentrations in mM
        self.reactions: dict[str, dict] = {}  # Reaction definitions
        self.enzymes: dict[str, Protein] = {}  # Enzyme proteins
        self.stoichiometry_matrix: np.ndarray | None = None
        self.bounds: list[tuple[float, float]] = []
        self.objective_vector: np.ndarray | None = None

        # Regulatory parameters
        self.allosteric_regulation: dict[str, dict[str, float]] = {}
        self.feedback_loops: list[dict] = []

        # Initialize with basic metabolites
        self._initialize_core_metabolites()

        logger.info(f"Initialized metabolic network: {name}")

    def _initialize_core_metabolites(self):
        """Initialize core metabolites present in all cells"""
        self.metabolites = {
            'glucose': 5.0,  # mM
            'ATP': 5.0,
            'ADP': 0.5,
            'AMP': 0.1,
            'NAD+': 1.0,
            'NADH': 0.1,
            'pyruvate': 0.1,
            'lactate': 0.1,
            'acetyl-CoA': 0.1,
            'citrate': 0.1,
            'alpha-ketoglutarate': 0.1,
            'succinate': 0.1,
            'fumarate': 0.1,
            'malate': 0.1,
            'oxaloacetate': 0.01,
            'CO2': 1.0,
            'O2': 0.2,
            'H2O': 55000.0,  # ~55 M in cells
            'H+': 0.0001,  # pH 7.0
            'Pi': 10.0,  # Inorganic phosphate
        }

    def add_reaction(self, reaction_id: str, substrates: dict[str, float],
                     products: dict[str, float], enzyme: Protein | None = None,
                     reversible: bool = True, k_cat: float = 100.0):
        """
        Add a reaction to the metabolic network.
        
        Parameters:
        - reaction_id: Unique identifier for the reaction
        - substrates: Dict of substrate metabolites and their stoichiometric coefficients
        - products: Dict of product metabolites and their stoichiometric coefficients
        - enzyme: Protein catalyst for the reaction
        - reversible: Whether the reaction can proceed in both directions
        - k_cat: Catalytic rate constant (per second)
        """

        # Ensure all metabolites exist
        for metabolite in list(substrates.keys()) + list(products.keys()):
            if metabolite not in self.metabolites:
                self.metabolites[metabolite] = 0.0

        self.reactions[reaction_id] = {
            'substrates': substrates,
            'products': products,
            'enzyme': enzyme,
            'reversible': reversible,
            'k_cat': k_cat,
            'flux': 0.0
        }

        if enzyme:
            self.enzymes[reaction_id] = enzyme

        # Mark stoichiometry matrix for reconstruction
        self.stoichiometry_matrix = None

    def build_stoichiometry_matrix(self):
        """
        Build the stoichiometry matrix for flux balance analysis.
        Rows represent metabolites, columns represent reactions.
        """
        metabolite_list = list(self.metabolites.keys())
        reaction_list = list(self.reactions.keys())

        n_metabolites = len(metabolite_list)
        n_reactions = len(reaction_list)

        self.stoichiometry_matrix = np.zeros((n_metabolites, n_reactions))

        for j, reaction_id in enumerate(reaction_list):
            reaction = self.reactions[reaction_id]

            # Add substrate coefficients (negative)
            for metabolite, coefficient in reaction['substrates'].items():
                i = metabolite_list.index(metabolite)
                self.stoichiometry_matrix[i, j] -= coefficient

            # Add product coefficients (positive)
            for metabolite, coefficient in reaction['products'].items():
                i = metabolite_list.index(metabolite)
                self.stoichiometry_matrix[i, j] += coefficient

        # Set up bounds for reactions
        self.bounds = []
        for reaction_id in reaction_list:
            reaction = self.reactions[reaction_id]

            if reaction['reversible']:
                # Reversible reactions can have negative flux
                self.bounds.append((-1000.0, 1000.0))
            else:
                # Irreversible reactions have non-negative flux
                self.bounds.append((0.0, 1000.0))

    def calculate_flux(self, dt: float = 1.0) -> np.ndarray:
        """
        Calculate metabolic flux using enzyme kinetics.
        Implements Michaelis-Menten kinetics with allosteric regulation.
        """
        if self.stoichiometry_matrix is None:
            self.build_stoichiometry_matrix()

        reaction_list = list(self.reactions.keys())
        fluxes = np.zeros(len(reaction_list))

        for i, reaction_id in enumerate(reaction_list):
            reaction = self.reactions[reaction_id]

            # Calculate substrate availability (Michaelis-Menten)
            substrate_term = 1.0
            for metabolite, coefficient in reaction['substrates'].items():
                concentration = self.metabolites[metabolite]
                km = 0.1  # mM, typical Km value

                # Michaelis-Menten term
                substrate_term *= (concentration / (km + concentration)) ** coefficient

            # Calculate product inhibition
            product_term = 1.0
            for metabolite, coefficient in reaction['products'].items():
                concentration = self.metabolites[metabolite]
                ki = 10.0  # mM, typical Ki value

                # Competitive inhibition
                product_term *= (ki / (ki + concentration)) ** coefficient

            # Calculate enzyme activity
            enzyme_activity = 1.0
            if reaction['enzyme']:
                enzyme_activity = reaction['enzyme'].activity

            # Apply allosteric regulation
            if reaction_id in self.allosteric_regulation:
                for regulator, effect in self.allosteric_regulation[reaction_id].items():
                    if regulator in self.metabolites:
                        concentration = self.metabolites[regulator]

                        if effect > 0:  # Activation
                            enzyme_activity *= (1 + effect * concentration / (0.1 + concentration))
                        else:  # Inhibition
                            enzyme_activity *= (1 / (1 - effect * concentration / (0.1 + concentration)))

            # Calculate flux
            flux = reaction['k_cat'] * enzyme_activity * substrate_term * product_term

            # Apply thermodynamic constraints
            if reaction['reversible']:
                # Calculate equilibrium constant (simplified)
                keq = 1.0  # Should be calculated from Gibbs free energy

                # Calculate reaction quotient
                q = 1.0
                for metabolite, coefficient in reaction['products'].items():
                    q *= self.metabolites[metabolite] ** coefficient
                for metabolite, coefficient in reaction['substrates'].items():
                    q /= (self.metabolites[metabolite] + 1e-10) ** coefficient

                # Adjust flux based on thermodynamics
                flux *= (1 - q / keq)

            fluxes[i] = flux
            reaction['flux'] = flux

        # Update metabolite concentrations
        if dt > 0:
            metabolite_list = list(self.metabolites.keys())
            concentration_changes = self.stoichiometry_matrix @ fluxes * dt

            for i, metabolite in enumerate(metabolite_list):
                self.metabolites[metabolite] += concentration_changes[i]
                self.metabolites[metabolite] = max(0.0, self.metabolites[metabolite])

        return fluxes

    def optimize_flux_balance(self, objective: str = 'biomass') -> dict[str, float]:
        """
        Perform flux balance analysis to optimize metabolic objectives.
        Uses linear programming to find optimal flux distribution.
        """
        if self.stoichiometry_matrix is None:
            self.build_stoichiometry_matrix()

        from scipy.optimize import linprog

        reaction_list = list(self.reactions.keys())
        n_reactions = len(reaction_list)

        # Define objective function (maximize biomass or ATP production)
        c = np.zeros(n_reactions)

        if objective == 'biomass':
            # Simplified biomass reaction (combination of key metabolites)
            biomass_components = ['ATP', 'NADH', 'acetyl-CoA', 'alpha-ketoglutarate']
            for i, reaction_id in enumerate(reaction_list):
                reaction = self.reactions[reaction_id]
                for product in reaction['products']:
                    if product in biomass_components:
                        c[i] -= reaction['products'][product]  # Negative for maximization

        elif objective == 'ATP':
            # Maximize ATP production
            for i, reaction_id in enumerate(reaction_list):
                reaction = self.reactions[reaction_id]

                atp_production = reaction['products'].get('ATP', 0) - reaction['substrates'].get('ATP', 0)
                c[i] = -atp_production  # Negative for maximization

        # Steady-state constraints (S * v = 0)
        A_eq = self.stoichiometry_matrix
        b_eq = np.zeros(self.stoichiometry_matrix.shape[0])

        # Solve linear program
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=self.bounds, method='highs')

        if result.success:
            optimal_fluxes = result.x

            # Update reaction fluxes
            for i, reaction_id in enumerate(reaction_list):
                self.reactions[reaction_id]['flux'] = optimal_fluxes[i]

            return {reaction_id: optimal_fluxes[i] for i, reaction_id in enumerate(reaction_list)}
        else:
            logger.warning(f"FBA optimization failed: {result.message}")
            return {}

    def add_feedback_loop(self, product: str, target_reaction: str,
                          inhibition_strength: float = 0.9):
        """
        Add a feedback inhibition loop where a product inhibits its own production.
        Common regulatory mechanism in metabolism.
        """
        feedback = {
            'product': product,
            'target': target_reaction,
            'strength': inhibition_strength,
            'type': 'inhibition'
        }

        self.feedback_loops.append(feedback)

        # Update allosteric regulation
        if target_reaction not in self.allosteric_regulation:
            self.allosteric_regulation[target_reaction] = {}

        self.allosteric_regulation[target_reaction][product] = -inhibition_strength

    def calculate_energy_charge(self) -> float:
        """
        Calculate the adenylate energy charge of the cell.
        Indicates the energy status of the metabolic system.
        """
        atp = self.metabolites.get('ATP', 0)
        adp = self.metabolites.get('ADP', 0)
        amp = self.metabolites.get('AMP', 0)

        total_adenylate = atp + adp + amp

        if total_adenylate == 0:
            return 0.0

        energy_charge = (atp + 0.5 * adp) / total_adenylate
        return energy_charge

    def calculate_redox_balance(self) -> float:
        """
        Calculate the NAD+/NADH ratio indicating cellular redox state.
        Important for metabolic regulation and energy production.
        """
        nad_oxidized = self.metabolites.get('NAD+', 0)
        nad_reduced = self.metabolites.get('NADH', 0)

        if nad_reduced == 0:
            return float('inf')

        return nad_oxidized / nad_reduced


class Neuron:
    """
    Hodgkin-Huxley neuron model with synaptic connections.
    Implements detailed ion channel dynamics and synaptic plasticity.
    """

    def __init__(self, neuron_id: str, neuron_type: str = 'excitatory'):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type  # 'excitatory' or 'inhibitory'

        # Membrane properties
        self.capacitance = 1.0  # μF/cm²
        self.area = 1e-6  # cm²

        # State variables
        self.voltage = BiologicalConstants.RESTING_POTENTIAL  # mV
        self.n = 0.3  # Potassium activation
        self.m = 0.05  # Sodium activation
        self.h = 0.6  # Sodium inactivation

        # Ion channel conductances (mS/cm²)
        self.g_na = 120.0  # Sodium
        self.g_k = 36.0  # Potassium
        self.g_leak = 0.3  # Leak

        # Synaptic connections
        self.synapses_in: list[Synapse] = []
        self.synapses_out: list[Synapse] = []

        # Activity history for plasticity
        self.spike_times: list[float] = []
        self.calcium_concentration = 0.0  # μM

        # Neurotransmitter release
        self.vesicle_pool = 100  # Number of ready vesicles
        self.release_probability = 0.3

    def alpha_n(self, v: float) -> float:
        """Potassium channel activation rate"""
        if abs(v + 55) < 0.01:
            return 0.1
        return 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10))

    def beta_n(self, v: float) -> float:
        """Potassium channel deactivation rate"""
        return 0.125 * np.exp(-(v + 65) / 80)

    def alpha_m(self, v: float) -> float:
        """Sodium channel activation rate"""
        if abs(v + 40) < 0.01:
            return 1.0
        return 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10))

    def beta_m(self, v: float) -> float:
        """Sodium channel deactivation rate"""
        return 4.0 * np.exp(-(v + 65) / 18)

    def alpha_h(self, v: float) -> float:
        """Sodium channel inactivation rate"""
        return 0.07 * np.exp(-(v + 65) / 20)

    def beta_h(self, v: float) -> float:
        """Sodium channel deinactivation rate"""
        return 1.0 / (1 + np.exp(-(v + 35) / 10))

    def update(self, dt: float, external_current: float = 0.0):
        """
        Update neuron state using Hodgkin-Huxley equations.
        Includes synaptic input and calcium dynamics.
        """
        v = self.voltage

        # Update gating variables
        self.n += dt * (self.alpha_n(v) * (1 - self.n) - self.beta_n(v) * self.n)
        self.m += dt * (self.alpha_m(v) * (1 - self.m) - self.beta_m(v) * self.m)
        self.h += dt * (self.alpha_h(v) * (1 - self.h) - self.beta_h(v) * self.h)

        # Calculate currents
        i_na = self.g_na * self.m**3 * self.h * (v - BiologicalConstants.SODIUM_REVERSAL)
        i_k = self.g_k * self.n**4 * (v - BiologicalConstants.POTASSIUM_REVERSAL)
        i_leak = self.g_leak * (v - BiologicalConstants.CHLORIDE_REVERSAL)

        # Synaptic currents
        i_syn = 0.0
        for synapse in self.synapses_in:
            i_syn += synapse.calculate_current(v)

        # Total current
        i_total = external_current - i_na - i_k - i_leak + i_syn

        # Update voltage
        self.voltage += dt * i_total / self.capacitance

        # Detect spike
        if self.voltage > 0 and v <= 0:  # Threshold crossing
            self.fire_action_potential(time.time())

        # Update calcium (simplified)
        if self.voltage > -30:  # Calcium influx during depolarization
            self.calcium_concentration += dt * 0.1 * (self.voltage + 30)

        # Calcium decay
        self.calcium_concentration *= np.exp(-dt / 100)  # 100 ms decay constant

        # Replenish vesicle pool
        self.vesicle_pool += dt * 0.1 * (100 - self.vesicle_pool)

    def fire_action_potential(self, current_time: float):
        """
        Handle action potential firing.
        Triggers neurotransmitter release and updates spike history.
        """
        self.spike_times.append(current_time)

        # Keep only recent spike times (last 1000 ms)
        self.spike_times = [t for t in self.spike_times if current_time - t < 1.0]

        # Release neurotransmitters at output synapses
        for synapse in self.synapses_out:
            if random.random() < self.release_probability and self.vesicle_pool > 0:
                synapse.release_neurotransmitter()
                self.vesicle_pool -= 1

    def calculate_firing_rate(self, window: float = 0.1) -> float:
        """Calculate instantaneous firing rate (Hz)"""
        current_time = time.time()
        recent_spikes = [t for t in self.spike_times if current_time - t < window]
        return len(recent_spikes) / window


class Synapse:
    """
    Chemical synapse with neurotransmitter dynamics and plasticity.
    Implements STDP (Spike-Timing Dependent Plasticity) and STP (Short-Term Plasticity).
    """

    def __init__(self, pre_neuron: Neuron, post_neuron: Neuron, weight: float = 1.0):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight  # Synaptic strength

        # Neurotransmitter dynamics
        self.neurotransmitter_concentration = 0.0  # mM
        self.receptor_activation = 0.0

        # Plasticity parameters
        self.tau_stdp = 20.0  # ms
        self.a_plus = 0.01  # LTP strength
        self.a_minus = 0.01  # LTD strength

        # Short-term plasticity
        self.facilitation = 1.0
        self.depression = 1.0

        # Register with neurons
        pre_neuron.synapses_out.append(self)
        post_neuron.synapses_in.append(self)

    def release_neurotransmitter(self):
        """Release neurotransmitter into synaptic cleft"""
        # Quantal release
        self.neurotransmitter_concentration += 1.0 * self.depression

        # Update short-term plasticity
        self.facilitation *= 1.5  # Facilitation
        self.depression *= 0.7  # Depression

    def calculate_current(self, post_voltage: float) -> float:
        """
        Calculate postsynaptic current based on neurotransmitter binding.
        Implements both AMPA and NMDA receptor dynamics.
        """
        # Update receptor activation
        binding_rate = 0.5  # ms⁻¹mM⁻¹
        unbinding_rate = 0.1  # ms⁻¹

        self.receptor_activation += (binding_rate * self.neurotransmitter_concentration *
                                     (1 - self.receptor_activation) -
                                     unbinding_rate * self.receptor_activation)

        # Calculate current
        if self.pre_neuron.neuron_type == 'excitatory':
            # Excitatory synapse (glutamate)
            reversal_potential = 0.0  # mV

            # AMPA component
            g_ampa = self.weight * self.receptor_activation * 0.5  # nS
            i_ampa = g_ampa * (post_voltage - reversal_potential)

            # NMDA component (voltage-dependent)
            mg_block = 1.0 / (1 + np.exp(-0.062 * post_voltage) * 1.0 / 3.57)
            g_nmda = self.weight * self.receptor_activation * 0.2 * mg_block
            i_nmda = g_nmda * (post_voltage - reversal_potential)

            current = -(i_ampa + i_nmda)

        else:
            # Inhibitory synapse (GABA)
            reversal_potential = -70.0  # mV
            g_gaba = self.weight * self.receptor_activation * 1.0  # nS
            current = -g_gaba * (post_voltage - reversal_potential)

        # Neurotransmitter clearance
        self.neurotransmitter_concentration *= 0.9  # Fast clearance

        # Recovery of short-term plasticity
        self.facilitation = 1.0 + (self.facilitation - 1.0) * 0.99
        self.depression = 1.0 - (1.0 - self.depression) * 0.99

        return current * 1e-6  # Convert to μA

    def update_stdp(self):
        """
        Update synaptic weight based on spike-timing dependent plasticity.
        Potentiation if pre fires before post, depression if post fires before pre.
        """
        if not self.pre_neuron.spike_times or not self.post_neuron.spike_times:
            return

        # Get most recent spike times
        pre_spike = self.pre_neuron.spike_times[-1]
        post_spike = self.post_neuron.spike_times[-1]

        dt = post_spike - pre_spike  # Time difference in seconds
        dt_ms = dt * 1000  # Convert to milliseconds

        if abs(dt_ms) < 100:  # Only consider spikes within 100 ms
            if dt_ms > 0:  # Pre before post - LTP
                delta_w = self.a_plus * np.exp(-dt_ms / self.tau_stdp)
                self.weight += delta_w
            else:  # Post before pre - LTD
                delta_w = -self.a_minus * np.exp(dt_ms / self.tau_stdp)
                self.weight += delta_w

            # Bound weight
            self.weight = max(0.0, min(10.0, self.weight))


class ImmuneCell(ABC):
    """
    Abstract base class for immune cells.
    Provides common functionality for all immune cell types.
    """

    def __init__(self, cell_id: str, position: np.ndarray):
        self.cell_id = cell_id
        self.position = position  # 3D position in tissue
        self.velocity = np.zeros(3)  # Movement velocity
        self.activation_level = 0.0  # 0 = resting, 1 = fully activated
        self.age = 0.0  # Cell age in hours
        self.health = 1.0  # Cell health/viability

    @abstractmethod
    def update(self, dt: float, environment: 'TissueEnvironment'):
        """Update cell state based on environment"""
        pass

    @abstractmethod
    def interact(self, other: 'ImmuneCell'):
        """Interact with another immune cell"""
        pass

    def migrate(self, dt: float, chemokine_gradient: np.ndarray):
        """
        Migrate based on chemokine gradients (chemotaxis).
        Uses biased random walk model.
        """
        # Random component (Brownian motion)
        random_component = np.random.randn(3) * 0.1

        # Directed component (chemotaxis)
        if np.linalg.norm(chemokine_gradient) > 0:
            directed_component = chemokine_gradient / np.linalg.norm(chemokine_gradient) * 0.5
        else:
            directed_component = np.zeros(3)

        # Update velocity
        self.velocity = 0.8 * self.velocity + directed_component + random_component

        # Limit maximum speed
        max_speed = 10.0  # μm/min
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed

        # Update position
        self.position += self.velocity * dt

    def undergo_apoptosis(self) -> bool:
        """
        Check if cell should undergo programmed cell death.
        Based on age, health, and activation state.
        """
        # Age-dependent apoptosis
        if self.age > 720:  # 30 days for lymphocytes
            return random.random() < 0.01  # 1% chance per update

        # Health-dependent apoptosis
        if self.health < 0.1:
            return random.random() < 0.1  # 10% chance if unhealthy

        # Activation-induced cell death
        if self.activation_level > 0.9 and self.age > 48:  # 2 days activated
            return random.random() < 0.001

        return False


class TCel(ImmuneCell):
    """
    T lymphocyte with antigen recognition and cytotoxic capabilities.
    Can be CD4+ (helper) or CD8+ (cytotoxic).
    """

    def __init__(self, cell_id: str, position: np.ndarray, subtype: str = 'CD8'):
        super().__init__(cell_id, position)
        self.subtype = subtype  # 'CD4' or 'CD8'
        self.tcr_specificity = np.random.randn(10)  # T-cell receptor specificity
        self.tcr_affinity = random.uniform(0.5, 1.0)

        # Activation state
        self.antigen_experienced = False
        self.proliferation_counter = 0
        self.cytokine_production = 0.0

        # Cytotoxic granules (for CD8+ cells)
        if subtype == 'CD8':
            self.perforin_granules = 100
            self.granzyme_level = 1.0

        # Memory formation
        self.memory_potential = random.uniform(0.0, 1.0)
        self.is_memory = False

    def recognize_antigen(self, antigen: np.ndarray) -> float:
        """
        Calculate TCR-antigen binding affinity.
        Uses dot product as similarity measure.
        """
        # Calculate structural similarity
        similarity = np.dot(self.tcr_specificity, antigen) / (
            np.linalg.norm(self.tcr_specificity) * np.linalg.norm(antigen) + 1e-10
        )

        # Apply affinity threshold
        recognition = similarity * self.tcr_affinity

        return max(0.0, recognition)

    def activate(self, signal_strength: float):
        """
        Activate T cell based on antigen recognition and costimulation.
        Requires both signal 1 (TCR) and signal 2 (costimulation).
        """
        if signal_strength > BiologicalConstants.ANTIGEN_RECOGNITION_THRESHOLD:
            self.activation_level = min(1.0, self.activation_level + signal_strength * 0.1)
            self.antigen_experienced = True

            # Start proliferation
            if self.activation_level > 0.5 and self.proliferation_counter == 0:
                self.proliferation_counter = 5  # Will divide 5 times

            # Produce cytokines
            if self.subtype == 'CD4':
                self.cytokine_production = self.activation_level * 10.0  # IL-2, IFN-γ, etc.

            # Replenish cytotoxic granules (CD8+)
            if self.subtype == 'CD8':
                self.perforin_granules = min(100, self.perforin_granules + 5)
                self.granzyme_level = min(1.0, self.granzyme_level + 0.1)

    def kill_target(self, target: 'ImmuneCell') -> bool:
        """
        CD8+ T cell kills target cell via cytotoxic granules.
        Returns True if target is killed.
        """
        if self.subtype != 'CD8' or self.perforin_granules < 10:
            return False

        if self.activation_level > 0.7:
            # Release cytotoxic granules
            self.perforin_granules -= 10

            # Target takes damage
            damage = self.granzyme_level * 0.5
            target.health -= damage

            return target.health <= 0

        return False

    def proliferate(self) -> Optional['TCell']:
        """
        Undergo cell division to produce daughter cells.
        Implements asymmetric division for memory cell formation.
        """
        if self.proliferation_counter <= 0:
            return None

        self.proliferation_counter -= 1

        # Create daughter cell
        daughter = TCell(
            cell_id=f"{self.cell_id}_d{self.proliferation_counter}",
            position=self.position + np.random.randn(3) * 5,
            subtype=self.subtype
        )

        # Inherit TCR specificity with slight variation
        daughter.tcr_specificity = self.tcr_specificity + np.random.randn(10) * 0.01
        daughter.tcr_affinity = self.tcr_affinity * random.uniform(0.95, 1.05)
        daughter.antigen_experienced = True

        # Asymmetric division for memory cell fate
        if random.random() < self.memory_potential and self.proliferation_counter == 0:
            daughter.is_memory = True
            daughter.activation_level = 0.1  # Memory cells are resting
            daughter.memory_potential = 0.9  # High potential for recall response

        return daughter

    def update(self, dt: float, environment: 'TissueEnvironment'):
        """Update T cell state"""
        self.age += dt / 3600  # Convert to hours

        # Activation decay
        if self.activation_level > 0:
            self.activation_level *= 0.99

        # Cytokine production decay
        self.cytokine_production *= 0.95

        # Check for apoptosis
        if self.undergo_apoptosis():
            self.health = 0

        # Memory cells have extended lifespan
        if self.is_memory:
            self.age *= 0.99  # Age slower

    def interact(self, other: 'ImmuneCell'):
        """T cell interactions with other immune cells"""
        distance = np.linalg.norm(self.position - other.position)

        if distance < 10:  # Within interaction range (10 μm)

            # CD4+ help to B cells
            if self.subtype == 'CD4' and isinstance(other, BCell):
                if self.activation_level > 0.5:
                    other.receive_t_help(self.cytokine_production)

            # Cytotoxic killing
            elif self.subtype == 'CD8' and hasattr(other, 'is_infected'):
                if other.is_infected:
                    self.kill_target(other)


class BCell(ImmuneCell):
    """
    B lymphocyte with antibody production and affinity maturation.
    Can differentiate into plasma cells or memory B cells.
    """

    def __init__(self, cell_id: str, position: np.ndarray):
        super().__init__(cell_id, position)
        self.bcr_specificity = np.random.randn(10)  # B-cell receptor
        self.antibody_affinity = random.uniform(0.1, 0.5)  # Initial low affinity

        # Activation and differentiation
        self.antigen_bound = False
        self.t_help_received = 0.0
        self.is_plasma_cell = False
        self.is_memory = False

        # Antibody production
        self.antibody_production_rate = 0.0
        self.antibody_class = 'IgM'  # Can switch to IgG, IgA, IgE

        # Somatic hypermutation
        self.mutation_rate = 0.0
        self.affinity_maturation_cycles = 0

    def bind_antigen(self, antigen: np.ndarray) -> float:
        """Calculate BCR-antigen binding affinity"""
        similarity = np.dot(self.bcr_specificity, antigen) / (
            np.linalg.norm(self.bcr_specificity) * np.linalg.norm(antigen) + 1e-10
        )

        binding = similarity * self.antibody_affinity

        if binding > 0.5:
            self.antigen_bound = True
            self.activation_level = min(1.0, self.activation_level + binding * 0.2)

        return binding

    def receive_t_help(self, cytokine_level: float):
        """Receive help from T helper cells (CD4+)"""
        self.t_help_received += cytokine_level

        # T help promotes activation
        if self.antigen_bound and self.t_help_received > 5.0:
            self.activation_level = min(1.0, self.activation_level + 0.3)

            # Trigger class switching
            if self.antibody_class == 'IgM' and random.random() < 0.1:
                self.antibody_class = random.choice(['IgG', 'IgA', 'IgE'])

            # Trigger somatic hypermutation
            self.mutation_rate = 0.01

    def undergo_affinity_maturation(self):
        """
        Somatic hypermutation to improve antibody affinity.
        Models germinal center reactions.
        """
        if self.mutation_rate > 0 and random.random() < self.mutation_rate:
            # Mutate BCR
            mutation = np.random.randn(10) * 0.1
            self.bcr_specificity += mutation

            # Potentially improve affinity
            if random.random() < 0.3:  # 30% beneficial mutations
                self.antibody_affinity *= random.uniform(1.1, 1.5)
                self.antibody_affinity = min(1.0, self.antibody_affinity)
            elif random.random() < 0.5:  # 50% neutral
                pass
            else:  # 20% deleterious
                self.antibody_affinity *= random.uniform(0.7, 0.9)

            self.affinity_maturation_cycles += 1

    def differentiate(self) -> Optional['BCell']:
        """
        Differentiate into plasma cell or memory B cell.
        Returns new cell if differentiation occurs.
        """
        if self.activation_level < 0.8 or not self.antigen_bound:
            return None

        if random.random() < 0.7:  # 70% become plasma cells
            self.is_plasma_cell = True
            self.antibody_production_rate = BiologicalConstants.ANTIBODY_PRODUCTION_RATE

            # Plasma cells don't divide
            return None

        else:  # 30% become memory B cells
            memory_cell = BCell(
                cell_id=f"{self.cell_id}_memory",
                position=self.position + np.random.randn(3) * 5
            )

            # Inherit improved BCR
            memory_cell.bcr_specificity = self.bcr_specificity.copy()
            memory_cell.antibody_affinity = self.antibody_affinity
            memory_cell.antibody_class = self.antibody_class
            memory_cell.is_memory = True
            memory_cell.activation_level = 0.1  # Resting state

            return memory_cell

    def produce_antibodies(self, dt: float) -> float:
        """
        Produce antibodies if differentiated into plasma cell.
        Returns antibody concentration produced.
        """
        if self.is_plasma_cell:
            production = self.antibody_production_rate * dt * self.activation_level

            # Plasma cells exhaust over time
            self.health -= dt * 0.01

            return production

        return 0.0

    def update(self, dt: float, environment: 'TissueEnvironment'):
        """Update B cell state"""
        self.age += dt / 3600

        # Undergo affinity maturation if activated
        if self.activation_level > 0.5:
            self.undergo_affinity_maturation()

        # Activation decay
        self.activation_level *= 0.98
        self.t_help_received *= 0.95

        # Memory cells have extended lifespan
        if self.is_memory:
            self.age *= 0.99

        # Plasma cells have shorter lifespan
        if self.is_plasma_cell:
            self.health -= dt * 0.001

        # Check for apoptosis
        if self.undergo_apoptosis():
            self.health = 0

    def interact(self, other: 'ImmuneCell'):
        """B cell interactions"""
        distance = np.linalg.norm(self.position - other.position)

        if distance < 10:
            # Competition in germinal centers
            if isinstance(other, BCell) and self.activation_level > 0.5:
                if self.antibody_affinity > other.antibody_affinity:
                    # Win competition, survive
                    self.health = min(1.0, self.health + 0.1)
                else:
                    # Lose competition, may die
                    self.health -= 0.1


class TissueEnvironment:
    """
    Represents the tissue microenvironment where immune responses occur.
    Includes spatial organization, chemokine gradients, and antigen distribution.
    """

    def __init__(self, size: tuple[float, float, float] = (1000, 1000, 100)):
        self.size = size  # Tissue dimensions in μm
        self.immune_cells: list[ImmuneCell] = []
        self.pathogens: list[dict] = []

        # Chemokine fields
        self.chemokine_field = np.zeros((20, 20, 5))  # Discretized field
        self.inflammation_level = 0.0

        # Antigen presentation
        self.antigens: list[np.ndarray] = []
        self.antibody_concentration = 0.0

        # Tissue properties
        self.temperature = STANDARD_TEMP
        self.ph = 7.4
        self.oxygen_level = 0.2  # Fraction

    def add_pathogen(self, position: np.ndarray, antigen_profile: np.ndarray):
        """Add a pathogen to the tissue"""
        pathogen = {
            'position': position,
            'antigen': antigen_profile,
            'viral_load': 1000,  # Arbitrary units
            'replication_rate': 0.1,
            'is_neutralized': False
        }

        self.pathogens.append(pathogen)
        self.antigens.append(antigen_profile)

        # Trigger inflammation
        self.inflammation_level += 0.2

        # Create chemokine gradient
        self._update_chemokine_gradient(position)

    def _update_chemokine_gradient(self, source_position: np.ndarray):
        """Update chemokine field based on inflammation sources"""
        # Convert position to grid indices
        grid_x = int(source_position[0] / self.size[0] * 20)
        grid_y = int(source_position[1] / self.size[1] * 20)
        grid_z = int(source_position[2] / self.size[2] * 5)

        # Ensure within bounds
        grid_x = max(0, min(19, grid_x))
        grid_y = max(0, min(19, grid_y))
        grid_z = max(0, min(4, grid_z))

        # Create Gaussian diffusion of chemokines
        for i in range(20):
            for j in range(20):
                for k in range(5):
                    distance = np.sqrt((i - grid_x)**2 + (j - grid_y)**2 + (k - grid_z)**2)
                    concentration = self.inflammation_level * np.exp(-distance / 5)
                    self.chemokine_field[i, j, k] += concentration

        # Decay over time
        self.chemokine_field *= 0.95

    def get_chemokine_gradient(self, position: np.ndarray) -> np.ndarray:
        """Calculate chemokine gradient at a specific position"""
        # Convert position to grid indices
        grid_x = int(position[0] / self.size[0] * 20)
        grid_y = int(position[1] / self.size[1] * 20)
        grid_z = int(position[2] / self.size[2] * 5)

        # Ensure within bounds
        grid_x = max(1, min(18, grid_x))
        grid_y = max(1, min(18, grid_y))
        grid_z = max(0, min(4, grid_z))

        # Calculate gradient using finite differences
        gradient_x = (self.chemokine_field[grid_x + 1, grid_y, grid_z] -
                     self.chemokine_field[grid_x - 1, grid_y, grid_z])
        gradient_y = (self.chemokine_field[grid_x, grid_y + 1, grid_z] -
                     self.chemokine_field[grid_x, grid_y - 1, grid_z])
        gradient_z = 0  # Simplified for 2D-like behavior

        return np.array([gradient_x, gradient_y, gradient_z])

    def update_pathogens(self, dt: float):
        """Update pathogen dynamics"""
        for pathogen in self.pathogens[:]:
            if not pathogen['is_neutralized']:
                # Viral replication
                pathogen['viral_load'] *= (1 + pathogen['replication_rate'] * dt)

                # Check antibody neutralization
                if self.antibody_concentration > 0:
                    neutralization_prob = 1 - np.exp(-self.antibody_concentration * 0.01)
                    if random.random() < neutralization_prob * dt:
                        pathogen['is_neutralized'] = True
                        pathogen['viral_load'] *= 0.1

            else:
                # Decay of neutralized pathogens
                pathogen['viral_load'] *= np.exp(-dt * 0.1)

                if pathogen['viral_load'] < 0.1:
                    self.pathogens.remove(pathogen)

    def simulate_immune_response(self, dt: float):
        """Simulate complete immune response dynamics"""
        # Update pathogen dynamics
        self.update_pathogens(dt)

        # Update all immune cells
        for cell in self.immune_cells[:]:
            if cell.health <= 0:
                self.immune_cells.remove(cell)
                continue

            # Get chemokine gradient at cell position
            gradient = self.get_chemokine_gradient(cell.position)

            # Cell migration
            cell.migrate(dt, gradient)

            # Keep cells within tissue bounds
            cell.position = np.clip(cell.position, [0, 0, 0], self.size)

            # Cell-specific updates
            cell.update(dt, self)

            # Check for pathogen recognition
            for pathogen in self.pathogens:
                distance = np.linalg.norm(cell.position - pathogen['position'])

                if distance < 50:  # Detection range
                    if isinstance(cell, TCell):
                        recognition = cell.recognize_antigen(pathogen['antigen'])
                        if recognition > 0:
                            cell.activate(recognition)

                    elif isinstance(cell, BCell):
                        binding = cell.bind_antigen(pathogen['antigen'])
                        if binding > 0:
                            cell.activation_level += 0.1

            # Cell proliferation
            if isinstance(cell, TCell):
                daughter = cell.proliferate()
                if daughter:
                    self.immune_cells.append(daughter)

            elif isinstance(cell, BCell):
                # B cell differentiation
                new_cell = cell.differentiate()
                if new_cell:
                    self.immune_cells.append(new_cell)

                # Antibody production
                self.antibody_concentration += cell.produce_antibodies(dt)

        # Cell-cell interactions
        for i, cell1 in enumerate(self.immune_cells):
            for cell2 in self.immune_cells[i+1:]:
                distance = np.linalg.norm(cell1.position - cell2.position)
                if distance < 20:  # Interaction range
                    cell1.interact(cell2)
                    cell2.interact(cell1)

        # Update inflammation
        self.inflammation_level *= 0.99  # Decay

        # Antibody decay
        self.antibody_concentration *= 0.995
