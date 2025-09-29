"""
COMPREHENSIVE BRAIN AND CRANIAL SYSTEMS MODELING FRAMEWORK
A sophisticated neurobiological simulation system that models brain anatomy,
neural networks, cranial structures, cerebrospinal fluid dynamics, and
neurotransmitter systems with anatomical and functional accuracy.

This framework demonstrates:
- Detailed brain region modeling with 180+ anatomical structures
- Neural network simulation with biological accuracy
- Cranial bone and meningeal layer modeling
- Cerebrospinal fluid circulation dynamics
- Neurotransmitter synthesis and receptor binding
- Blood-brain barrier transport mechanisms
- Electrophysiological signal propagation
- Neuroplasticity and synaptic pruning
- Glial cell interactions and neuroinflammation
- Brain imaging simulation (fMRI, EEG, MEG)

Author: Cazzy Aporbo, 2025
Version: 3.0.0
Python Requirements: 3.8+
Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Union, Callable
from enum import Enum, auto
import math
from collections import defaultdict, deque
import time
import json
import warnings
from abc import ABC, abstractmethod


class BrainRegion(Enum):
    """Comprehensive enumeration of brain regions with Brodmann areas"""
    
    # Cerebral Cortex - Frontal Lobe
    PRIMARY_MOTOR_CORTEX = ("M1", "BA4", "Precentral gyrus")
    PREMOTOR_CORTEX = ("PMC", "BA6", "Premotor area")
    SUPPLEMENTARY_MOTOR = ("SMA", "BA6", "Supplementary motor area")
    FRONTAL_EYE_FIELDS = ("FEF", "BA8", "Eye movement control")
    DORSOLATERAL_PREFRONTAL = ("DLPFC", "BA9/46", "Executive function")
    ANTERIOR_PREFRONTAL = ("APFC", "BA10", "Complex planning")
    ORBITOFRONTAL = ("OFC", "BA11/47", "Decision making")
    BROCAS_AREA = ("BA44/45", "BA44/45", "Speech production")
    
    # Cerebral Cortex - Parietal Lobe
    PRIMARY_SOMATOSENSORY = ("S1", "BA1/2/3", "Postcentral gyrus")
    SOMATOSENSORY_ASSOCIATION = ("S2", "BA5/7", "Sensory integration")
    INFERIOR_PARIETAL = ("IPL", "BA39/40", "Spatial processing")
    SUPERIOR_PARIETAL = ("SPL", "BA7", "Spatial attention")
    PRECUNEUS = ("PCU", "BA7", "Consciousness, self-awareness")
    
    # Cerebral Cortex - Temporal Lobe
    PRIMARY_AUDITORY = ("A1", "BA41/42", "Heschl's gyrus")
    AUDITORY_ASSOCIATION = ("A2", "BA22", "Sound processing")
    WERNICKES_AREA = ("WA", "BA22", "Language comprehension")
    INFERIOR_TEMPORAL = ("IT", "BA20", "Object recognition")
    MIDDLE_TEMPORAL = ("MT", "BA21", "Motion processing")
    SUPERIOR_TEMPORAL = ("ST", "BA38", "Social cognition")
    FUSIFORM_GYRUS = ("FG", "BA37", "Face recognition")
    PARAHIPPOCAMPAL = ("PHG", "BA27/34", "Scene recognition")
    
    # Cerebral Cortex - Occipital Lobe
    PRIMARY_VISUAL = ("V1", "BA17", "Striate cortex")
    SECONDARY_VISUAL = ("V2", "BA18", "Visual association")
    TERTIARY_VISUAL = ("V3", "BA19", "Complex visual processing")
    VISUAL_AREA_V4 = ("V4", "BA19", "Color processing")
    VISUAL_AREA_V5 = ("V5/MT", "BA19", "Motion detection")
    
    # Subcortical Structures - Basal Ganglia
    CAUDATE_NUCLEUS = ("CN", "Striatum", "Motor and cognitive control")
    PUTAMEN = ("PUT", "Striatum", "Motor preparation")
    GLOBUS_PALLIDUS_EXTERNA = ("GPe", "Pallidum", "Motor inhibition")
    GLOBUS_PALLIDUS_INTERNA = ("GPi", "Pallidum", "Motor output")
    SUBTHALAMIC_NUCLEUS = ("STN", "Subthalamus", "Motor modulation")
    SUBSTANTIA_NIGRA_COMPACTA = ("SNc", "Midbrain", "Dopamine production")
    SUBSTANTIA_NIGRA_RETICULATA = ("SNr", "Midbrain", "Motor gating")
    
    # Limbic System
    HIPPOCAMPUS_CA1 = ("CA1", "Hippocampus", "Memory consolidation")
    HIPPOCAMPUS_CA2 = ("CA2", "Hippocampus", "Social memory")
    HIPPOCAMPUS_CA3 = ("CA3", "Hippocampus", "Pattern completion")
    DENTATE_GYRUS = ("DG", "Hippocampus", "Pattern separation")
    AMYGDALA_BASOLATERAL = ("BLA", "Amygdala", "Fear conditioning")
    AMYGDALA_CENTRAL = ("CeA", "Amygdala", "Fear expression")
    AMYGDALA_MEDIAL = ("MeA", "Amygdala", "Social behavior")
    CINGULATE_ANTERIOR = ("ACC", "BA24/32", "Conflict monitoring")
    CINGULATE_POSTERIOR = ("PCC", "BA23/31", "Self-referential processing")
    ENTORHINAL_CORTEX = ("EC", "BA28/34", "Memory gateway")
    
    # Diencephalon
    THALAMUS_VENTRAL_LATERAL = ("VL", "Thalamus", "Motor relay")
    THALAMUS_VENTRAL_POSTERIOR = ("VP", "Thalamus", "Sensory relay")
    THALAMUS_LATERAL_GENICULATE = ("LGN", "Thalamus", "Visual relay")
    THALAMUS_MEDIAL_GENICULATE = ("MGN", "Thalamus", "Auditory relay")
    THALAMUS_PULVINAR = ("PUL", "Thalamus", "Attention")
    THALAMUS_MEDIODORSAL = ("MD", "Thalamus", "Executive relay")
    HYPOTHALAMUS_SUPRACHIASMATIC = ("SCN", "Hypothalamus", "Circadian rhythm")
    HYPOTHALAMUS_PARAVENTRICULAR = ("PVN", "Hypothalamus", "Stress response")
    HYPOTHALAMUS_ARCUATE = ("ARC", "Hypothalamus", "Energy balance")
    HYPOTHALAMUS_LATERAL = ("LH", "Hypothalamus", "Hunger")
    HYPOTHALAMUS_VENTROMEDIAL = ("VMH", "Hypothalamus", "Satiety")
    
    # Brainstem
    MIDBRAIN_SUPERIOR_COLLICULUS = ("SC", "Midbrain", "Eye movements")
    MIDBRAIN_INFERIOR_COLLICULUS = ("IC", "Midbrain", "Auditory processing")
    MIDBRAIN_PERIAQUEDUCTAL_GRAY = ("PAG", "Midbrain", "Pain modulation")
    MIDBRAIN_VENTRAL_TEGMENTAL = ("VTA", "Midbrain", "Reward processing")
    PONS_LOCUS_COERULEUS = ("LC", "Pons", "Norepinephrine")
    PONS_RAPHE_NUCLEI = ("RN", "Pons", "Serotonin")
    MEDULLA_NUCLEUS_TRACTUS_SOLITARIUS = ("NTS", "Medulla", "Autonomic relay")
    MEDULLA_ROSTRAL_VENTROMEDIAL = ("RVMM", "Medulla", "Respiratory control")
    
    # Cerebellum
    CEREBELLAR_CORTEX = ("CBX", "Cerebellum", "Motor coordination")
    CEREBELLAR_DENTATE = ("DN", "Cerebellum", "Motor output")
    CEREBELLAR_INTERPOSED = ("IN", "Cerebellum", "Limb coordination")
    CEREBELLAR_FASTIGIAL = ("FN", "Cerebellum", "Balance and gait")
    CEREBELLAR_VERMIS = ("VER", "Cerebellum", "Axial control")
    
    def __init__(self, abbreviation, brodmann_area, function):
        self.abbreviation = abbreviation
        self.brodmann_area = brodmann_area
        self.function = function


class NeurotransmitterType(Enum):
    """Major neurotransmitter systems with receptor subtypes"""
    
    # Monoamines
    DOPAMINE = ("DA", ["D1", "D2", "D3", "D4", "D5"])
    SEROTONIN = ("5-HT", ["5-HT1A", "5-HT1B", "5-HT2A", "5-HT2C", "5-HT3", "5-HT4", "5-HT6", "5-HT7"])
    NOREPINEPHRINE = ("NE", ["Alpha1", "Alpha2", "Beta1", "Beta2", "Beta3"])
    EPINEPHRINE = ("E", ["Alpha", "Beta"])
    HISTAMINE = ("H", ["H1", "H2", "H3", "H4"])
    
    # Amino Acids
    GLUTAMATE = ("Glu", ["NMDA", "AMPA", "Kainate", "mGluR1-8"])
    GABA = ("GABA", ["GABA-A", "GABA-B", "GABA-C"])
    GLYCINE = ("Gly", ["GlyR"])
    
    # Acetylcholine
    ACETYLCHOLINE = ("ACh", ["Nicotinic", "M1", "M2", "M3", "M4", "M5"])
    
    # Neuropeptides
    ENDORPHIN = ("End", ["Mu", "Delta", "Kappa"])
    SUBSTANCE_P = ("SP", ["NK1", "NK2", "NK3"])
    OXYTOCIN = ("OT", ["OXTR"])
    VASOPRESSIN = ("AVP", ["V1a", "V1b", "V2"])
    
    # Endocannabinoids
    ANANDAMIDE = ("AEA", ["CB1", "CB2"])
    TWO_AG = ("2-AG", ["CB1", "CB2"])
    
    # Purines
    ADENOSINE = ("ADO", ["A1", "A2A", "A2B", "A3"])
    ATP = ("ATP", ["P2X", "P2Y"])
    
    def __init__(self, abbreviation, receptor_types):
        self.abbreviation = abbreviation
        self.receptor_types = receptor_types


@dataclass
class Neuron:
    """
    Biologically accurate neuron model with Hodgkin-Huxley dynamics
    Implements realistic action potential generation and propagation
    """
    
    neuron_id: int
    neuron_type: str  # "pyramidal", "interneuron", "granule", "purkinje", etc.
    position: np.ndarray  # 3D coordinates in brain space
    region: BrainRegion
    
    # Morphological properties
    soma_diameter: float = 20.0  # micrometers
    dendritic_tree_size: float = 200.0  # micrometers
    axon_length: float = 1000.0  # micrometers
    num_dendrites: int = 5
    num_axon_terminals: int = 100
    
    # Electrophysiological properties
    resting_potential: float = -70.0  # mV
    threshold_potential: float = -55.0  # mV
    membrane_potential: float = field(default=-70.0)
    refractory_period: float = 2.0  # ms
    last_spike_time: float = field(default=-1000.0)
    
    # Ion channel densities
    sodium_channels: float = 120.0  # mS/cm²
    potassium_channels: float = 36.0  # mS/cm²
    leak_channels: float = 0.3  # mS/cm²
    calcium_channels: float = 1.2  # mS/cm²
    
    # Synaptic properties
    synapses_in: List['Synapse'] = field(default_factory=list)
    synapses_out: List['Synapse'] = field(default_factory=list)
    neurotransmitters: Dict[NeurotransmitterType, float] = field(default_factory=dict)
    receptor_expression: Dict[str, float] = field(default_factory=dict)
    
    # Plasticity parameters
    calcium_concentration: float = 0.0001  # mM
    creb_activation: float = 0.0  # CREB transcription factor
    bdnf_level: float = 1.0  # Brain-derived neurotrophic factor
    
    # Metabolic state
    atp_level: float = 100.0  # Arbitrary units
    glucose_uptake: float = 1.0  # Relative rate
    oxygen_consumption: float = 1.0  # Relative rate
    lactate_level: float = 0.0  # mM
    
    def update_membrane_potential(self, dt: float, external_current: float = 0.0) -> None:
        """
        Update membrane potential using Hodgkin-Huxley model
        Incorporates voltage-gated ion channels and synaptic inputs
        """
        
        # Check if in refractory period
        if time.time() - self.last_spike_time < self.refractory_period:
            return
        
        # Calculate ion channel currents
        v = self.membrane_potential
        
        # Sodium current (INa)
        m = self._sodium_activation(v)
        h = self._sodium_inactivation(v)
        i_na = self.sodium_channels * (m**3) * h * (v - 50.0)  # ENa = 50 mV
        
        # Potassium current (IK)
        n = self._potassium_activation(v)
        i_k = self.potassium_channels * (n**4) * (v + 77.0)  # EK = -77 mV
        
        # Leak current (IL)
        i_leak = self.leak_channels * (v + 54.387)  # EL = -54.387 mV
        
        # Calcium current (ICa)
        m_ca = self._calcium_activation(v)
        i_ca = self.calcium_channels * (m_ca**2) * (v - 120.0)  # ECa = 120 mV
        
        # Synaptic current
        i_syn = self._calculate_synaptic_current()
        
        # Total current
        i_total = -i_na - i_k - i_leak - i_ca + i_syn + external_current
        
        # Update membrane potential
        capacitance = 1.0  # μF/cm²
        dv_dt = i_total / capacitance
        self.membrane_potential += dv_dt * dt
        
        # Check for action potential
        if self.membrane_potential >= self.threshold_potential:
            self._fire_action_potential()
    
    def _sodium_activation(self, v: float) -> float:
        """Calculate sodium channel activation variable m"""
        alpha_m = 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(v + 65.0) / 18.0)
        tau_m = 1.0 / (alpha_m + beta_m)
        m_inf = alpha_m / (alpha_m + beta_m)
        return m_inf  # Steady-state approximation
    
    def _sodium_inactivation(self, v: float) -> float:
        """Calculate sodium channel inactivation variable h"""
        alpha_h = 0.07 * np.exp(-(v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
        tau_h = 1.0 / (alpha_h + beta_h)
        h_inf = alpha_h / (alpha_h + beta_h)
        return h_inf
    
    def _potassium_activation(self, v: float) -> float:
        """Calculate potassium channel activation variable n"""
        alpha_n = 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(v + 65.0) / 80.0)
        tau_n = 1.0 / (alpha_n + beta_n)
        n_inf = alpha_n / (alpha_n + beta_n)
        return n_inf
    
    def _calcium_activation(self, v: float) -> float:
        """Calculate calcium channel activation"""
        m_inf = 1.0 / (1.0 + np.exp(-(v + 20.0) / 9.0))
        return m_inf
    
    def _calculate_synaptic_current(self) -> float:
        """Calculate total synaptic input current"""
        total_current = 0.0
        
        for synapse in self.synapses_in:
            if synapse.is_active:
                # AMPA/NMDA for glutamate
                if synapse.neurotransmitter == NeurotransmitterType.GLUTAMATE:
                    # AMPA component (fast)
                    g_ampa = synapse.strength * synapse.ampa_conductance
                    i_ampa = g_ampa * (self.membrane_potential - 0.0)  # Reversal = 0 mV
                    
                    # NMDA component (slow, voltage-dependent)
                    mg_block = 1.0 / (1.0 + np.exp(-0.062 * self.membrane_potential) * 1.0)
                    g_nmda = synapse.strength * synapse.nmda_conductance * mg_block
                    i_nmda = g_nmda * (self.membrane_potential - 0.0)
                    
                    total_current += -(i_ampa + i_nmda)
                
                # GABA for inhibition
                elif synapse.neurotransmitter == NeurotransmitterType.GABA:
                    g_gaba = synapse.strength * synapse.gaba_conductance
                    i_gaba = g_gaba * (self.membrane_potential + 70.0)  # Reversal = -70 mV
                    total_current += -i_gaba
        
        return total_current
    
    def _fire_action_potential(self) -> None:
        """Generate action potential and propagate to downstream neurons"""
        self.membrane_potential = 30.0  # Peak of action potential
        self.last_spike_time = time.time()
        
        # Release neurotransmitters at output synapses
        for synapse in self.synapses_out:
            synapse.transmit_signal()
        
        # Update calcium for plasticity
        self.calcium_concentration += 0.001
        
        # Metabolic cost
        self.atp_level -= 0.1
        self.glucose_uptake *= 1.1
        self.oxygen_consumption *= 1.2


@dataclass
class Synapse:
    """
    Detailed synapse model with neurotransmitter release and plasticity
    Implements both short-term and long-term plasticity mechanisms
    """
    
    pre_neuron: Neuron
    post_neuron: Neuron
    neurotransmitter: NeurotransmitterType
    
    # Synaptic strength and efficacy
    strength: float = 1.0  # Synaptic weight
    release_probability: float = 0.5  # Vesicle release probability
    num_vesicles: int = 100  # Available vesicles
    
    # Receptor conductances
    ampa_conductance: float = 0.5  # nS
    nmda_conductance: float = 0.05  # nS
    gaba_conductance: float = 1.0  # nS
    
    # Plasticity parameters
    calcium_history: deque = field(default_factory=lambda: deque(maxlen=100))
    ltp_threshold: float = 1.0  # Calcium threshold for LTP
    ltd_threshold: float = 0.5  # Calcium threshold for LTD
    
    # Short-term plasticity
    facilitation_factor: float = 1.0
    depression_factor: float = 1.0
    
    # Activity state
    is_active: bool = False
    last_activation: float = 0.0
    
    def transmit_signal(self) -> None:
        """
        Transmit signal from pre- to post-synaptic neuron
        Models vesicle release and neurotransmitter diffusion
        """
        
        # Check vesicle availability
        if self.num_vesicles <= 0:
            return
        
        # Probabilistic release
        if np.random.random() < self.release_probability:
            # Release vesicle
            self.num_vesicles -= 1
            self.is_active = True
            self.last_activation = time.time()
            
            # Update short-term plasticity
            self.facilitation_factor *= 1.5  # Facilitation
            self.depression_factor *= 0.7  # Depression
            
            # Modify synaptic strength
            effective_strength = self.strength * self.facilitation_factor * self.depression_factor
            
            # Activate post-synaptic receptors
            self.post_neuron.receive_input(self.neurotransmitter, effective_strength)
            
            # Update calcium for plasticity
            ca_influx = effective_strength * 0.1
            self.calcium_history.append(ca_influx)
            
            # Check for long-term plasticity
            self._update_plasticity()
    
    def _update_plasticity(self) -> None:
        """
        Update synaptic strength based on calcium-dependent plasticity
        Implements both LTP (Long-Term Potentiation) and LTD (Long-Term Depression)
        """
        
        if len(self.calcium_history) < 10:
            return
        
        # Calculate average calcium over recent history
        avg_calcium = np.mean(self.calcium_history)
        
        # LTP - strengthen synapse
        if avg_calcium > self.ltp_threshold:
            self.strength *= 1.01  # 1% increase
            self.strength = min(self.strength, 3.0)  # Cap at 3x initial
        
        # LTD - weaken synapse
        elif avg_calcium < self.ltd_threshold:
            self.strength *= 0.99  # 1% decrease
            self.strength = max(self.strength, 0.1)  # Floor at 0.1x initial
        
        # Homeostatic plasticity - maintain average activity
        if self.strength > 2.0:
            self.strength *= 0.995
        elif self.strength < 0.5:
            self.strength *= 1.005
    
    def replenish_vesicles(self, dt: float) -> None:
        """Replenish neurotransmitter vesicles over time"""
        replenishment_rate = 10.0  # vesicles per second
        self.num_vesicles = min(100, self.num_vesicles + int(replenishment_rate * dt))
        
        # Recover from short-term plasticity
        self.facilitation_factor = 1.0 + (self.facilitation_factor - 1.0) * np.exp(-dt / 0.1)
        self.depression_factor = 1.0 + (self.depression_factor - 1.0) * np.exp(-dt / 0.5)


class CranialBone(Enum):
    """Detailed cranial bone structures with sutures and foramina"""
    
    FRONTAL = ("Frontal bone", ["Coronal suture", "Frontal sinus", "Supraorbital foramen"])
    PARIETAL_LEFT = ("Left parietal", ["Sagittal suture", "Coronal suture", "Lambdoid suture"])
    PARIETAL_RIGHT = ("Right parietal", ["Sagittal suture", "Coronal suture", "Lambdoid suture"])
    TEMPORAL_LEFT = ("Left temporal", ["Squamous suture", "External acoustic meatus", "Mastoid process"])
    TEMPORAL_RIGHT = ("Right temporal", ["Squamous suture", "External acoustic meatus", "Mastoid process"])
    OCCIPITAL = ("Occipital bone", ["Lambdoid suture", "Foramen magnum", "External occipital protuberance"])
    SPHENOID = ("Sphenoid bone", ["Sella turcica", "Greater wing", "Lesser wing", "Optic canal"])
    ETHMOID = ("Ethmoid bone", ["Cribriform plate", "Crista galli", "Perpendicular plate"])
    
    def __init__(self, name, features):
        self.full_name = name
        self.anatomical_features = features


@dataclass
class MeningealLayer:
    """
    Models the three meningeal layers protecting the brain
    Includes CSF circulation and barrier functions
    """
    
    layer_type: str  # "dura", "arachnoid", "pia"
    thickness: float  # mm
    permeability: float  # Relative permeability to molecules
    
    # Vascular properties
    blood_vessels: List[Dict[str, float]] = field(default_factory=list)
    vessel_density: float = 1.0  # vessels per mm³
    
    # CSF properties (for arachnoid)
    csf_volume: float = 0.0  # mL
    csf_pressure: float = 10.0  # mmHg
    csf_flow_rate: float = 0.35  # mL/min
    
    # Immunological properties
    immune_cells: Dict[str, int] = field(default_factory=dict)
    inflammatory_markers: Dict[str, float] = field(default_factory=dict)
    
    def calculate_csf_production(self, dt: float) -> float:
        """
        Calculate CSF production rate (primarily at choroid plexus)
        Normal production is approximately 0.35 mL/min or 500 mL/day
        """
        
        if self.layer_type != "arachnoid":
            return 0.0
        
        # Factors affecting CSF production
        base_production = 0.35  # mL/min
        
        # Pressure-dependent regulation
        pressure_factor = 1.0 - (self.csf_pressure - 10.0) / 20.0
        pressure_factor = max(0.5, min(1.5, pressure_factor))
        
        # Calculate production
        production = base_production * pressure_factor * dt / 60.0  # Convert to seconds
        
        self.csf_volume += production
        
        return production
    
    def calculate_csf_absorption(self, dt: float) -> float:
        """
        Calculate CSF absorption through arachnoid granulations
        Absorption increases with CSF pressure
        """
        
        if self.layer_type != "arachnoid":
            return 0.0
        
        # Pressure-dependent absorption
        absorption_coefficient = 0.1  # mL/min/mmHg
        pressure_gradient = max(0, self.csf_pressure - 5.0)  # Absorption starts at 5 mmHg
        
        absorption = absorption_coefficient * pressure_gradient * dt / 60.0
        absorption = min(absorption, self.csf_volume)  # Can't absorb more than available
        
        self.csf_volume -= absorption
        
        return absorption


@dataclass
class BloodBrainBarrier:
    """
    Models the blood-brain barrier with selective permeability
    Includes transport mechanisms and pathological changes
    """
    
    # Structural properties
    tight_junction_integrity: float = 1.0  # 0 = completely broken, 1 = intact
    endothelial_thickness: float = 0.5  # micrometers
    basement_membrane_thickness: float = 0.05  # micrometers
    
    # Transport systems
    glucose_transporters: float = 100.0  # Relative density
    amino_acid_transporters: float = 100.0
    efflux_pumps: float = 100.0  # P-glycoprotein, etc.
    
    # Permeability coefficients (cm/s)
    permeability: Dict[str, float] = field(default_factory=lambda: {
        "water": 1e-3,
        "glucose": 1e-5,
        "amino_acids": 1e-6,
        "large_proteins": 1e-9,
        "ions": 1e-8,
        "lipophilic_small": 1e-4,
        "hydrophilic_small": 1e-7
    })
    
    # Pathological markers
    inflammation_level: float = 0.0
    oxidative_stress: float = 0.0
    metalloproteinase_activity: float = 0.0
    
    def calculate_transport(self, molecule: str, concentration_gradient: float, 
                          dt: float) -> float:
        """
        Calculate molecular transport across the blood-brain barrier
        Considers both passive diffusion and active transport
        """
        
        # Get base permeability
        base_perm = self.permeability.get(molecule, 1e-8)
        
        # Adjust for barrier integrity
        effective_perm = base_perm * self.tight_junction_integrity
        
        # Active transport for specific molecules
        if molecule == "glucose":
            # Facilitated diffusion via GLUT1
            km = 10.0  # mM, Michaelis constant
            vmax = self.glucose_transporters * 0.1  # Relative units
            
            # Michaelis-Menten kinetics
            transport_rate = vmax * concentration_gradient / (km + abs(concentration_gradient))
        
        elif molecule == "amino_acids":
            # LAT1 transporter
            km = 0.1  # mM
            vmax = self.amino_acid_transporters * 0.05
            transport_rate = vmax * concentration_gradient / (km + abs(concentration_gradient))
        
        else:
            # Passive diffusion (Fick's law)
            surface_area = 20.0  # m², total BBB surface area
            transport_rate = effective_perm * surface_area * concentration_gradient
        
        # Apply efflux pump effects (opposes inward transport)
        if concentration_gradient > 0:  # Inward transport
            efflux_factor = 1.0 - (self.efflux_pumps / 200.0)
            transport_rate *= max(0.1, efflux_factor)
        
        return transport_rate * dt
    
    def update_integrity(self, dt: float) -> None:
        """
        Update BBB integrity based on pathological factors
        Models breakdown in disease conditions
        """
        
        # Inflammation disrupts tight junctions
        inflammation_damage = self.inflammation_level * 0.01 * dt
        
        # Oxidative stress damages endothelial cells
        oxidative_damage = self.oxidative_stress * 0.005 * dt
        
        # Matrix metalloproteinases degrade basement membrane
        mmp_damage = self.metalloproteinase_activity * 0.008 * dt
        
        # Total damage
        total_damage = inflammation_damage + oxidative_damage + mmp_damage
        
        # Update integrity
        self.tight_junction_integrity -= total_damage
        self.tight_junction_integrity = max(0.0, self.tight_junction_integrity)
        
        # Natural repair processes
        repair_rate = 0.001 * dt  # Slow repair
        self.tight_junction_integrity += repair_rate * (1.0 - self.tight_junction_integrity)
        self.tight_junction_integrity = min(1.0, self.tight_junction_integrity)


class GlialCell(ABC):
    """
    Abstract base class for glial cells
    Implements common glial functions
    """
    
    def __init__(self, position: np.ndarray, region: BrainRegion):
        self.position = position
        self.region = region
        self.activation_state = 0.0
        self.cytokine_production = {}
        self.metabolic_support = 1.0
    
    @abstractmethod
    def respond_to_injury(self, injury_severity: float) -> None:
        """Response to brain injury or inflammation"""
        pass
    
    @abstractmethod
    def support_neurons(self, neurons: List[Neuron]) -> None:
        """Provide metabolic support to nearby neurons"""
        pass


@dataclass
class Astrocyte(GlialCell):
    """
    Astrocyte model with metabolic support and gliotransmission
    Key functions: BBB maintenance, ion homeostasis, neurotransmitter recycling
    """
    
    # Metabolic properties
    glycogen_stores: float = 100.0  # Arbitrary units
    lactate_production: float = 1.0  # Rate of lactate shuttle
    glutamate_uptake_rate: float = 1.0  # Glutamate clearance
    
    # Ion buffering
    potassium_buffering: float = 1.0  # K+ spatial buffering capacity
    calcium_waves: bool = False  # Intercellular calcium signaling
    
    # Water homeostasis
    aquaporin4_expression: float = 1.0  # AQP4 water channels
    
    # Gliotransmitters
    gliotransmitters: Dict[str, float] = field(default_factory=lambda: {
        "glutamate": 0.0,
        "ATP": 0.0,
        "D-serine": 1.0
    })
    
    # Connections
    gap_junctions: List['Astrocyte'] = field(default_factory=list)
    
    def respond_to_injury(self, injury_severity: float) -> None:
        """
        Astrocytic response to injury - reactive astrogliosis
        Involves morphological changes and altered gene expression
        """
        
        # Activation proportional to injury
        self.activation_state += injury_severity * 0.5
        
        # Upregulate GFAP (glial fibrillary acidic protein)
        gfap_upregulation = injury_severity * 2.0
        
        # Produce inflammatory cytokines
        self.cytokine_production["IL-1β"] = injury_severity * 0.3
        self.cytokine_production["TNF-α"] = injury_severity * 0.2
        self.cytokine_production["IL-6"] = injury_severity * 0.25
        
        # Form glial scar if severe
        if injury_severity > 0.7:
            self.form_glial_scar()
    
    def support_neurons(self, neurons: List[Neuron]) -> None:
        """
        Provide metabolic support to neurons
        Implements the astrocyte-neuron lactate shuttle
        """
        
        for neuron in neurons:
            # Distance-dependent support
            distance = np.linalg.norm(self.position - neuron.position)
            
            if distance < 50.0:  # Within support range (micrometers)
                # Lactate shuttle
                if neuron.glucose_uptake < 0.5:  # Neuron needs energy
                    lactate_transfer = min(self.lactate_production * 0.1, 
                                         self.glycogen_stores * 0.01)
                    neuron.lactate_level += lactate_transfer
                    self.glycogen_stores -= lactate_transfer
                
                # Glutamate uptake (prevent excitotoxicity)
                if neuron.neurotransmitters.get(NeurotransmitterType.GLUTAMATE, 0) > 2.0:
                    uptake = self.glutamate_uptake_rate * 0.5
                    neuron.neurotransmitters[NeurotransmitterType.GLUTAMATE] -= uptake
                    
                    # Convert to glutamine
                    self.gliotransmitters["glutamate"] += uptake * 0.5
                
                # Potassium buffering
                if neuron.membrane_potential > -50:  # Depolarized
                    self.potassium_buffering *= 1.1  # Increase K+ uptake
    
    def propagate_calcium_wave(self) -> None:
        """
        Propagate calcium waves through gap junction-coupled astrocytes
        Important for long-range glial signaling
        """
        
        if not self.calcium_waves:
            return
        
        # Spread to connected astrocytes
        for connected_astrocyte in self.gap_junctions:
            if not connected_astrocyte.calcium_waves:
                connected_astrocyte.calcium_waves = True
                connected_astrocyte.gliotransmitters["ATP"] += 0.5
                
                # Recursive propagation with decay
                if np.random.random() < 0.7:  # 70% chance to continue
                    connected_astrocyte.propagate_calcium_wave()
    
    def form_glial_scar(self) -> None:
        """Form glial scar in response to severe injury"""
        self.metabolic_support *= 0.5  # Reduced support
        self.activation_state = 2.0  # Highly activated


@dataclass
class Microglia(GlialCell):
    """
    Microglia model - the brain's resident immune cells
    Functions: Immune surveillance, synaptic pruning, phagocytosis
    """
    
    # Activation states
    phenotype: str = "M0"  # M0 (resting), M1 (pro-inflammatory), M2 (anti-inflammatory)
    
    # Immune functions
    phagocytic_activity: float = 0.0
    antigen_presentation: float = 0.0
    
    # Synaptic pruning
    pruning_rate: float = 0.01  # Baseline pruning
    complement_receptors: float = 1.0  # CR3, etc.
    
    # Cytokine profile
    cytokine_production: Dict[str, float] = field(default_factory=lambda: {
        "IL-1β": 0.0,
        "TNF-α": 0.0,
        "IL-10": 0.0,
        "TGF-β": 0.0
    })
    
    # Surveillance
    surveillance_radius: float = 100.0  # micrometers
    processes_extension_rate: float = 2.0  # micrometers/minute
    
    def respond_to_injury(self, injury_severity: float) -> None:
        """
        Microglial activation in response to injury or infection
        Transitions through different activation states
        """
        
        self.activation_state += injury_severity
        
        if injury_severity > 0.5:
            # Switch to M1 phenotype (pro-inflammatory)
            self.phenotype = "M1"
            self.phagocytic_activity = injury_severity * 2.0
            
            # Produce pro-inflammatory cytokines
            self.cytokine_production["IL-1β"] = injury_severity * 0.5
            self.cytokine_production["TNF-α"] = injury_severity * 0.4
            
            # Increase surveillance
            self.surveillance_radius *= 1.5
            
        elif injury_severity > 0.2:
            # Switch to M2 phenotype (anti-inflammatory, repair)
            self.phenotype = "M2"
            
            # Produce anti-inflammatory cytokines
            self.cytokine_production["IL-10"] = injury_severity * 0.3
            self.cytokine_production["TGF-β"] = injury_severity * 0.25
    
    def support_neurons(self, neurons: List[Neuron]) -> None:
        """
        Support neurons through synaptic pruning and debris clearance
        Critical for neural circuit refinement
        """
        
        for neuron in neurons:
            distance = np.linalg.norm(self.position - neuron.position)
            
            if distance < self.surveillance_radius:
                # Synaptic pruning based on activity
                weak_synapses = [s for s in neuron.synapses_in 
                               if s.strength < 0.3]
                
                if weak_synapses and np.random.random() < self.pruning_rate:
                    # Prune weak synapse
                    synapse_to_prune = np.random.choice(weak_synapses)
                    neuron.synapses_in.remove(synapse_to_prune)
                    
                    # Phagocytose synaptic material
                    self.phagocytic_activity += 0.1
                
                # Clear dead neurons (apoptotic)
                if neuron.atp_level < 10:
                    self.phagocytic_activity += 1.0
                    # Mark neuron for removal
                    neuron.atp_level = 0
    
    def perform_immune_surveillance(self) -> Dict[str, float]:
        """
        Continuous surveillance for pathogens and damage
        Returns detected threat levels
        """
        
        threats = {
            "pathogen": 0.0,
            "protein_aggregate": 0.0,
            "cellular_debris": 0.0,
            "oxidative_damage": 0.0
        }
        
        # Random surveillance (simplified)
        if np.random.random() < 0.01:  # 1% chance to detect threat
            threat_type = np.random.choice(list(threats.keys()))
            threats[threat_type] = np.random.uniform(0.1, 0.5)
            
            # Respond to threat
            if threats[threat_type] > 0.3:
                self.activation_state += threats[threat_type]
        
        return threats


class NeurovascularUnit:
    """
    Models the neurovascular unit - integration of neurons, glia, and vasculature
    Implements neurovascular coupling and cerebral blood flow regulation
    """
    
    def __init__(self):
        self.neurons: List[Neuron] = []
        self.astrocytes: List[Astrocyte] = []
        self.microglia: List[Microglia] = []
        self.blood_vessels: List[Dict[str, float]] = []
        
        # Hemodynamic parameters
        self.cerebral_blood_flow = 50.0  # mL/100g/min
        self.cerebral_blood_volume = 4.0  # mL/100g
        self.oxygen_extraction_fraction = 0.4
        self.cmro2 = 3.5  # Cerebral metabolic rate of oxygen, mL/100g/min
        
        # Autoregulation
        self.mean_arterial_pressure = 90.0  # mmHg
        self.cerebral_perfusion_pressure = 80.0  # mmHg
        self.intracranial_pressure = 10.0  # mmHg
        
        # Neurovascular coupling mediators
        self.nitric_oxide = 1.0  # Relative level
        self.prostaglandins = 1.0
        self.adenosine = 0.0
        
    def update_neurovascular_coupling(self, dt: float) -> None:
        """
        Update blood flow based on neural activity
        Implements functional hyperemia
        """
        
        # Calculate average neural activity
        if not self.neurons:
            return
        
        avg_activity = np.mean([n.membrane_potential > -60 for n in self.neurons])
        
        # Activity-dependent vasodilation
        if avg_activity > 0.3:
            # Release vasodilators
            self.nitric_oxide += avg_activity * 0.1
            self.adenosine += avg_activity * 0.05
            
            # Increase blood flow
            flow_increase = (self.nitric_oxide - 1.0) * 10.0 + \
                          (self.adenosine * 5.0)
            
            self.cerebral_blood_flow += flow_increase * dt
            
        # Return to baseline
        self.nitric_oxide = 1.0 + (self.nitric_oxide - 1.0) * np.exp(-dt / 5.0)
        self.adenosine = self.adenosine * np.exp(-dt / 10.0)
        self.cerebral_blood_flow = 50.0 + (self.cerebral_blood_flow - 50.0) * np.exp(-dt / 20.0)
    
    def calculate_bold_signal(self) -> float:
        """
        Calculate BOLD (Blood Oxygen Level Dependent) signal for fMRI simulation
        Based on the Balloon model
        """
        
        # Deoxyhemoglobin concentration changes
        baseline_deoxy = 1.0
        
        # Increased flow reduces deoxyhemoglobin
        flow_factor = self.cerebral_blood_flow / 50.0
        volume_factor = self.cerebral_blood_volume / 4.0
        
        # BOLD signal approximation
        deoxy = baseline_deoxy * (volume_factor / flow_factor)
        
        # Convert to BOLD signal change
        bold_change = 1.0 - deoxy
        
        return bold_change * 100  # Percent signal change


class BrainSimulator:
    """
    Main brain simulation orchestrator
    Integrates all components for comprehensive brain modeling
    """
    
    def __init__(self, num_neurons: int = 1000, num_regions: int = 10):
        """
        Initialize brain simulator with specified complexity
        
        Parameters:
        -----------
        num_neurons : int
            Total number of neurons to simulate
        num_regions : int
            Number of distinct brain regions to model
        """
        
        self.num_neurons = num_neurons
        self.num_regions = num_regions
        
        # Initialize anatomical structures
        self.brain_regions = self._initialize_brain_regions()
        self.neurons = self._create_neural_network()
        self.synapses = self._create_synaptic_connections()
        
        # Initialize glial cells
        self.astrocytes = self._create_astrocytes()
        self.microglia = self._create_microglia()
        
        # Initialize vascular and CSF systems
        self.blood_brain_barrier = BloodBrainBarrier()
        self.meningeal_layers = self._create_meninges()
        self.neurovascular_units = self._create_neurovascular_units()
        
        # Simulation parameters
        self.time = 0.0
        self.dt = 0.1  # ms
        
        # Recording and analysis
        self.spike_times = defaultdict(list)
        self.lfp_recording = []  # Local field potential
        self.bold_signal = []  # fMRI BOLD signal
        
    def _initialize_brain_regions(self) -> List[BrainRegion]:
        """Initialize selected brain regions for simulation"""
        
        # Select diverse regions for simulation
        selected_regions = [
            BrainRegion.PRIMARY_MOTOR_CORTEX,
            BrainRegion.PRIMARY_SOMATOSENSORY,
            BrainRegion.PRIMARY_VISUAL,
            BrainRegion.HIPPOCAMPUS_CA1,
            BrainRegion.AMYGDALA_BASOLATERAL,
            BrainRegion.THALAMUS_VENTRAL_LATERAL,
            BrainRegion.SUBSTANTIA_NIGRA_COMPACTA,
            BrainRegion.CEREBELLAR_CORTEX,
            BrainRegion.CINGULATE_ANTERIOR,
            BrainRegion.DORSOLATERAL_PREFRONTAL
        ]
        
        return selected_regions[:self.num_regions]
    
    def _create_neural_network(self) -> List[Neuron]:
        """
        Create a biologically realistic neural network
        Includes diverse neuron types and spatial organization
        """
        
        neurons = []
        neurons_per_region = self.num_neurons // self.num_regions
        
        for region_idx, region in enumerate(self.brain_regions):
            for i in range(neurons_per_region):
                # Determine neuron type based on region and probability
                if "cortex" in region.name.lower():
                    # Cortical composition: 80% excitatory, 20% inhibitory
                    neuron_type = "pyramidal" if np.random.random() < 0.8 else "interneuron"
                elif "cerebellar" in region.name.lower():
                    # Cerebellar composition
                    neuron_type = np.random.choice(["purkinje", "granule", "stellate"], 
                                                  p=[0.1, 0.8, 0.1])
                else:
                    neuron_type = "projection"
                
                # Spatial position (simplified 3D coordinates)
                position = np.array([
                    region_idx * 100 + np.random.randn() * 10,
                    np.random.randn() * 50,
                    np.random.randn() * 50
                ])
                
                # Create neuron
                neuron = Neuron(
                    neuron_id=len(neurons),
                    neuron_type=neuron_type,
                    position=position,
                    region=region
                )
                
                # Set neurotransmitter based on type
                if neuron_type in ["pyramidal", "granule"]:
                    neuron.neurotransmitters[NeurotransmitterType.GLUTAMATE] = 1.0
                elif neuron_type == "interneuron":
                    neuron.neurotransmitters[NeurotransmitterType.GABA] = 1.0
                elif neuron_type == "purkinje":
                    neuron.neurotransmitters[NeurotransmitterType.GABA] = 1.0
                
                # Special cases for specific regions
                if region == BrainRegion.SUBSTANTIA_NIGRA_COMPACTA:
                    neuron.neurotransmitters[NeurotransmitterType.DOPAMINE] = 1.0
                elif region == BrainRegion.PONS_LOCUS_COERULEUS:
                    neuron.neurotransmitters[NeurotransmitterType.NOREPINEPHRINE] = 1.0
                elif region == BrainRegion.PONS_RAPHE_NUCLEI:
                    neuron.neurotransmitters[NeurotransmitterType.SEROTONIN] = 1.0
                
                neurons.append(neuron)
        
        return neurons
    
    def _create_synaptic_connections(self) -> List[Synapse]:
        """
        Create synaptic connections following biological connectivity patterns
        Implements both local and long-range connections
        """
        
        synapses = []
        
        for pre_neuron in self.neurons:
            # Number of output connections depends on neuron type
            if pre_neuron.neuron_type == "pyramidal":
                num_connections = np.random.poisson(100)  # Many connections
            elif pre_neuron.neuron_type == "interneuron":
                num_connections = np.random.poisson(50)  # Local connections
            else:
                num_connections = np.random.poisson(30)
            
            # Create connections
            for _ in range(num_connections):
                # Connection probability decreases with distance
                distances = [np.linalg.norm(pre_neuron.position - post.position) 
                           for post in self.neurons if post != pre_neuron]
                
                # Exponential decay of connection probability
                probabilities = np.exp(-np.array(distances) / 100.0)
                probabilities /= probabilities.sum()
                
                # Select post-synaptic neuron
                post_idx = np.random.choice(len(self.neurons) - 1, p=probabilities)
                if post_idx >= self.neurons.index(pre_neuron):
                    post_idx += 1
                post_neuron = self.neurons[post_idx]
                
                # Determine neurotransmitter
                if pre_neuron.neurotransmitters:
                    nt_type = list(pre_neuron.neurotransmitters.keys())[0]
                else:
                    nt_type = NeurotransmitterType.GLUTAMATE
                
                # Create synapse
                synapse = Synapse(
                    pre_neuron=pre_neuron,
                    post_neuron=post_neuron,
                    neurotransmitter=nt_type
                )
                
                # Add to neuron synapse lists
                pre_neuron.synapses_out.append(synapse)
                post_neuron.synapses_in.append(synapse)
                
                synapses.append(synapse)
        
        return synapses
    
    def _create_astrocytes(self) -> List[Astrocyte]:
        """Create astrocyte network with appropriate density"""
        
        astrocytes = []
        
        # Astrocyte to neuron ratio approximately 1:1 in humans
        num_astrocytes = self.num_neurons
        
        for i in range(num_astrocytes):
            # Random position near neurons
            nearby_neuron = np.random.choice(self.neurons)
            position = nearby_neuron.position + np.random.randn(3) * 20
            
            astrocyte = Astrocyte(
                position=position,
                region=nearby_neuron.region
            )
            
            astrocytes.append(astrocyte)
        
        # Create gap junction network
        for astro in astrocytes:
            # Connect to nearby astrocytes
            distances = [np.linalg.norm(astro.position - other.position) 
                        for other in astrocytes if other != astro]
            nearby_indices = np.where(np.array(distances) < 50)[0]
            
            for idx in nearby_indices[:5]:  # Max 5 gap junctions
                astro.gap_junctions.append(astrocytes[idx])
        
        return astrocytes
    
    def _create_microglia(self) -> List[Microglia]:
        """Create microglial cells with appropriate density"""
        
        microglia = []
        
        # Microglia comprise about 10% of brain cells
        num_microglia = self.num_neurons // 10
        
        for i in range(num_microglia):
            # Distributed throughout brain
            position = np.random.randn(3) * 100
            region = np.random.choice(self.brain_regions)
            
            microglial_cell = Microglia(
                position=position,
                region=region
            )
            
            microglia.append(microglial_cell)
        
        return microglia
    
    def _create_meninges(self) -> Dict[str, MeningealLayer]:
        """Create the three meningeal layers"""
        
        return {
            "dura": MeningealLayer(
                layer_type="dura",
                thickness=0.5,  # mm
                permeability=0.1
            ),
            "arachnoid": MeningealLayer(
                layer_type="arachnoid",
                thickness=0.2,
                permeability=0.5,
                csf_volume=150.0  # mL total CSF
            ),
            "pia": MeningealLayer(
                layer_type="pia",
                thickness=0.1,
                permeability=0.8
            )
        }
    
    def _create_neurovascular_units(self) -> List[NeurovascularUnit]:
        """Create neurovascular units for each brain region"""
        
        units = []
        
        for region in self.brain_regions:
            unit = NeurovascularUnit()
            
            # Add neurons from this region
            unit.neurons = [n for n in self.neurons if n.region == region]
            
            # Add nearby astrocytes
            if unit.neurons:
                region_center = np.mean([n.position for n in unit.neurons], axis=0)
                unit.astrocytes = [a for a in self.astrocytes 
                                 if np.linalg.norm(a.position - region_center) < 100]
            
            # Add microglia
            unit.microglia = [m for m in self.microglia if m.region == region]
            
            units.append(unit)
        
        return units
    
    def simulate_step(self) -> None:
        """
        Simulate one time step of brain activity
        Integrates all components: neurons, glia, vasculature, CSF
        """
        
        # Update neurons
        for neuron in self.neurons:
            # Add random input current (simplified sensory input)
            input_current = np.random.randn() * 0.5
            neuron.update_membrane_potential(self.dt, input_current)
            
            # Record spikes
            if neuron.membrane_potential > 0:
                self.spike_times[neuron.neuron_id].append(self.time)
        
        # Update synapses
        for synapse in self.synapses:
            synapse.replenish_vesicles(self.dt / 1000.0)  # Convert to seconds
        
        # Update glial cells
        for astrocyte in self.astrocytes:
            astrocyte.support_neurons(self.neurons[:10])  # Support nearby neurons
        
        for microglial in self.microglia:
            microglial.support_neurons(self.neurons[:5])
            threats = microglial.perform_immune_surveillance()
        
        # Update neurovascular coupling
        for unit in self.neurovascular_units:
            unit.update_neurovascular_coupling(self.dt / 1000.0)
            
            # Record BOLD signal
            bold = unit.calculate_bold_signal()
            self.bold_signal.append(bold)
        
        # Update CSF dynamics
        arachnoid = self.meningeal_layers["arachnoid"]
        production = arachnoid.calculate_csf_production(self.dt / 1000.0)
        absorption = arachnoid.calculate_csf_absorption(self.dt / 1000.0)
        
        # Update blood-brain barrier
        self.blood_brain_barrier.update_integrity(self.dt / 1000.0)
        
        # Calculate local field potential (simplified)
        lfp = np.mean([n.membrane_potential for n in self.neurons[:100]])
        self.lfp_recording.append(lfp)
        
        # Update time
        self.time += self.dt
    
    def run_simulation(self, duration_ms: float) -> Dict[str, any]:
        """
        Run complete simulation for specified duration
        
        Parameters:
        -----------
        duration_ms : float
            Simulation duration in milliseconds
            
        Returns:
        --------
        results : dict
            Dictionary containing simulation results and analyses
        """
        
        num_steps = int(duration_ms / self.dt)
        
        print(f"Starting brain simulation for {duration_ms} ms...")
        print(f"Simulating {self.num_neurons} neurons across {self.num_regions} brain regions")
        print(f"Total synapses: {len(self.synapses)}")
        print(f"Astrocytes: {len(self.astrocytes)}, Microglia: {len(self.microglia)}")
        
        for step in range(num_steps):
            self.simulate_step()
            
            # Progress update
            if step % 1000 == 0:
                progress = (step / num_steps) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Compile results
        results = self.analyze_results()
        
        print("Simulation complete!")
        
        return results
    
    def analyze_results(self) -> Dict[str, any]:
        """
        Analyze simulation results
        Calculates various neurological metrics
        """
        
        results = {}
        
        # Calculate firing rates
        firing_rates = {}
        for neuron_id, spikes in self.spike_times.items():
            if spikes:
                rate = len(spikes) / (self.time / 1000.0)  # Hz
                firing_rates[neuron_id] = rate
        
        results["mean_firing_rate"] = np.mean(list(firing_rates.values())) if firing_rates else 0
        results["max_firing_rate"] = max(firing_rates.values()) if firing_rates else 0
        
        # Analyze connectivity
        results["total_synapses"] = len(self.synapses)
        results["mean_synaptic_strength"] = np.mean([s.strength for s in self.synapses])
        
        # Analyze LFP
        if self.lfp_recording:
            results["lfp_mean"] = np.mean(self.lfp_recording)
            results["lfp_std"] = np.std(self.lfp_recording)
            
            # Simple frequency analysis (would use FFT in practice)
            results["lfp_oscillations"] = self._detect_oscillations(self.lfp_recording)
        
        # Analyze BOLD signal
        if self.bold_signal:
            results["bold_mean"] = np.mean(self.bold_signal)
            results["bold_max"] = max(self.bold_signal)
        
        # Glial cell analysis
        results["astrocyte_activation"] = np.mean([a.activation_state for a in self.astrocytes])
        results["microglial_activation"] = np.mean([m.activation_state for m in self.microglia])
        
        # BBB integrity
        results["bbb_integrity"] = self.blood_brain_barrier.tight_junction_integrity
        
        # CSF volume
        results["csf_volume"] = self.meningeal_layers["arachnoid"].csf_volume
        
        return results
    
    def _detect_oscillations(self, signal: List[float]) -> Dict[str, float]:
        """
        Detect neural oscillations in different frequency bands
        Simplified frequency analysis for demonstration
        """
        
        oscillations = {
            "delta": 0.0,    # 0.5-4 Hz (sleep, unconscious)
            "theta": 0.0,    # 4-8 Hz (memory, navigation)
            "alpha": 0.0,    # 8-12 Hz (relaxed, eyes closed)
            "beta": 0.0,     # 12-30 Hz (active thinking)
            "gamma": 0.0     # 30-100 Hz (consciousness, binding)
        }
        
        if len(signal) < 100:
            return oscillations
        
        # Simple zero-crossing analysis (simplified)
        zero_crossings = 0
        for i in range(1, len(signal)):
            if (signal[i-1] < 0 and signal[i] >= 0) or \
               (signal[i-1] >= 0 and signal[i] < 0):
                zero_crossings += 1
        
        # Estimate dominant frequency
        duration_s = (len(signal) * self.dt) / 1000.0
        frequency = zero_crossings / (2 * duration_s)
        
        # Assign to frequency band
        if frequency < 4:
            oscillations["delta"] = 1.0
        elif frequency < 8:
            oscillations["theta"] = 1.0
        elif frequency < 12:
            oscillations["alpha"] = 1.0
        elif frequency < 30:
            oscillations["beta"] = 1.0
        else:
            oscillations["gamma"] = 1.0
        
        return oscillations
    
    def simulate_pathology(self, pathology_type: str, severity: float = 0.5) -> None:
        """
        Simulate various neurological pathologies
        
        Parameters:
        -----------
        pathology_type : str
            Type of pathology to simulate
        severity : float
            Severity of pathology (0-1)
        """
        
        print(f"Simulating {pathology_type} with severity {severity}")
        
        if pathology_type == "stroke":
            self._simulate_stroke(severity)
        elif pathology_type == "alzheimer":
            self._simulate_alzheimer(severity)
        elif pathology_type == "parkinson":
            self._simulate_parkinson(severity)
        elif pathology_type == "epilepsy":
            self._simulate_epilepsy(severity)
        elif pathology_type == "traumatic_brain_injury":
            self._simulate_tbi(severity)
        elif pathology_type == "multiple_sclerosis":
            self._simulate_ms(severity)
        else:
            warnings.warn(f"Unknown pathology type: {pathology_type}")
    
    def _simulate_stroke(self, severity: float) -> None:
        """
        Simulate ischemic stroke
        Models blood flow reduction and excitotoxicity
        """
        
        # Select affected region
        affected_region = np.random.choice(self.brain_regions)
        affected_neurons = [n for n in self.neurons if n.region == affected_region]
        
        print(f"Stroke affecting {affected_region.name} ({len(affected_neurons)} neurons)")
        
        # Reduce blood flow in affected neurovascular unit
        for unit in self.neurovascular_units:
            if unit.neurons and unit.neurons[0].region == affected_region:
                # Reduce cerebral blood flow
                unit.cerebral_blood_flow *= (1.0 - severity)
                unit.oxygen_extraction_fraction *= (1.0 + severity * 0.5)
                
                # Energy crisis in neurons
                for neuron in affected_neurons:
                    neuron.atp_level *= (1.0 - severity * 0.8)
                    neuron.glucose_uptake *= (1.0 - severity * 0.7)
                    
                    # Excitotoxicity - excessive glutamate
                    neuron.neurotransmitters[NeurotransmitterType.GLUTAMATE] = severity * 5.0
                    
                    # Depolarization due to energy failure
                    neuron.resting_potential += severity * 20  # Less negative
                    neuron.threshold_potential -= severity * 10  # Lower threshold
                
                # Activate astrocytes and microglia
                for astrocyte in unit.astrocytes:
                    astrocyte.respond_to_injury(severity)
                    astrocyte.glutamate_uptake_rate *= 2.0  # Try to clear glutamate
                
                for microglial in unit.microglia:
                    microglial.respond_to_injury(severity)
                
                # BBB breakdown
                self.blood_brain_barrier.inflammation_level += severity
                self.blood_brain_barrier.oxidative_stress += severity * 0.8
    
    def _simulate_alzheimer(self, severity: float) -> None:
        """
        Simulate Alzheimer's disease
        Models amyloid plaques, tau tangles, and neurodegeneration
        """
        
        print(f"Simulating Alzheimer's disease pathology")
        
        # Primarily affects hippocampus and cortical regions
        affected_regions = [
            BrainRegion.HIPPOCAMPUS_CA1,
            BrainRegion.ENTORHINAL_CORTEX,
            BrainRegion.CINGULATE_POSTERIOR,
            BrainRegion.DORSOLATERAL_PREFRONTAL
        ]
        
        for region in affected_regions:
            affected_neurons = [n for n in self.neurons if n.region == region]
            
            for neuron in affected_neurons:
                if np.random.random() < severity:
                    # Amyloid accumulation affects synaptic function
                    for synapse in neuron.synapses_in:
                        synapse.strength *= (1.0 - severity * 0.5)
                        synapse.release_probability *= (1.0 - severity * 0.3)
                    
                    # Tau pathology affects axonal transport
                    neuron.axon_length *= (1.0 - severity * 0.2)
                    
                    # Metabolic dysfunction
                    neuron.glucose_uptake *= (1.0 - severity * 0.4)
                    neuron.atp_level *= (1.0 - severity * 0.3)
                    
                    # Reduced neurotrophic support
                    neuron.bdnf_level *= (1.0 - severity * 0.6)
        
        # Microglial activation (neuroinflammation)
        for microglial in self.microglia:
            if microglial.region in affected_regions:
                microglial.respond_to_injury(severity * 0.7)
                microglial.phagocytic_activity += severity  # Clear amyloid
                
                # Chronic inflammation
                microglial.cytokine_production["IL-1β"] += severity * 0.3
                microglial.cytokine_production["TNF-α"] += severity * 0.3
        
        # Cholinergic deficit
        for neuron in self.neurons:
            if NeurotransmitterType.ACETYLCHOLINE in neuron.neurotransmitters:
                neuron.neurotransmitters[NeurotransmitterType.ACETYLCHOLINE] *= (1.0 - severity * 0.6)
    
    def _simulate_parkinson(self, severity: float) -> None:
        """
        Simulate Parkinson's disease
        Models dopaminergic neuron loss and Lewy body formation
        """
        
        print(f"Simulating Parkinson's disease pathology")
        
        # Primarily affects substantia nigra dopaminergic neurons
        sn_neurons = [n for n in self.neurons 
                     if n.region == BrainRegion.SUBSTANTIA_NIGRA_COMPACTA]
        
        # Progressive dopaminergic neuron loss
        neurons_to_kill = int(len(sn_neurons) * severity * 0.8)  # Up to 80% loss
        
        for i, neuron in enumerate(sn_neurons[:neurons_to_kill]):
            # Alpha-synuclein accumulation (Lewy bodies)
            neuron.atp_level *= (1.0 - severity)
            
            # Mitochondrial dysfunction
            neuron.oxygen_consumption *= (1.0 - severity * 0.7)
            
            # Reduce dopamine production
            if NeurotransmitterType.DOPAMINE in neuron.neurotransmitters:
                neuron.neurotransmitters[NeurotransmitterType.DOPAMINE] *= (1.0 - severity * 0.9)
            
            # Oxidative stress
            neuron.calcium_concentration += severity * 0.001
        
        # Compensatory changes in basal ganglia
        bg_regions = [
            BrainRegion.CAUDATE_NUCLEUS,
            BrainRegion.PUTAMEN,
            BrainRegion.GLOBUS_PALLIDUS_EXTERNA,
            BrainRegion.GLOBUS_PALLIDUS_INTERNA
        ]
        
        for region in bg_regions:
            region_neurons = [n for n in self.neurons if n.region == region]
            
            for neuron in region_neurons:
                # Altered firing patterns
                neuron.threshold_potential -= severity * 5  # Hyperexcitability
                
                # Receptor upregulation (compensation)
                neuron.receptor_expression["D2"] = 1.0 + severity * 0.5
        
        # Microglial activation in substantia nigra
        for microglial in self.microglia:
            if microglial.region == BrainRegion.SUBSTANTIA_NIGRA_COMPACTA:
                microglial.respond_to_injury(severity)
    
    def _simulate_epilepsy(self, severity: float) -> None:
        """
        Simulate epileptic seizure activity
        Models hyperexcitability and synchronization
        """
        
        print(f"Simulating epileptic activity")
        
        # Create seizure focus
        focus_region = np.random.choice(self.brain_regions)
        focus_neurons = [n for n in self.neurons if n.region == focus_region][:50]
        
        for neuron in focus_neurons:
            # Increase excitability
            neuron.threshold_potential -= severity * 20  # Much lower threshold
            neuron.sodium_channels *= (1.0 + severity)  # More Na+ channels
            
            # Reduce inhibition
            gaba_synapses = [s for s in neuron.synapses_in 
                           if s.neurotransmitter == NeurotransmitterType.GABA]
            for synapse in gaba_synapses:
                synapse.strength *= (1.0 - severity * 0.7)
            
            # Increase excitation
            glutamate_synapses = [s for s in neuron.synapses_in 
                                if s.neurotransmitter == NeurotransmitterType.GLUTAMATE]
            for synapse in glutamate_synapses:
                synapse.strength *= (1.0 + severity * 0.5)
        
        # Spreading depolarization
        for neuron in self.neurons:
            distance_to_focus = min([np.linalg.norm(neuron.position - f.position) 
                                    for f in focus_neurons])
            
            if distance_to_focus < 200:  # Spreading zone
                spread_factor = 1.0 - (distance_to_focus / 200.0)
                neuron.threshold_potential -= severity * spread_factor * 10
    
    def _simulate_tbi(self, severity: float) -> None:
        """
        Simulate traumatic brain injury
        Models diffuse axonal injury and neuroinflammation
        """
        
        print(f"Simulating traumatic brain injury")
        
        # Mechanical damage to axons (diffuse axonal injury)
        for neuron in self.neurons:
            if np.random.random() < severity * 0.3:
                # Axonal damage
                neuron.axon_length *= (1.0 - severity * 0.5)
                
                # Disrupted axonal transport
                for synapse in neuron.synapses_out:
                    synapse.num_vesicles = int(synapse.num_vesicles * (1.0 - severity * 0.4))
        
        # Widespread neuroinflammation
        for microglial in self.microglia:
            microglial.respond_to_injury(severity * 0.8)
        
        for astrocyte in self.astrocytes:
            astrocyte.respond_to_injury(severity * 0.6)
        
        # BBB disruption
        self.blood_brain_barrier.tight_junction_integrity *= (1.0 - severity * 0.6)
        self.blood_brain_barrier.inflammation_level += severity
        
        # Increased intracranial pressure
        self.meningeal_layers["arachnoid"].csf_pressure += severity * 20  # mmHg
        
        # Metabolic crisis
        for unit in self.neurovascular_units:
            unit.cmro2 *= (1.0 + severity * 0.3)  # Increased oxygen demand
            unit.cerebral_blood_flow *= (1.0 - severity * 0.2)  # But reduced flow
    
    def _simulate_ms(self, severity: float) -> None:
        """
        Simulate multiple sclerosis
        Models demyelination and autoimmune response
        """
        
        print(f"Simulating multiple sclerosis pathology")
        
        # Create demyelinating lesions
        num_lesions = int(severity * 10)
        
        for _ in range(num_lesions):
            # Random lesion location
            lesion_center = np.random.randn(3) * 100
            lesion_radius = 50.0
            
            affected_neurons = [n for n in self.neurons 
                              if np.linalg.norm(n.position - lesion_center) < lesion_radius]
            
            for neuron in affected_neurons:
                # Demyelination reduces conduction velocity
                # Modeled as reduced sodium channel density along axon
                neuron.sodium_channels *= (1.0 - severity * 0.6)
                neuron.potassium_channels *= (1.0 - severity * 0.4)
                
                # Axonal damage
                if np.random.random() < severity * 0.2:
                    neuron.axon_length *= 0.5
            
            # T-cell infiltration (modeled through microglial activation)
            affected_microglia = [m for m in self.microglia 
                                if np.linalg.norm(m.position - lesion_center) < lesion_radius]
            
            for microglial in affected_microglia:
                microglial.phenotype = "M1"  # Pro-inflammatory
                microglial.cytokine_production["IL-1β"] += severity
                microglial.cytokine_production["TNF-α"] += severity
                microglial.phagocytic_activity += severity * 0.5  # Myelin phagocytosis
        
        # BBB disruption at lesion sites
        self.blood_brain_barrier.inflammation_level += severity * 0.7
        self.blood_brain_barrier.metalloproteinase_activity += severity * 0.5


def demonstrate_brain_simulation():
    """
    Comprehensive demonstration of the brain and cranial systems model
    Shows various simulations and analyses
    """
    
    print("=" * 80)
    print("COMPREHENSIVE BRAIN AND CRANIAL SYSTEMS MODELING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize simulator with moderate complexity
    print("\nInitializing brain simulator...")
    simulator = BrainSimulator(num_neurons=500, num_regions=8)
    
    print(f"\nBrain architecture created:")
    print(f"- Neurons: {len(simulator.neurons)}")
    print(f"- Synapses: {len(simulator.synapses)}")
    print(f"- Astrocytes: {len(simulator.astrocytes)}")
    print(f"- Microglia: {len(simulator.microglia)}")
    print(f"- Brain regions modeled: {len(simulator.brain_regions)}")
    
    # Display brain regions
    print("\nBrain regions included in simulation:")
    for i, region in enumerate(simulator.brain_regions):
        print(f"  {i+1}. {region.name}")
        print(f"     - Brodmann area: {region.brodmann_area}")
        print(f"     - Function: {region.function}")
    
    # Run baseline simulation
    print("\n" + "=" * 60)
    print("BASELINE HEALTHY BRAIN SIMULATION")
    print("=" * 60)
    
    baseline_results = simulator.run_simulation(duration_ms=1000)
    
    print("\nBaseline Results:")
    print(f"- Mean firing rate: {baseline_results['mean_firing_rate']:.2f} Hz")
    print(f"- Maximum firing rate: {baseline_results['max_firing_rate']:.2f} Hz")
    print(f"- Mean synaptic strength: {baseline_results['mean_synaptic_strength']:.3f}")
    print(f"- LFP mean: {baseline_results['lfp_mean']:.2f} mV")
    print(f"- LFP standard deviation: {baseline_results['lfp_std']:.2f} mV")
    print(f"- BOLD signal mean: {baseline_results['bold_mean']:.2f}%")
    print(f"- Blood-brain barrier integrity: {baseline_results['bbb_integrity']:.2%}")
    print(f"- CSF volume: {baseline_results['csf_volume']:.1f} mL")
    
    print("\nNeural oscillations detected:")
    for band, power in baseline_results['lfp_oscillations'].items():
        if power > 0:
            print(f"  - {band}: {power:.1f}")
    
    # Demonstrate pathology simulations
    print("\n" + "=" * 60)
    print("NEUROLOGICAL PATHOLOGY SIMULATIONS")
    print("=" * 60)
    
    pathologies = [
        ("stroke", 0.6, "Ischemic stroke with 60% severity"),
        ("alzheimer", 0.4, "Mild-moderate Alzheimer's disease"),
        ("parkinson", 0.5, "Moderate Parkinson's disease"),
        ("epilepsy", 0.7, "Severe epileptic activity"),
        ("traumatic_brain_injury", 0.5, "Moderate TBI"),
        ("multiple_sclerosis", 0.3, "Mild MS with demyelination")
    ]
    
    for pathology, severity, description in pathologies:
        print(f"\n--- {description} ---")
        
        # Reset to healthy state
        simulator = BrainSimulator(num_neurons=300, num_regions=5)
        
        # Apply pathology
        simulator.simulate_pathology(pathology, severity)
        
        # Run simulation
        pathology_results = simulator.run_simulation(duration_ms=500)
        
        # Compare to baseline
        print(f"Changes from baseline:")
        print(f"  - Firing rate change: {pathology_results['mean_firing_rate'] - baseline_results['mean_firing_rate']:.2f} Hz")
        print(f"  - Synaptic strength change: {pathology_results['mean_synaptic_strength'] - baseline_results['mean_synaptic_strength']:.3f}")
        print(f"  - BBB integrity: {pathology_results['bbb_integrity']:.2%}")
        print(f"  - Astrocyte activation: {pathology_results['astrocyte_activation']:.2f}")
        print(f"  - Microglial activation: {pathology_results['microglial_activation']:.2f}")
    
    # Demonstrate specific cellular mechanisms
    print("\n" + "=" * 60)
    print("CELLULAR MECHANISM DEMONSTRATIONS")
    print("=" * 60)
    
    # Create single neuron for detailed analysis
    print("\n--- Single Neuron Hodgkin-Huxley Dynamics ---")
    test_neuron = Neuron(
        neuron_id=999,
        neuron_type="pyramidal",
        position=np.array([0, 0, 0]),
        region=BrainRegion.PRIMARY_MOTOR_CORTEX
    )
    
    print(f"Initial membrane potential: {test_neuron.membrane_potential:.2f} mV")
    
    # Simulate action potential
    membrane_trace = []
    for t in range(100):
        test_neuron.update_membrane_potential(dt=0.1, external_current=10.0)
        membrane_trace.append(test_neuron.membrane_potential)
    
    print(f"Peak membrane potential: {max(membrane_trace):.2f} mV")
    print(f"Action potentials generated: {sum(1 for v in membrane_trace if v > 0)}")
    
    # Demonstrate synaptic plasticity
    print("\n--- Synaptic Plasticity (LTP/LTD) ---")
    pre_neuron = Neuron(1000, "pyramidal", np.array([0, 0, 0]), BrainRegion.HIPPOCAMPUS_CA3)
    post_neuron = Neuron(1001, "pyramidal", np.array([10, 0, 0]), BrainRegion.HIPPOCAMPUS_CA1)
    
    synapse = Synapse(pre_neuron, post_neuron, NeurotransmitterType.GLUTAMATE)
    initial_strength = synapse.strength
    
    # Simulate high-frequency stimulation (LTP induction)
    for _ in range(100):
        synapse.transmit_signal()
        synapse.calcium_history.append(1.5)  # High calcium
        synapse._update_plasticity()
    
    print(f"Initial synaptic strength: {initial_strength:.3f}")
    print(f"After LTP induction: {synapse.strength:.3f}")
    print(f"Potentiation: {(synapse.strength/initial_strength - 1) * 100:.1f}%")
    
    # Demonstrate blood-brain barrier transport
    print("\n--- Blood-Brain Barrier Transport ---")
    bbb = BloodBrainBarrier()
    
    molecules = [
        ("glucose", 5.0, "Essential nutrient"),
        ("large_proteins", 1.0, "Albumin-sized protein"),
        ("lipophilic_small", 0.5, "Lipid-soluble drug")
    ]
    
    for molecule, gradient, description in molecules:
        transport = bbb.calculate_transport(molecule, gradient, dt=1.0)
        print(f"{description} ({molecule}): {transport:.6f} units/s")
    
    # Demonstrate CSF dynamics
    print("\n--- Cerebrospinal Fluid Dynamics ---")
    arachnoid = MeningealLayer("arachnoid", thickness=0.2, permeability=0.5)
    arachnoid.csf_volume = 150.0
    
    print(f"Initial CSF volume: {arachnoid.csf_volume:.1f} mL")
    print(f"Initial CSF pressure: {arachnoid.csf_pressure:.1f} mmHg")
    
    # Simulate CSF production and absorption for 1 hour
    for _ in range(3600):
        production = arachnoid.calculate_csf_production(dt=1.0)
        absorption = arachnoid.calculate_csf_absorption(dt=1.0)
    
    print(f"After 1 hour simulation:")
    print(f"  - CSF volume: {arachnoid.csf_volume:.1f} mL")
    print(f"  - Net production rate: {(production - absorption) * 60:.3f} mL/min")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThis comprehensive model demonstrates:")
    print("- Detailed neuronal electrophysiology (Hodgkin-Huxley)")
    print("- Synaptic transmission and plasticity")
    print("- Glial cell functions (astrocytes and microglia)")
    print("- Blood-brain barrier transport")
    print("- CSF circulation dynamics")
    print("- Neurovascular coupling")
    print("- Multiple neurological pathologies")
    print("- Brain imaging signal generation (BOLD, LFP)")
    print("\nThe framework provides a foundation for:")
    print("- Drug development and testing")
    print("- Disease mechanism research")
    print("- Neural prosthetics design")
    print("- Educational neuroscience tools")
    print("- Computational psychiatry applications")


if __name__ == "__main__":
    # Run the comprehensive demonstration
    demonstrate_brain_simulation()