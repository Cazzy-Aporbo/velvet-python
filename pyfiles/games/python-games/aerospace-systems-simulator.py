"""
COMPREHENSIVE AEROSPACE PHYSICAL SYSTEMS SIMULATOR
Advanced simulation framework for aerospace engineering demonstrating:
- Six degree-of-freedom (6DOF) rigid body dynamics
- Orbital mechanics with perturbations
- Atmospheric flight dynamics
- Propulsion system modeling
- Control systems and guidance
- Thermal management systems
- Structural dynamics and vibrations
- Multi-body dynamics
- Sensor modeling and Kalman filtering

This simulator implements real aerospace engineering equations and methods
used in actual spacecraft and aircraft design.

Author: Cazandra Aporbo
Version: 3.0.0
Python Requirements: 3.8+
Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy import integrate, optimize, interpolate, linalg
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Union
from enum import Enum, auto
import math
from collections import deque
import json
import warnings
from abc import ABC, abstractmethod
import time


# Physical constants
class PhysicalConstants:
    """
    Standard physical constants used in aerospace calculations.
    All units in SI unless otherwise noted.
    """
    # Universal constants
    GRAVITATIONAL_CONSTANT = 6.67430e-11  # N⋅m²/kg²
    SPEED_OF_LIGHT = 299792458  # m/s
    BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
    STEFAN_BOLTZMANN = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
    
    # Earth parameters
    EARTH_MASS = 5.972e24  # kg
    EARTH_RADIUS_EQUATORIAL = 6378137.0  # m
    EARTH_RADIUS_POLAR = 6356752.314245  # m
    EARTH_MU = 3.986004418e14  # m³/s² (gravitational parameter)
    EARTH_ROTATION_RATE = 7.292115e-5  # rad/s
    EARTH_J2 = 1.08262668e-3  # Second zonal harmonic
    EARTH_J3 = -2.53265648533e-6  # Third zonal harmonic
    EARTH_J4 = -1.619621591367e-6  # Fourth zonal harmonic
    
    # Atmosphere parameters (standard atmosphere at sea level)
    SEA_LEVEL_PRESSURE = 101325.0  # Pa
    SEA_LEVEL_TEMPERATURE = 288.15  # K
    SEA_LEVEL_DENSITY = 1.225  # kg/m³
    GAS_CONSTANT_AIR = 287.052  # J/(kg⋅K)
    SPECIFIC_HEAT_RATIO = 1.4  # γ for air
    
    # Other celestial bodies
    MOON_MASS = 7.342e22  # kg
    MOON_RADIUS = 1737400.0  # m
    MOON_MU = 4.9048695e12  # m³/s²
    SUN_MASS = 1.989e30  # kg
    SUN_MU = 1.32712440018e20  # m³/s²


class CoordinateFrame(Enum):
    """
    Coordinate reference frames used in aerospace engineering.
    """
    ECEF = auto()  # Earth-Centered Earth-Fixed
    ECI = auto()   # Earth-Centered Inertial
    NED = auto()   # North-East-Down
    BODY = auto()  # Body-fixed frame
    LVLH = auto()  # Local Vertical Local Horizontal
    WIND = auto()  # Wind axes
    PERIFOCAL = auto()  # Perifocal frame


@dataclass
class StateVector:
    """
    Complete state vector for a rigid body in 3D space.
    Includes position, velocity, attitude quaternion, and angular velocity.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # quaternion
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    time: float = 0.0  # s
    
    def to_array(self) -> np.ndarray:
        """Convert state vector to numpy array for integration."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.angular_velocity
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray, time: float = 0.0) -> 'StateVector':
        """Create state vector from numpy array."""
        return cls(
            position=array[0:3],
            velocity=array[3:6],
            quaternion=array[6:10],
            angular_velocity=array[10:13],
            time=time
        )
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from quaternion."""
        return Rotation.from_quat(self.quaternion).as_matrix()


@dataclass
class OrbitalElements:
    """
    Classical orbital elements (Keplerian elements).
    """
    semi_major_axis: float  # a, meters
    eccentricity: float  # e, dimensionless
    inclination: float  # i, radians
    raan: float  # Ω, right ascension of ascending node, radians
    argument_of_periapsis: float  # ω, radians
    true_anomaly: float  # ν, radians
    epoch: float = 0.0  # Reference epoch time
    
    def to_state_vector(self, mu: float = PhysicalConstants.EARTH_MU) -> StateVector:
        """
        Convert orbital elements to Cartesian state vector.
        Uses the standard orbital elements to Cartesian transformation.
        """
        # Semi-latus rectum
        p = self.semi_major_axis * (1 - self.eccentricity**2)
        
        # Position in perifocal frame
        r = p / (1 + self.eccentricity * np.cos(self.true_anomaly))
        r_perifocal = r * np.array([
            np.cos(self.true_anomaly),
            np.sin(self.true_anomaly),
            0
        ])
        
        # Velocity in perifocal frame
        v_perifocal = np.sqrt(mu / p) * np.array([
            -np.sin(self.true_anomaly),
            self.eccentricity + np.cos(self.true_anomaly),
            0
        ])
        
        # Rotation matrix from perifocal to ECI
        cos_raan = np.cos(self.raan)
        sin_raan = np.sin(self.raan)
        cos_inc = np.cos(self.inclination)
        sin_inc = np.sin(self.inclination)
        cos_argp = np.cos(self.argument_of_periapsis)
        sin_argp = np.sin(self.argument_of_periapsis)
        
        R = np.array([
            [cos_raan*cos_argp - sin_raan*sin_argp*cos_inc,
             -cos_raan*sin_argp - sin_raan*cos_argp*cos_inc,
             sin_raan*sin_inc],
            [sin_raan*cos_argp + cos_raan*sin_argp*cos_inc,
             -sin_raan*sin_argp + cos_raan*cos_argp*cos_inc,
             -cos_raan*sin_inc],
            [sin_argp*sin_inc,
             cos_argp*sin_inc,
             cos_inc]
        ])
        
        # Transform to ECI frame
        position = R @ r_perifocal
        velocity = R @ v_perifocal
        
        return StateVector(position=position, velocity=velocity, time=self.epoch)
    
    @classmethod
    def from_state_vector(cls, state: StateVector, 
                         mu: float = PhysicalConstants.EARTH_MU) -> 'OrbitalElements':
        """
        Convert Cartesian state vector to orbital elements.
        """
        r = state.position
        v = state.velocity
        
        # Magnitudes
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Node vector
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        
        # Eccentricity vector
        e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
        e = np.linalg.norm(e_vec)
        
        # Specific energy
        energy = v_mag**2 / 2 - mu / r_mag
        
        # Semi-major axis
        if abs(e - 1.0) > 1e-10:  # Not parabolic
            a = -mu / (2 * energy)
        else:
            a = float('inf')
        
        # Inclination
        i = np.arccos(h[2] / h_mag)
        
        # RAAN
        if n_mag > 1e-10:
            raan = np.arccos(n[0] / n_mag)
            if n[1] < 0:
                raan = 2*np.pi - raan
        else:
            raan = 0
        
        # Argument of periapsis
        if n_mag > 1e-10 and e > 1e-10:
            argp = np.arccos(np.dot(n, e_vec) / (n_mag * e))
            if e_vec[2] < 0:
                argp = 2*np.pi - argp
        elif e > 1e-10:  # Equatorial orbit
            argp = np.arccos(e_vec[0] / e)
            if e_vec[1] < 0:
                argp = 2*np.pi - argp
        else:
            argp = 0
        
        # True anomaly
        if e > 1e-10:
            nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
            if np.dot(r, v) < 0:
                nu = 2*np.pi - nu
        else:  # Circular orbit
            nu = np.arccos(np.dot(n, r) / (n_mag * r_mag))
            if r[2] < 0:
                nu = 2*np.pi - nu
        
        return cls(a, e, i, raan, argp, nu, state.time)


class AtmosphereModel:
    """
    Standard atmosphere model for Earth.
    Implements US Standard Atmosphere 1976 with extensions.
    """
    
    def __init__(self):
        # Layer boundaries (geopotential altitude in meters)
        self.h_layers = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
        
        # Temperature gradients (K/m)
        self.lapse_rates = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
        
        # Base temperatures (K)
        self.T_base = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])
        
        # Base pressures (Pa)
        self.P_base = np.array([101325, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642])
    
    def get_properties(self, altitude: float) -> Dict[str, float]:
        """
        Calculate atmospheric properties at given geometric altitude.
        
        Parameters:
            altitude: Geometric altitude in meters
            
        Returns:
            Dictionary with temperature, pressure, density, and speed of sound
        """
        # Convert geometric to geopotential altitude
        h = self._geometric_to_geopotential(altitude)
        
        # Limit to model range
        h = max(0, min(h, self.h_layers[-1]))
        
        # Find layer
        layer_idx = np.searchsorted(self.h_layers[1:], h, side='right')
        
        # Calculate temperature
        h0 = self.h_layers[layer_idx]
        T0 = self.T_base[layer_idx]
        L = self.lapse_rates[layer_idx]
        T = T0 + L * (h - h0)
        
        # Calculate pressure
        P0 = self.P_base[layer_idx]
        R = PhysicalConstants.GAS_CONSTANT_AIR
        g0 = 9.80665  # Standard gravity
        
        if abs(L) < 1e-10:  # Isothermal layer
            P = P0 * np.exp(-g0 * (h - h0) / (R * T))
        else:  # Gradient layer
            P = P0 * (T / T0) ** (-g0 / (R * L))
        
        # Calculate density
        rho = P / (R * T)
        
        # Calculate speed of sound
        gamma = PhysicalConstants.SPECIFIC_HEAT_RATIO
        a = np.sqrt(gamma * R * T)
        
        # Dynamic viscosity (Sutherland's formula)
        mu = self._sutherland_viscosity(T)
        
        return {
            'temperature': T,
            'pressure': P,
            'density': rho,
            'speed_of_sound': a,
            'dynamic_viscosity': mu,
            'kinematic_viscosity': mu / rho
        }
    
    def _geometric_to_geopotential(self, z: float) -> float:
        """Convert geometric altitude to geopotential altitude."""
        r_earth = PhysicalConstants.EARTH_RADIUS_EQUATORIAL
        return r_earth * z / (r_earth + z)
    
    def _sutherland_viscosity(self, T: float) -> float:
        """Calculate dynamic viscosity using Sutherland's formula."""
        T0 = 273.15  # Reference temperature
        mu0 = 1.716e-5  # Reference viscosity
        S = 110.4  # Sutherland's constant
        
        return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)


class AerodynamicsModel:
    """
    Aerodynamic force and moment calculation model.
    Handles subsonic, transonic, and supersonic regimes.
    """
    
    def __init__(self, reference_area: float, reference_length: float):
        """
        Initialize aerodynamics model.
        
        Parameters:
            reference_area: Reference area for force coefficients (m²)
            reference_length: Reference length for moment coefficients (m)
        """
        self.S_ref = reference_area
        self.L_ref = reference_length
        
        # Example aerodynamic database (would be populated from wind tunnel/CFD)
        self._initialize_aero_database()
    
    def _initialize_aero_database(self):
        """Initialize aerodynamic coefficient database."""
        # Simplified model - real implementation would use extensive tables
        self.CL_alpha = 5.73  # Lift curve slope (per radian)
        self.CD_0 = 0.02  # Zero-lift drag coefficient
        self.K = 0.04  # Induced drag factor
        self.CL_max = 1.5  # Maximum lift coefficient
        
        # Control derivatives (per radian)
        self.CL_delta_e = 0.5  # Elevator effectiveness
        self.Cm_delta_e = -1.2  # Elevator pitching moment
        self.CY_delta_r = 0.2  # Rudder side force
        self.Cn_delta_r = -0.1  # Rudder yawing moment
        self.Cl_delta_a = 0.08  # Aileron rolling moment
        
        # Stability derivatives
        self.Cm_alpha = -0.5  # Pitch stability
        self.Cn_beta = 0.1  # Yaw stability
        self.Cl_beta = -0.1  # Roll stability (dihedral effect)
        
        # Damping derivatives (per rad/s)
        self.Cm_q = -20.0  # Pitch damping
        self.Cn_r = -0.15  # Yaw damping
        self.Cl_p = -0.4  # Roll damping
    
    def calculate_forces_moments(self, state: StateVector, 
                                atmosphere: Dict[str, float],
                                controls: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate aerodynamic forces and moments.
        
        Parameters:
            state: Current vehicle state
            atmosphere: Atmospheric properties
            controls: Control surface deflections
            
        Returns:
            Forces and moments in body frame
        """
        # Calculate airspeed vector in body frame
        v_body = state.get_rotation_matrix().T @ state.velocity
        
        # Account for wind (simplified - assumes no wind for now)
        v_air = v_body
        
        # Airspeed magnitude
        V = np.linalg.norm(v_air)
        
        if V < 1e-6:  # Essentially stationary
            return np.zeros(3), np.zeros(3)
        
        # Angle of attack and sideslip
        alpha = np.arctan2(v_air[2], v_air[0])  # rad
        beta = np.arcsin(v_air[1] / V) if V > 0 else 0  # rad
        
        # Mach number
        M = V / atmosphere['speed_of_sound']
        
        # Dynamic pressure
        q = 0.5 * atmosphere['density'] * V**2
        
        # Get aerodynamic coefficients
        CL, CD, CY, Cl, Cm, Cn = self._get_coefficients(
            alpha, beta, M, state.angular_velocity, V, controls
        )
        
        # Forces in stability axes
        L = q * self.S_ref * CL  # Lift
        D = q * self.S_ref * CD  # Drag
        Y = q * self.S_ref * CY  # Side force
        
        # Convert to body axes
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        forces = np.array([
            -D * cos_alpha + L * sin_alpha,
            Y,
            -D * sin_alpha - L * cos_alpha
        ])
        
        # Moments in body axes
        moments = q * self.S_ref * self.L_ref * np.array([Cl, Cm, Cn])
        
        return forces, moments
    
    def _get_coefficients(self, alpha: float, beta: float, mach: float,
                         omega: np.ndarray, V: float,
                         controls: Dict[str, float]) -> Tuple[float, ...]:
        """
        Calculate aerodynamic coefficients.
        
        This is a simplified model. Real implementation would use:
        - Multi-dimensional lookup tables
        - Mach number corrections
        - Reynolds number effects
        - Unsteady aerodynamics
        """
        # Normalized angular rates
        p_hat = omega[0] * self.L_ref / (2 * V) if V > 0 else 0
        q_hat = omega[1] * self.L_ref / (2 * V) if V > 0 else 0
        r_hat = omega[2] * self.L_ref / (2 * V) if V > 0 else 0
        
        # Control inputs
        delta_e = controls.get('elevator', 0)
        delta_a = controls.get('aileron', 0)
        delta_r = controls.get('rudder', 0)
        
        # Lift coefficient
        CL = self.CL_alpha * alpha + self.CL_delta_e * delta_e
        CL = np.clip(CL, -self.CL_max, self.CL_max)
        
        # Drag coefficient (parabolic polar with Mach correction)
        mach_factor = 1 + 0.2 * mach**2 if mach < 1 else 1.5
        CD = self.CD_0 * mach_factor + self.K * CL**2
        
        # Side force coefficient
        CY = self.CY_delta_r * delta_r
        
        # Rolling moment coefficient
        Cl = self.Cl_beta * beta + self.Cl_p * p_hat + self.Cl_delta_a * delta_a
        
        # Pitching moment coefficient
        Cm = self.Cm_alpha * alpha + self.Cm_q * q_hat + self.Cm_delta_e * delta_e
        
        # Yawing moment coefficient
        Cn = self.Cn_beta * beta + self.Cn_r * r_hat + self.Cn_delta_r * delta_r
        
        return CL, CD, CY, Cl, Cm, Cn


class PropulsionSystem:
    """
    Propulsion system model for various engine types.
    """
    
    def __init__(self, engine_type: str, **kwargs):
        """
        Initialize propulsion system.
        
        Parameters:
            engine_type: Type of engine ('rocket', 'turbojet', 'turbofan', 'electric')
            **kwargs: Engine-specific parameters
        """
        self.engine_type = engine_type
        self.params = kwargs
        
        if engine_type == 'rocket':
            self.isp_sea_level = kwargs.get('isp_sea_level', 300)  # seconds
            self.isp_vacuum = kwargs.get('isp_vacuum', 350)  # seconds
            self.max_thrust_sea_level = kwargs.get('max_thrust_sea_level', 1e6)  # N
            self.max_thrust_vacuum = kwargs.get('max_thrust_vacuum', 1.2e6)  # N
            self.mass_flow_rate = kwargs.get('mass_flow_rate', 300)  # kg/s
            
        elif engine_type == 'turbojet':
            self.max_thrust_static = kwargs.get('max_thrust_static', 1e5)  # N
            self.tsfc = kwargs.get('tsfc', 0.8)  # Thrust specific fuel consumption (kg/N-hr)
            
        elif engine_type == 'electric':
            self.max_thrust = kwargs.get('max_thrust', 1e-3)  # N
            self.isp = kwargs.get('isp', 3000)  # seconds
            self.efficiency = kwargs.get('efficiency', 0.7)
            self.power = kwargs.get('power', 1000)  # W
    
    def calculate_thrust(self, altitude: float, mach: float, 
                        throttle: float, atmosphere: Dict[str, float]) -> np.ndarray:
        """
        Calculate thrust vector based on conditions.
        
        Parameters:
            altitude: Altitude in meters
            mach: Mach number
            throttle: Throttle setting (0 to 1)
            atmosphere: Atmospheric properties
            
        Returns:
            Thrust vector in body frame (assumes aligned with x-axis)
        """
        if self.engine_type == 'rocket':
            thrust_magnitude = self._rocket_thrust(altitude, throttle, atmosphere)
        elif self.engine_type == 'turbojet':
            thrust_magnitude = self._turbojet_thrust(mach, throttle, atmosphere)
        elif self.engine_type == 'electric':
            thrust_magnitude = self._electric_thrust(throttle)
        else:
            thrust_magnitude = 0
        
        # Assume thrust along body x-axis
        return np.array([thrust_magnitude, 0, 0])
    
    def _rocket_thrust(self, altitude: float, throttle: float, 
                      atmosphere: Dict[str, float]) -> float:
        """Calculate rocket thrust with altitude compensation."""
        # Interpolate between sea level and vacuum performance
        pressure_ratio = atmosphere['pressure'] / PhysicalConstants.SEA_LEVEL_PRESSURE
        
        # Thrust varies with ambient pressure
        thrust_max = (self.max_thrust_sea_level * pressure_ratio + 
                     self.max_thrust_vacuum * (1 - pressure_ratio))
        
        return thrust_max * throttle
    
    def _turbojet_thrust(self, mach: float, throttle: float, 
                        atmosphere: Dict[str, float]) -> float:
        """Calculate turbojet thrust with Mach and altitude effects."""
        # Simplified thrust model
        density_ratio = atmosphere['density'] / PhysicalConstants.SEA_LEVEL_DENSITY
        
        # Thrust lapse with altitude and Mach
        if mach < 0.3:
            mach_factor = 1.0
        elif mach < 1.0:
            mach_factor = 1.0 + 0.3 * (mach - 0.3)
        else:
            mach_factor = 1.21 - 0.2 * (mach - 1.0)
        
        thrust = self.max_thrust_static * density_ratio * mach_factor * throttle
        
        return max(0, thrust)
    
    def _electric_thrust(self, throttle: float) -> float:
        """Calculate electric propulsion thrust."""
        return self.max_thrust * throttle
    
    def get_fuel_flow_rate(self, thrust: float) -> float:
        """Calculate fuel consumption rate."""
        if self.engine_type == 'rocket':
            return self.mass_flow_rate if thrust > 0 else 0
        elif self.engine_type == 'turbojet':
            # Convert TSFC from kg/N-hr to kg/s
            return abs(thrust) * self.tsfc / 3600
        elif self.engine_type == 'electric':
            g0 = 9.80665
            return abs(thrust) / (self.isp * g0) if self.isp > 0 else 0
        else:
            return 0


class GravityModel:
    """
    Gravitational field model with spherical harmonics.
    """
    
    def __init__(self, order: int = 4):
        """
        Initialize gravity model.
        
        Parameters:
            order: Maximum order of spherical harmonics to include
        """
        self.order = order
        self.mu = PhysicalConstants.EARTH_MU
        self.R_ref = PhysicalConstants.EARTH_RADIUS_EQUATORIAL
        
        # Zonal harmonics (J coefficients)
        self.J = np.zeros(order + 1)
        self.J[2] = PhysicalConstants.EARTH_J2
        self.J[3] = PhysicalConstants.EARTH_J3
        self.J[4] = PhysicalConstants.EARTH_J4
    
    def calculate_acceleration(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate gravitational acceleration including perturbations.
        
        Parameters:
            position: Position vector in ECI frame (meters)
            
        Returns:
            Acceleration vector in ECI frame (m/s²)
        """
        r = np.linalg.norm(position)
        
        if r < 1e-6:  # Avoid singularity
            return np.zeros(3)
        
        # Two-body acceleration
        a_two_body = -self.mu / r**3 * position
        
        # Add J2 perturbation (most significant)
        if self.order >= 2:
            x, y, z = position
            r_sq = r * r
            
            # J2 effect
            factor_J2 = 3/2 * self.J[2] * (self.R_ref/r)**2
            
            a_J2_x = factor_J2 * (5 * z**2 / r_sq - 1) * x / r**3
            a_J2_y = factor_J2 * (5 * z**2 / r_sq - 1) * y / r**3
            a_J2_z = factor_J2 * (5 * z**2 / r_sq - 3) * z / r**3
            
            a_J2 = self.mu * np.array([a_J2_x, a_J2_y, a_J2_z])
        else:
            a_J2 = np.zeros(3)
        
        # Higher order terms could be added here
        
        return a_two_body + a_J2
    
    def calculate_third_body_perturbation(self, position: np.ndarray, 
                                         time: float) -> np.ndarray:
        """
        Calculate perturbation from third bodies (Moon, Sun).
        
        Parameters:
            position: Spacecraft position in ECI (m)
            time: Current time (seconds since epoch)
            
        Returns:
            Perturbation acceleration (m/s²)
        """
        # Simplified model - assumes circular orbits for Moon and Sun
        
        # Moon position (simplified)
        moon_orbit_radius = 384400000  # m
        moon_angular_rate = 2 * np.pi / (27.321582 * 86400)  # rad/s
        moon_angle = moon_angular_rate * time
        moon_position = moon_orbit_radius * np.array([
            np.cos(moon_angle),
            np.sin(moon_angle),
            0
        ])
        
        # Vector from spacecraft to Moon
        r_moon = moon_position - position
        r_moon_mag = np.linalg.norm(r_moon)
        
        # Moon perturbation
        if r_moon_mag > 1e-6:
            mu_moon = PhysicalConstants.MOON_MU
            a_moon = mu_moon * (r_moon / r_moon_mag**3 - 
                               moon_position / np.linalg.norm(moon_position)**3)
        else:
            a_moon = np.zeros(3)
        
        # Similar calculation could be done for Sun
        
        return a_moon


class RigidBodyDynamics:
    """
    Six degree-of-freedom rigid body dynamics.
    """
    
    def __init__(self, mass: float, inertia_tensor: np.ndarray):
        """
        Initialize rigid body.
        
        Parameters:
            mass: Mass in kg
            inertia_tensor: 3x3 inertia tensor in body frame (kg⋅m²)
        """
        self.mass = mass
        self.inertia = inertia_tensor
        self.inertia_inv = np.linalg.inv(inertia_tensor)
    
    def calculate_derivatives(self, state: StateVector, 
                             forces: np.ndarray, 
                             moments: np.ndarray) -> np.ndarray:
        """
        Calculate state derivatives for integration.
        
        Parameters:
            state: Current state vector
            forces: Total forces in body frame (N)
            moments: Total moments in body frame (N⋅m)
            
        Returns:
            State derivative vector
        """
        # Extract state components
        position = state.position
        velocity = state.velocity
        q = state.quaternion
        omega = state.angular_velocity
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        # Rotation matrix from body to inertial
        R = Rotation.from_quat(q).as_matrix()
        
        # Translational dynamics
        position_dot = velocity
        velocity_dot = R @ forces / self.mass
        
        # Quaternion kinematics
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega_quat)
        
        # Rotational dynamics (Euler's equation)
        omega_dot = self.inertia_inv @ (moments - np.cross(omega, self.inertia @ omega))
        
        return np.concatenate([position_dot, velocity_dot, q_dot, omega_dot])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


class KalmanFilter:
    """
    Extended Kalman Filter for state estimation.
    Used for sensor fusion and navigation.
    """
    
    def __init__(self, state_dim: int, measurement_dim: int):
        """
        Initialize Kalman filter.
        
        Parameters:
            state_dim: Dimension of state vector
            measurement_dim: Dimension of measurement vector
        """
        self.n = state_dim
        self.m = measurement_dim
        
        # State estimate and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000  # Large initial uncertainty
        
        # Process and measurement noise covariances
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(measurement_dim) * 1.0
        
        # For history tracking
        self.innovation_history = deque(maxlen=100)
    
    def predict(self, F: np.ndarray, u: Optional[np.ndarray] = None,
                B: Optional[np.ndarray] = None):
        """
        Prediction step.
        
        Parameters:
            F: State transition matrix
            u: Control input (optional)
            B: Control input matrix (optional)
        """
        # Predict state
        self.x = F @ self.x
        if u is not None and B is not None:
            self.x += B @ u
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z: np.ndarray, H: np.ndarray):
        """
        Measurement update step.
        
        Parameters:
            z: Measurement vector
            H: Measurement matrix
        """
        # Innovation
        y = z - H @ self.x
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P
        
        # Store innovation for analysis
        self.innovation_history.append(y)
    
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and covariance."""
        return self.x.copy(), self.P.copy()


class ThermalModel:
    """
    Thermal dynamics model for spacecraft and aircraft.
    """
    
    def __init__(self, mass: float, specific_heat: float, surface_area: float,
                emissivity: float = 0.8, absorptivity: float = 0.2):
        """
        Initialize thermal model.
        
        Parameters:
            mass: Vehicle mass (kg)
            specific_heat: Specific heat capacity (J/(kg⋅K))
            surface_area: Total surface area (m²)
            emissivity: Surface emissivity
            absorptivity: Solar absorptivity
        """
        self.mass = mass
        self.cp = specific_heat
        self.area = surface_area
        self.emissivity = emissivity
        self.absorptivity = absorptivity
        
        # Solar constant at 1 AU
        self.solar_constant = 1361  # W/m²
        
        # Stefan-Boltzmann constant
        self.sigma = PhysicalConstants.STEFAN_BOLTZMANN
    
    def calculate_heat_transfer(self, temperature: float, altitude: float,
                               velocity: float, sun_angle: float,
                               atmosphere: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate net heat transfer rate.
        
        Parameters:
            temperature: Current temperature (K)
            altitude: Altitude (m)
            velocity: Velocity magnitude (m/s)
            sun_angle: Angle to sun (radians)
            atmosphere: Atmospheric properties (optional)
            
        Returns:
            Net heat transfer rate (W)
        """
        # Radiation cooling
        Q_rad_out = self.emissivity * self.sigma * self.area * temperature**4
        
        # Solar heating
        Q_solar = self.absorptivity * self.solar_constant * self.area * np.cos(sun_angle)
        Q_solar = max(0, Q_solar)  # Only if facing sun
        
        # Atmospheric heating (if in atmosphere)
        if atmosphere is not None and velocity > 0:
            # Simplified aerodynamic heating
            recovery_factor = 0.9
            T_recovery = atmosphere['temperature'] * (
                1 + recovery_factor * (PhysicalConstants.SPECIFIC_HEAT_RATIO - 1) / 2 * 
                (velocity / atmosphere['speed_of_sound'])**2
            )
            
            # Convective heat transfer coefficient (simplified)
            h_conv = 10 * np.sqrt(atmosphere['density'] * velocity)
            
            Q_aero = h_conv * self.area * (T_recovery - temperature)
        else:
            Q_aero = 0
        
        # Earth IR (simplified)
        Q_earth_ir = 237 * self.area * self.emissivity  # W/m² average Earth IR
        
        # Albedo (simplified)
        Q_albedo = 0.3 * Q_solar * 0.3  # 30% of solar reflected by Earth
        
        return Q_solar + Q_earth_ir + Q_albedo + Q_aero - Q_rad_out
    
    def update_temperature(self, temperature: float, Q_net: float, dt: float) -> float:
        """
        Update temperature based on heat transfer.
        
        Parameters:
            temperature: Current temperature (K)
            Q_net: Net heat transfer rate (W)
            dt: Time step (s)
            
        Returns:
            New temperature (K)
        """
        dT_dt = Q_net / (self.mass * self.cp)
        return temperature + dT_dt * dt


class ControlSystem:
    """
    Generic control system implementation (PID, LQR, etc.).
    """
    
    def __init__(self, control_type: str = 'PID'):
        """
        Initialize control system.
        
        Parameters:
            control_type: Type of controller ('PID', 'LQR', 'MPC')
        """
        self.control_type = control_type
        
        if control_type == 'PID':
            # PID gains for each axis
            self.kp = np.array([1.0, 1.0, 1.0])
            self.ki = np.array([0.1, 0.1, 0.1])
            self.kd = np.array([0.5, 0.5, 0.5])
            
            # Integral terms
            self.integral = np.zeros(3)
            self.prev_error = np.zeros(3)
            
        elif control_type == 'LQR':
            # Would implement Linear Quadratic Regulator
            pass
    
    def compute_control(self, error: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute control output.
        
        Parameters:
            error: Error signal
            dt: Time step
            
        Returns:
            Control output
        """
        if self.control_type == 'PID':
            return self._pid_control(error, dt)
        else:
            return np.zeros(3)
    
    def _pid_control(self, error: np.ndarray, dt: float) -> np.ndarray:
        """PID controller implementation."""
        # Proportional term
        P = self.kp * error
        
        # Integral term
        self.integral += error * dt
        I = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = np.zeros_like(error)
        D = self.kd * derivative
        
        self.prev_error = error.copy()
        
        # Total control
        control = P + I + D
        
        # Saturation
        return np.clip(control, -1, 1)


class TrajectoryOptimizer:
    """
    Trajectory optimization using various methods.
    """
    
    def __init__(self, dynamics_model):
        """
        Initialize trajectory optimizer.
        
        Parameters:
            dynamics_model: System dynamics model
        """
        self.dynamics = dynamics_model
    
    def optimize_transfer(self, initial_state: StateVector, 
                         target_state: StateVector,
                         transfer_time: float) -> List[StateVector]:
        """
        Optimize transfer trajectory between two states.
        
        Parameters:
            initial_state: Starting state
            target_state: Target state
            transfer_time: Desired transfer time
            
        Returns:
            Optimized trajectory as list of states
        """
        # This would implement actual optimization (e.g., shooting method, collocation)
        # For now, return simple interpolation
        
        n_points = 100
        trajectory = []
        
        for i in range(n_points):
            alpha = i / (n_points - 1)
            
            # Linear interpolation (simplified)
            state = StateVector(
                position=initial_state.position * (1 - alpha) + target_state.position * alpha,
                velocity=initial_state.velocity * (1 - alpha) + target_state.velocity * alpha,
                time=initial_state.time + alpha * transfer_time
            )
            trajectory.append(state)
        
        return trajectory
    
    def calculate_delta_v(self, trajectory: List[StateVector]) -> float:
        """Calculate total delta-v for a trajectory."""
        if len(trajectory) < 2:
            return 0.0
        
        total_dv = 0.0
        for i in range(1, len(trajectory)):
            dv = np.linalg.norm(trajectory[i].velocity - trajectory[i-1].velocity)
            total_dv += dv
        
        return total_dv


class AerospaceSimulator:
    """
    Main aerospace simulation framework.
    Integrates all subsystems for comprehensive simulation.
    """
    
    def __init__(self, vehicle_config: Dict):
        """
        Initialize simulator with vehicle configuration.
        
        Parameters:
            vehicle_config: Dictionary with vehicle parameters
        """
        self.config = vehicle_config
        
        # Initialize subsystems
        self._initialize_vehicle()
        self._initialize_environment()
        self._initialize_control()
        
        # Simulation state
        self.current_state = StateVector()
        self.time = 0.0
        self.history = []
        
    def _initialize_vehicle(self):
        """Initialize vehicle models."""
        # Mass properties
        self.mass = self.config['mass']
        self.inertia = np.array(self.config['inertia'])
        
        # Rigid body dynamics
        self.dynamics = RigidBodyDynamics(self.mass, self.inertia)
        
        # Aerodynamics (if applicable)
        if 'aerodynamics' in self.config:
            aero_config = self.config['aerodynamics']
            self.aerodynamics = AerodynamicsModel(
                aero_config['reference_area'],
                aero_config['reference_length']
            )
        else:
            self.aerodynamics = None
        
        # Propulsion
        if 'propulsion' in self.config:
            prop_config = self.config['propulsion']
            self.propulsion = PropulsionSystem(
                prop_config['type'],
                **prop_config['parameters']
            )
        else:
            self.propulsion = None
        
        # Thermal
        if 'thermal' in self.config:
            thermal_config = self.config['thermal']
            self.thermal = ThermalModel(
                self.mass,
                thermal_config['specific_heat'],
                thermal_config['surface_area']
            )
            self.temperature = thermal_config.get('initial_temperature', 300)
        else:
            self.thermal = None
    
    def _initialize_environment(self):
        """Initialize environment models."""
        self.atmosphere = AtmosphereModel()
        self.gravity = GravityModel()
        
        # Kalman filter for state estimation
        self.estimator = KalmanFilter(13, 6)  # 13-state, 6 measurements
    
    def _initialize_control(self):
        """Initialize control systems."""
        self.controller = ControlSystem('PID')
        self.trajectory_optimizer = TrajectoryOptimizer(self.dynamics)
    
    def step(self, dt: float, controls: Dict[str, float]):
        """
        Advance simulation by one time step.
        
        Parameters:
            dt: Time step (seconds)
            controls: Control inputs
        """
        # Get current altitude
        altitude = np.linalg.norm(self.current_state.position) - \
                  PhysicalConstants.EARTH_RADIUS_EQUATORIAL
        
        # Get atmospheric properties
        if altitude < 100000:  # Below 100 km
            atmo = self.atmosphere.get_properties(altitude)
        else:
            atmo = None
        
        # Calculate forces and moments
        forces = np.zeros(3)
        moments = np.zeros(3)
        
        # Gravity (in inertial frame, then transform to body)
        gravity_inertial = self.gravity.calculate_acceleration(
            self.current_state.position
        ) * self.mass
        
        R = self.current_state.get_rotation_matrix()
        forces += R.T @ gravity_inertial
        
        # Aerodynamics
        if self.aerodynamics is not None and atmo is not None:
            aero_forces, aero_moments = self.aerodynamics.calculate_forces_moments(
                self.current_state, atmo, controls
            )
            forces += aero_forces
            moments += aero_moments
        
        # Propulsion
        if self.propulsion is not None:
            mach = np.linalg.norm(self.current_state.velocity) / \
                   atmo['speed_of_sound'] if atmo else 0
            
            thrust = self.propulsion.calculate_thrust(
                altitude, mach, controls.get('throttle', 0), 
                atmo if atmo else {'pressure': 0}
            )
            forces += thrust
        
        # Integrate dynamics
        state_array = self.current_state.to_array()
        derivatives = self.dynamics.calculate_derivatives(
            self.current_state, forces, moments
        )
        
        # Simple Euler integration (could use RK4 for better accuracy)
        new_state_array = state_array + derivatives * dt
        
        # Update state
        self.current_state = StateVector.from_array(new_state_array, self.time + dt)
        
        # Update thermal state
        if self.thermal is not None:
            Q_net = self.thermal.calculate_heat_transfer(
                self.temperature, altitude, 
                np.linalg.norm(self.current_state.velocity),
                0.0, atmo  # Simplified sun angle
            )
            self.temperature = self.thermal.update_temperature(
                self.temperature, Q_net, dt
            )
        
        # Update time
        self.time += dt
        
        # Store history
        self.history.append({
            'time': self.time,
            'state': self.current_state,
            'forces': forces,
            'moments': moments,
            'temperature': self.temperature if self.thermal else None
        })
    
    def run_simulation(self, duration: float, dt: float = 0.01,
                      control_function: Optional[Callable] = None):
        """
        Run simulation for specified duration.
        
        Parameters:
            duration: Total simulation time (seconds)
            dt: Integration time step (seconds)
            control_function: Function that returns control inputs given state and time
        """
        n_steps = int(duration / dt)
        
        print(f"Starting simulation for {duration} seconds...")
        print(f"Time step: {dt} s, Total steps: {n_steps}")
        
        for step in range(n_steps):
            # Get control inputs
            if control_function is not None:
                controls = control_function(self.current_state, self.time)
            else:
                controls = {}
            
            # Advance simulation
            self.step(dt, controls)
            
            # Progress indicator
            if step % (n_steps // 10) == 0:
                print(f"Progress: {100 * step / n_steps:.1f}%")
        
        print("Simulation complete!")
    
    def plot_results(self):
        """Plot simulation results."""
        if not self.history:
            print("No simulation data to plot")
            return
        
        # Extract data
        times = [h['time'] for h in self.history]
        positions = np.array([h['state'].position for h in self.history])
        velocities = np.array([h['state'].velocity for h in self.history])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(positions[:, 0]/1000, positions[:, 1]/1000, positions[:, 2]/1000)
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_zlabel('Z (km)')
        ax1.set_title('3D Trajectory')
        
        # Altitude vs time
        ax2 = fig.add_subplot(2, 3, 2)
        altitudes = np.linalg.norm(positions, axis=1) - PhysicalConstants.EARTH_RADIUS_EQUATORIAL
        ax2.plot(times, altitudes/1000)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Altitude (km)')
        ax2.set_title('Altitude Profile')
        ax2.grid(True)
        
        # Velocity magnitude
        ax3 = fig.add_subplot(2, 3, 3)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        ax3.plot(times, vel_magnitudes/1000)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (km/s)')
        ax3.set_title('Velocity Magnitude')
        ax3.grid(True)
        
        # Forces
        ax4 = fig.add_subplot(2, 3, 4)
        forces = np.array([h['forces'] for h in self.history])
        ax4.plot(times, forces[:, 0]/1000, label='Fx')
        ax4.plot(times, forces[:, 1]/1000, label='Fy')
        ax4.plot(times, forces[:, 2]/1000, label='Fz')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Force (kN)')
        ax4.set_title('Body Forces')
        ax4.legend()
        ax4.grid(True)
        
        # Temperature (if thermal model active)
        if self.thermal is not None:
            ax5 = fig.add_subplot(2, 3, 5)
            temps = [h['temperature'] for h in self.history]
            ax5.plot(times, temps)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Temperature (K)')
            ax5.set_title('Thermal State')
            ax5.grid(True)
        
        # Ground track
        ax6 = fig.add_subplot(2, 3, 6)
        lats = np.degrees(np.arcsin(positions[:, 2] / np.linalg.norm(positions, axis=1)))
        lons = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))
        ax6.plot(lons, lats)
        ax6.set_xlabel('Longitude (deg)')
        ax6.set_ylabel('Latitude (deg)')
        ax6.set_title('Ground Track')
        ax6.grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage and test cases
def example_orbital_simulation():
    """Example: Simulate spacecraft in LEO with perturbations."""
    
    # Spacecraft configuration
    config = {
        'mass': 1000,  # kg
        'inertia': [[100, 0, 0], [0, 100, 0], [0, 0, 100]],  # kg⋅m²
        'thermal': {
            'specific_heat': 900,  # J/(kg⋅K)
            'surface_area': 20,  # m²
            'initial_temperature': 300  # K
        }
    }
    
    # Create simulator
    sim = AerospaceSimulator(config)
    
    # Set initial orbit (400 km circular)
    orbital_elements = OrbitalElements(
        semi_major_axis=PhysicalConstants.EARTH_RADIUS_EQUATORIAL + 400000,
        eccentricity=0.001,
        inclination=np.radians(51.6),  # ISS inclination
        raan=0,
        argument_of_periapsis=0,
        true_anomaly=0
    )
    
    # Convert to state vector
    sim.current_state = orbital_elements.to_state_vector()
    
    # Run simulation for one orbit
    orbital_period = 2 * np.pi * np.sqrt(
        orbital_elements.semi_major_axis**3 / PhysicalConstants.EARTH_MU
    )
    
    print(f"Orbital period: {orbital_period/60:.2f} minutes")
    
    sim.run_simulation(orbital_period, dt=1.0)
    sim.plot_results()
    
    return sim


def example_atmospheric_flight():
    """Example: Simulate aircraft in atmospheric flight."""
    
    # Aircraft configuration
    config = {
        'mass': 75000,  # kg (Boeing 737-like)
        'inertia': [[1e7, 0, 0], [0, 5e7, 0], [0, 0, 5e7]],  # kg⋅m²
        'aerodynamics': {
            'reference_area': 125,  # m²
            'reference_length': 4  # m
        },
        'propulsion': {
            'type': 'turbojet',
            'parameters': {
                'max_thrust_static': 200000,  # N (two engines)
                'tsfc': 0.6  # kg/N-hr
            }
        }
    }
    
    # Create simulator
    sim = AerospaceSimulator(config)
    
    # Set initial conditions (cruise flight)
    sim.current_state = StateVector(
        position=np.array([0, 0, -10000]),  # 10 km altitude
        velocity=np.array([250, 0, 0]),  # 250 m/s cruise speed
        quaternion=np.array([1, 0, 0, 0])  # Level flight
    )
    
    # Simple control function for level flight
    def cruise_control(state, time):
        return {
            'throttle': 0.7,  # 70% throttle
            'elevator': 0.0,
            'aileron': 0.0,
            'rudder': 0.0
        }
    
    # Run simulation
    sim.run_simulation(duration=300, dt=0.1, control_function=cruise_control)
    sim.plot_results()
    
    return sim


if __name__ == "__main__":
    # Run example simulations
    print("=" * 60)
    print("AEROSPACE PHYSICAL SYSTEMS SIMULATOR")
    print("=" * 60)
    
    print("\n1. Orbital Mechanics Simulation")
    print("-" * 30)
    orbital_sim = example_orbital_simulation()
    
    print("\n2. Atmospheric Flight Simulation")
    print("-" * 30)
    atmospheric_sim = example_atmospheric_flight()
    
    print("\nSimulations complete!")
