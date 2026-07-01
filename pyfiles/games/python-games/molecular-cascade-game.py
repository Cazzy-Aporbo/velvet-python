"""
MOLECULAR CASCADE - A Strategic Chemistry Learning Game
A unique educational game where players manipulate electron shells and molecular bonds
to create chain reactions that teach real chemistry concepts through gameplay.

The core mechanic involves "electron surfing" where players guide electrons through
orbital shells to create specific molecules, triggering cascading reactions that
score points based on chemical accuracy and reaction efficiency.

Author: Advanced Game Design System
Version: 1.0.0
Python Requirements: 3.8+
External Requirements: pygame (pip install pygame)
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pygame

# Initialize Pygame
pygame.init()


class ElementType(Enum):
    """Enumeration of chemical elements used in the game"""
    HYDROGEN = ("H", 1, 1, (255, 255, 255))
    CARBON = ("C", 6, 4, (50, 50, 50))
    OXYGEN = ("O", 8, 2, (255, 100, 100))
    NITROGEN = ("N", 7, 3, (100, 100, 255))
    PHOSPHORUS = ("P", 15, 5, (255, 150, 50))
    SULFUR = ("S", 16, 2, (255, 255, 100))

    def __init__(self, symbol, atomic_number, valence, color):
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.valence = valence
        self.color = color


@dataclass
class Electron:
    """Represents an electron that can be manipulated by the player"""
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    energy_level: int  # 1-4 representing s, p, d, f orbitals
    spin: int  # -1 or 1 for spin up/down
    paired: bool
    parent_atom: Optional['Atom'] = None

    def update_position(self, dt: float):
        """Update electron position based on velocity and time delta"""
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt

        # Apply quantum uncertainty (small random movements)
        self.x += random.gauss(0, 0.5)
        self.y += random.gauss(0, 0.5)


class Atom:
    """Represents an atom with electron shells and bonding capabilities"""

    def __init__(self, element: ElementType, x: float, y: float):
        self.element = element
        self.x = x
        self.y = y
        self.electrons: list[Electron] = []
        self.bonds: list[Bond] = []
        self.shell_radii = [30, 50, 70, 90]  # Pixel radii for electron shells
        self.stability_score = 0.0
        self.excitation_level = 0
        self.can_react = True
        self.reaction_cooldown = 0

        # Initialize with correct number of electrons
        self._initialize_electrons()

    def _initialize_electrons(self):
        """Place electrons in appropriate shells following Aufbau principle"""
        electrons_to_place = self.element.atomic_number

        # Simplified electron configuration (for game purposes)
        shells_capacity = [2, 8, 8, 18]  # Simplified for gameplay

        for shell_index, capacity in enumerate(shells_capacity):
            if electrons_to_place <= 0:
                break

            electrons_in_shell = min(electrons_to_place, capacity)
            angle_step = 2 * math.pi / electrons_in_shell

            for i in range(electrons_in_shell):
                angle = i * angle_step
                radius = self.shell_radii[shell_index]
                electron_x = self.x + radius * math.cos(angle)
                electron_y = self.y + radius * math.sin(angle)

                electron = Electron(
                    x=electron_x,
                    y=electron_y,
                    velocity_x=random.uniform(-10, 10),
                    velocity_y=random.uniform(-10, 10),
                    energy_level=shell_index + 1,
                    spin=1 if i % 2 == 0 else -1,
                    paired=(i % 2 == 1),
                    parent_atom=self
                )
                self.electrons.append(electron)

            electrons_to_place -= electrons_in_shell

    def calculate_stability(self) -> float:
        """Calculate atom stability based on electron configuration"""
        # Check octet rule satisfaction
        valence_electrons = self.get_valence_electrons()
        ideal_valence = 8 if self.element != ElementType.HYDROGEN else 2

        stability = 1.0 - abs(ideal_valence - len(valence_electrons)) / ideal_valence

        # Bonus for complete shells
        if len(valence_electrons) == ideal_valence:
            stability += 0.5

        # Penalty for unpaired electrons
        unpaired_count = sum(1 for e in valence_electrons if not e.paired)
        stability -= unpaired_count * 0.1

        self.stability_score = max(0, min(1, stability))
        return self.stability_score

    def get_valence_electrons(self) -> list[Electron]:
        """Return list of valence (outermost shell) electrons"""
        if not self.electrons:
            return []

        max_level = max(e.energy_level for e in self.electrons)
        return [e for e in self.electrons if e.energy_level == max_level]

    def can_bond_with(self, other: 'Atom') -> bool:
        """Check if this atom can form a bond with another atom"""
        if self == other:
            return False

        # Check if atoms are close enough
        distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        if distance > 150:  # Maximum bonding distance
            return False

        # Check if both atoms have available valence slots
        my_available = self.element.valence - len(self.bonds)
        other_available = other.element.valence - len(other.bonds)

        return my_available > 0 and other_available > 0


class Bond:
    """Represents a chemical bond between two atoms"""

    def __init__(self, atom1: Atom, atom2: Atom, bond_type: str = "single"):
        self.atom1 = atom1
        self.atom2 = atom2
        self.bond_type = bond_type  # "single", "double", "triple"
        self.strength = {"single": 1.0, "double": 1.5, "triple": 2.0}[bond_type]
        self.electrons_shared = {"single": 2, "double": 4, "triple": 6}[bond_type]
        self.resonance_energy = 0
        self.is_breaking = False

    def update_resonance(self):
        """Calculate resonance energy for delocalized electrons"""
        # Simplified resonance calculation for aromatic compounds
        if len(self.atom1.bonds) >= 3 and len(self.atom2.bonds) >= 3:
            self.resonance_energy = 0.3
        else:
            self.resonance_energy = 0


class Molecule:
    """Represents a complete molecule formed from bonded atoms"""

    def __init__(self, atoms: list[Atom]):
        self.atoms = atoms
        self.bonds = []
        self.molecular_formula = self._calculate_formula()
        self.stability = self._calculate_stability()
        self.is_organic = self._check_if_organic()
        self.special_properties = self._identify_special_properties()

    def _calculate_formula(self) -> str:
        """Generate molecular formula string"""
        element_counts = {}
        for atom in self.atoms:
            symbol = atom.element.symbol
            element_counts[symbol] = element_counts.get(symbol, 0) + 1

        formula = ""
        # Order: C, H, then alphabetical for others
        for symbol in ["C", "H"] + sorted(set(element_counts.keys()) - {"C", "H"}):
            if symbol in element_counts:
                count = element_counts[symbol]
                formula += symbol + (str(count) if count > 1 else "")

        return formula

    def _calculate_stability(self) -> float:
        """Calculate overall molecular stability"""
        if not self.atoms:
            return 0

        total_stability = sum(atom.calculate_stability() for atom in self.atoms)
        return total_stability / len(self.atoms)

    def _check_if_organic(self) -> bool:
        """Check if molecule contains carbon-hydrogen bonds"""
        has_carbon = any(atom.element == ElementType.CARBON for atom in self.atoms)
        has_hydrogen = any(atom.element == ElementType.HYDROGEN for atom in self.atoms)
        return has_carbon and has_hydrogen

    def _identify_special_properties(self) -> dict[str, bool]:
        """Identify special chemical properties for bonus points"""
        properties = {
            "aromatic": False,
            "acidic": False,
            "basic": False,
            "polar": False,
            "symmetrical": False
        }

        # Check for aromaticity (simplified - just check for ring with alternating bonds)
        if len(self.atoms) >= 6:
            properties["aromatic"] = self._check_aromaticity()

        # Check for acidic groups (has oxygen-hydrogen)
        for atom in self.atoms:
            if atom.element == ElementType.OXYGEN:
                for bond in atom.bonds:
                    other = bond.atom2 if bond.atom1 == atom else bond.atom1
                    if other.element == ElementType.HYDROGEN:
                        properties["acidic"] = True

        # Check for basic groups (has nitrogen with lone pairs)
        properties["basic"] = any(
            atom.element == ElementType.NITROGEN and len(atom.bonds) < 3
            for atom in self.atoms
        )

        # Check polarity based on electronegativity differences
        properties["polar"] = self._check_polarity()

        return properties

    def _check_aromaticity(self) -> bool:
        """Simplified aromaticity check"""
        # Would implement Huckel's rule (4n+2 pi electrons) in full version
        carbon_count = sum(1 for atom in self.atoms if atom.element == ElementType.CARBON)
        return carbon_count == 6  # Benzene ring for now

    def _check_polarity(self) -> bool:
        """Check if molecule has significant polarity"""
        # Simplified - just check for different elements
        elements = set(atom.element for atom in self.atoms)
        return len(elements) > 1


class ReactionManager:
    """Manages chemical reactions and cascade effects"""

    def __init__(self):
        self.reaction_database = self._load_reaction_database()
        self.active_reactions = []
        self.reaction_history = []
        self.cascade_multiplier = 1.0

    def _load_reaction_database(self) -> dict[str, dict]:
        """Load database of possible chemical reactions"""
        # Simplified reaction database for game purposes
        return {
            "combustion": {
                "reactants": ["C", "O2"],
                "products": ["CO2"],
                "energy_released": 100,
                "cascade_potential": 0.8
            },
            "water_formation": {
                "reactants": ["H2", "O"],
                "products": ["H2O"],
                "energy_released": 50,
                "cascade_potential": 0.6
            },
            "ammonia_synthesis": {
                "reactants": ["N", "H3"],
                "products": ["NH3"],
                "energy_released": 30,
                "cascade_potential": 0.4
            },
            "acid_base": {
                "reactants": ["H+", "OH-"],
                "products": ["H2O"],
                "energy_released": 40,
                "cascade_potential": 0.7
            }
        }

    def check_for_reactions(self, molecules: list[Molecule]) -> list[dict]:
        """Check if any molecules can react together"""
        reactions = []

        for i, mol1 in enumerate(molecules):
            for mol2 in molecules[i+1:]:
                reaction = self._can_react(mol1, mol2)
                if reaction:
                    reactions.append({
                        "reactants": [mol1, mol2],
                        "type": reaction,
                        "products": self._calculate_products(mol1, mol2, reaction)
                    })

        return reactions

    def _can_react(self, mol1: Molecule, mol2: Molecule) -> str | None:
        """Determine if two molecules can react and what type of reaction"""
        # Simplified reaction checking
        formula1 = mol1.molecular_formula
        formula2 = mol2.molecular_formula

        # Check for combustion
        if ("C" in formula1 and "O2" in formula2) or ("C" in formula2 and "O2" in formula1):
            return "combustion"

        # Check for water formation
        if ("H2" in formula1 and "O" in formula2) or ("H2" in formula2 and "O" in formula1):
            return "water_formation"

        return None

    def _calculate_products(self, mol1: Molecule, mol2: Molecule, reaction_type: str) -> list[str]:
        """Calculate reaction products based on reaction type"""
        if reaction_type in self.reaction_database:
            return self.reaction_database[reaction_type]["products"]
        return []

    def trigger_cascade(self, initial_reaction: dict) -> int:
        """Trigger cascading reactions for bonus points"""
        cascade_score = 0
        cascade_level = 1

        # Each successful reaction can trigger nearby reactions
        self.cascade_multiplier = 1.0 + (cascade_level * 0.5)
        cascade_score = int(100 * self.cascade_multiplier)

        return cascade_score


class GameBoard:
    """Main game board managing all game objects"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.atoms: list[Atom] = []
        self.molecules: list[Molecule] = []
        self.free_electrons: list[Electron] = []
        self.reaction_manager = ReactionManager()
        self.score = 0
        self.level = 1
        self.target_molecule = None
        self.time_remaining = 120  # seconds per level
        self.cascade_active = False
        self.electron_field = self._initialize_electron_field()

    def _initialize_electron_field(self):
        """Create electromagnetic field that affects electron movement"""
        field = []
        for x in range(0, self.width, 50):
            for y in range(0, self.height, 50):
                field.append({
                    "x": x,
                    "y": y,
                    "strength": random.uniform(-1, 1),
                    "direction": random.uniform(0, 2 * math.pi)
                })
        return field

    def add_atom(self, element: ElementType, x: float, y: float) -> Atom:
        """Add a new atom to the board"""
        atom = Atom(element, x, y)
        self.atoms.append(atom)
        return atom

    def launch_electron(self, start_x: float, start_y: float,
                       target_x: float, target_y: float, energy: int):
        """Launch an electron from start position toward target with given energy"""
        # Calculate velocity based on target direction
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > 0:
            velocity_x = (dx / distance) * energy * 10
            velocity_y = (dy / distance) * energy * 10
        else:
            velocity_x = velocity_y = 0

        electron = Electron(
            x=start_x,
            y=start_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            energy_level=energy,
            spin=random.choice([-1, 1]),
            paired=False
        )

        self.free_electrons.append(electron)
        return electron

    def update(self, dt: float):
        """Update all game objects"""
        # Update free electrons
        for electron in self.free_electrons[:]:
            electron.update_position(dt)

            # Apply electromagnetic field effects
            self._apply_field_effects(electron)

            # Check for electron capture by atoms
            captured = self._check_electron_capture(electron)
            if captured:
                self.free_electrons.remove(electron)

        # Update atoms and check for bond formation
        self._update_bonding()

        # Check for completed molecules
        self._check_molecule_formation()

        # Process any reactions
        reactions = self.reaction_manager.check_for_reactions(self.molecules)
        for reaction in reactions:
            self._process_reaction(reaction)

        # Update time
        self.time_remaining -= dt

    def _apply_field_effects(self, electron: Electron):
        """Apply electromagnetic field effects to electron movement"""
        for field_point in self.electron_field:
            dx = electron.x - field_point["x"]
            dy = electron.y - field_point["y"]
            distance = math.sqrt(dx**2 + dy**2)

            if distance < 100 and distance > 0:
                # Apply force based on field strength and direction
                force = field_point["strength"] / (distance / 50)
                electron.velocity_x += force * math.cos(field_point["direction"])
                electron.velocity_y += force * math.sin(field_point["direction"])

    def _check_electron_capture(self, electron: Electron) -> bool:
        """Check if electron can be captured by an atom"""
        for atom in self.atoms:
            dx = electron.x - atom.x
            dy = electron.y - atom.y
            distance = math.sqrt(dx**2 + dy**2)

            # Check if electron is within capture radius
            max_shell_radius = atom.shell_radii[-1]
            if distance < max_shell_radius + 20:
                # Check if atom can accept electron (simplified)
                if len(atom.electrons) < atom.element.atomic_number + 1:
                    # Capture probability based on energy compatibility
                    energy_match = 1.0 / (1 + abs(electron.energy_level - 2))
                    if random.random() < energy_match:
                        electron.parent_atom = atom
                        atom.electrons.append(electron)
                        atom.excitation_level += 1
                        return True

        return False

    def _update_bonding(self):
        """Check and update chemical bonds between atoms"""
        for i, atom1 in enumerate(self.atoms):
            for atom2 in self.atoms[i+1:]:
                if atom1.can_bond_with(atom2):
                    # Check if electrons are aligned for bonding
                    if self._check_orbital_overlap(atom1, atom2):
                        bond = Bond(atom1, atom2)
                        atom1.bonds.append(bond)
                        atom2.bonds.append(bond)

                        # Score for bond formation
                        self.score += 10

    def _check_orbital_overlap(self, atom1: Atom, atom2: Atom) -> bool:
        """Check if electron orbitals overlap sufficiently for bonding"""
        # Simplified orbital overlap check
        distance = math.sqrt((atom1.x - atom2.x)**2 + (atom1.y - atom2.y)**2)
        overlap_threshold = atom1.shell_radii[-1] + atom2.shell_radii[-1]

        return distance < overlap_threshold

    def _check_molecule_formation(self):
        """Identify completed molecules from bonded atoms"""
        visited = set()

        for atom in self.atoms:
            if atom not in visited and atom.bonds:
                # Traverse connected atoms to find complete molecule
                molecule_atoms = self._traverse_molecule(atom, visited)
                if len(molecule_atoms) > 1:
                    molecule = Molecule(molecule_atoms)
                    self.molecules.append(molecule)

                    # Score based on molecule complexity and stability
                    complexity_score = len(molecule_atoms) * 20
                    stability_bonus = int(molecule.stability * 50)
                    property_bonus = sum(30 for prop in molecule.special_properties.values() if prop)

                    total_score = complexity_score + stability_bonus + property_bonus
                    self.score += total_score

                    # Check if target molecule achieved
                    if self.target_molecule and molecule.molecular_formula == self.target_molecule:
                        self.score += 500  # Level completion bonus
                        self.advance_level()

    def _traverse_molecule(self, start_atom: Atom, visited: set[Atom]) -> list[Atom]:
        """Traverse bonded atoms to identify complete molecule"""
        molecule_atoms = []
        stack = [start_atom]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                molecule_atoms.append(current)

                for bond in current.bonds:
                    other = bond.atom2 if bond.atom1 == current else bond.atom1
                    if other not in visited:
                        stack.append(other)

        return molecule_atoms

    def _process_reaction(self, reaction: dict):
        """Process a chemical reaction"""
        # Remove reactant molecules
        for reactant in reaction["reactants"]:
            if reactant in self.molecules:
                self.molecules.remove(reactant)

        # Add energy release effect
        energy_released = 100  # Base energy

        # Trigger cascade if conditions met
        if self.cascade_active:
            cascade_score = self.reaction_manager.trigger_cascade(reaction)
            self.score += cascade_score

        self.score += energy_released

    def advance_level(self):
        """Advance to next level with new target molecule"""
        self.level += 1
        self.time_remaining = 120 + (self.level * 10)  # More time for harder levels

        # Generate new target molecule based on level
        target_molecules = [
            "H2O",      # Level 1: Water
            "CO2",      # Level 2: Carbon dioxide
            "NH3",      # Level 3: Ammonia
            "CH4",      # Level 4: Methane
            "C2H5OH",   # Level 5: Ethanol
            "C6H12O6",  # Level 6: Glucose
            "C8H10N4O2" # Level 7: Caffeine
        ]

        if self.level <= len(target_molecules):
            self.target_molecule = target_molecules[self.level - 1]
        else:
            # Random complex molecule for higher levels
            self.target_molecule = self._generate_random_target()

    def _generate_random_target(self) -> str:
        """Generate random target molecule formula for advanced levels"""
        elements = ["C", "H", "O", "N", "P", "S"]
        formula = ""

        # Always start with carbon for organic molecules
        c_count = random.randint(3, 12)
        formula += f"C{c_count}"

        # Add hydrogen (usually about 2x carbon)
        h_count = random.randint(c_count, c_count * 3)
        formula += f"H{h_count}"

        # Randomly add other elements
        for element in elements[2:]:
            if random.random() < 0.3:
                count = random.randint(1, 4)
                formula += element + (str(count) if count > 1 else "")

        return formula


class MolecularCascadeGame:
    """Main game class handling display and user interaction"""

    def __init__(self):
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Molecular Cascade - Learn Chemistry Through Chain Reactions")

        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 48)

        self.board = GameBoard(self.screen_width, self.screen_height - 100)
        self.running = True
        self.paused = False

        # Player controls
        self.selected_element = ElementType.HYDROGEN
        self.electron_launcher_active = False
        self.launcher_start = None
        self.launcher_energy = 1

        # Visual effects
        self.particle_effects = []
        self.bond_animations = []

        # Tutorial state
        self.tutorial_step = 0
        self.show_tutorial = True

        # Initialize first level
        self._setup_level(1)

    def _setup_level(self, level: int):
        """Setup initial atoms and target for level"""
        self.board.level = level
        self.board.atoms.clear()
        self.board.molecules.clear()
        self.board.free_electrons.clear()

        # Place starting atoms based on level
        if level == 1:
            # Level 1: Make water
            self.board.add_atom(ElementType.HYDROGEN, 300, 400)
            self.board.add_atom(ElementType.HYDROGEN, 500, 400)
            self.board.add_atom(ElementType.OXYGEN, 400, 300)
            self.board.target_molecule = "H2O"

        elif level == 2:
            # Level 2: Make carbon dioxide
            self.board.add_atom(ElementType.CARBON, 400, 400)
            self.board.add_atom(ElementType.OXYGEN, 300, 400)
            self.board.add_atom(ElementType.OXYGEN, 500, 400)
            self.board.target_molecule = "CO2"

        else:
            # Random setup for higher levels
            for _ in range(level + 2):
                element = random.choice(list(ElementType))
                x = random.randint(100, self.screen_width - 100)
                y = random.randint(100, self.screen_height - 200)
                self.board.add_atom(element, x, y)

    def run(self):
        """Main game loop"""
        dt = 0

        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Convert to seconds

            self._handle_events()

            if not self.paused:
                self.board.update(dt)
                self._update_effects(dt)

            self._draw()

        pygame.quit()

    def _handle_events(self):
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

                # Element selection hotkeys
                elif event.key == pygame.K_1:
                    self.selected_element = ElementType.HYDROGEN
                elif event.key == pygame.K_2:
                    self.selected_element = ElementType.CARBON
                elif event.key == pygame.K_3:
                    self.selected_element = ElementType.OXYGEN
                elif event.key == pygame.K_4:
                    self.selected_element = ElementType.NITROGEN

                # Activate cascade mode
                elif event.key == pygame.K_c:
                    self.board.cascade_active = not self.board.cascade_active

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Place atom
                    x, y = pygame.mouse.get_pos()
                    if y < self.screen_height - 100:  # Not in UI area
                        self.board.add_atom(self.selected_element, x, y)
                        self._create_particle_effect(x, y, self.selected_element.color)

                elif event.button == 3:  # Right click
                    # Start electron launcher
                    self.electron_launcher_active = True
                    self.launcher_start = pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3 and self.electron_launcher_active:
                    # Launch electron
                    end_pos = pygame.mouse.get_pos()
                    if self.launcher_start:
                        self.board.launch_electron(
                            self.launcher_start[0], self.launcher_start[1],
                            end_pos[0], end_pos[1],
                            self.launcher_energy
                        )
                    self.electron_launcher_active = False
                    self.launcher_start = None

            elif event.type == pygame.MOUSEWHEEL:
                # Adjust electron launcher energy
                self.launcher_energy = max(1, min(4, self.launcher_energy + event.y))

    def _update_effects(self, dt: float):
        """Update visual effects"""
        # Update particle effects
        for effect in self.particle_effects[:]:
            effect["lifetime"] -= dt
            effect["y"] -= effect["velocity"] * dt
            effect["size"] *= 0.98

            if effect["lifetime"] <= 0 or effect["size"] < 1:
                self.particle_effects.remove(effect)

        # Update bond animations
        for anim in self.bond_animations[:]:
            anim["progress"] += dt
            if anim["progress"] >= 1.0:
                self.bond_animations.remove(anim)

    def _create_particle_effect(self, x: float, y: float, color: tuple[int, int, int]):
        """Create particle effect at position"""
        for _ in range(10):
            self.particle_effects.append({
                "x": x + random.randint(-20, 20),
                "y": y + random.randint(-20, 20),
                "velocity": random.uniform(20, 50),
                "size": random.uniform(2, 8),
                "color": color,
                "lifetime": random.uniform(0.5, 1.5)
            })

    def _draw(self):
        """Draw everything to screen"""
        # Clear screen with gradient background
        self._draw_gradient_background()

        # Draw electromagnetic field visualization
        self._draw_field()

        # Draw bonds
        for atom in self.board.atoms:
            for bond in atom.bonds:
                self._draw_bond(bond)

        # Draw atoms
        for atom in self.board.atoms:
            self._draw_atom(atom)

        # Draw free electrons
        for electron in self.board.free_electrons:
            self._draw_electron(electron)

        # Draw particle effects
        for effect in self.particle_effects:
            pygame.draw.circle(
                self.screen,
                effect["color"],
                (int(effect["x"]), int(effect["y"])),
                int(effect["size"])
            )

        # Draw electron launcher
        if self.electron_launcher_active and self.launcher_start:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(
                self.screen,
                (255, 255, 0),
                self.launcher_start,
                mouse_pos,
                2
            )
            # Draw energy indicator
            for i in range(self.launcher_energy):
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 0),
                    mouse_pos,
                    10 + i * 5,
                    1
                )

        # Draw UI
        self._draw_ui()

        # Draw tutorial if active
        if self.show_tutorial:
            self._draw_tutorial()

        pygame.display.flip()

    def _draw_gradient_background(self):
        """Draw gradient background"""
        for i in range(self.screen_height):
            color_value = int(20 + (i / self.screen_height) * 30)
            color = (color_value, color_value, color_value + 10)
            pygame.draw.line(self.screen, color, (0, i), (self.screen_width, i))

    def _draw_field(self):
        """Draw electromagnetic field visualization"""
        for field_point in self.board.electron_field:
            # Draw field lines
            strength = abs(field_point["strength"])
            if strength > 0.5:
                end_x = field_point["x"] + 20 * math.cos(field_point["direction"])
                end_y = field_point["y"] + 20 * math.sin(field_point["direction"])
                color = (50, 50, 100) if field_point["strength"] > 0 else (100, 50, 50)
                pygame.draw.line(
                    self.screen,
                    color,
                    (field_point["x"], field_point["y"]),
                    (end_x, end_y),
                    1
                )

    def _draw_atom(self, atom: Atom):
        """Draw an atom with its electron shells"""
        # Draw electron shells
        for radius in atom.shell_radii:
            pygame.draw.circle(
                self.screen,
                (50, 50, 50),
                (int(atom.x), int(atom.y)),
                radius,
                1
            )

        # Draw nucleus
        pygame.draw.circle(
            self.screen,
            atom.element.color,
            (int(atom.x), int(atom.y)),
            15
        )

        # Draw element symbol
        text = self.font_small.render(atom.element.symbol, True, (255, 255, 255))
        text_rect = text.get_rect(center=(int(atom.x), int(atom.y)))
        self.screen.blit(text, text_rect)

        # Draw electrons in shells
        for electron in atom.electrons:
            self._draw_electron(electron)

        # Draw stability indicator
        stability_color = (
            int(255 * (1 - atom.stability_score)),
            int(255 * atom.stability_score),
            0
        )
        pygame.draw.circle(
            self.screen,
            stability_color,
            (int(atom.x + 20), int(atom.y - 20)),
            5
        )

    def _draw_electron(self, electron: Electron):
        """Draw an electron with spin indicator"""
        color = (100, 100, 255) if electron.spin == 1 else (255, 100, 100)
        pygame.draw.circle(
            self.screen,
            color,
            (int(electron.x), int(electron.y)),
            3
        )

        # Draw spin arrow
        if electron.spin == 1:
            pygame.draw.line(
                self.screen,
                color,
                (int(electron.x), int(electron.y - 3)),
                (int(electron.x), int(electron.y - 6)),
                1
            )
        else:
            pygame.draw.line(
                self.screen,
                color,
                (int(electron.x), int(electron.y + 3)),
                (int(electron.x), int(electron.y + 6)),
                1
            )

    def _draw_bond(self, bond: Bond):
        """Draw a chemical bond between atoms"""
        # Calculate bond positions
        x1, y1 = bond.atom1.x, bond.atom1.y
        x2, y2 = bond.atom2.x, bond.atom2.y

        # Draw multiple lines for double/triple bonds
        if bond.bond_type == "single":
            pygame.draw.line(self.screen, (150, 150, 150), (x1, y1), (x2, y2), 2)
        elif bond.bond_type == "double":
            # Draw two parallel lines
            offset = 3
            angle = math.atan2(y2 - y1, x2 - x1) + math.pi / 2
            dx = offset * math.cos(angle)
            dy = offset * math.sin(angle)

            pygame.draw.line(
                self.screen, (150, 150, 150),
                (x1 + dx, y1 + dy), (x2 + dx, y2 + dy), 2
            )
            pygame.draw.line(
                self.screen, (150, 150, 150),
                (x1 - dx, y1 - dy), (x2 - dx, y2 - dy), 2
            )
        elif bond.bond_type == "triple":
            # Draw three parallel lines
            offset = 4
            angle = math.atan2(y2 - y1, x2 - x1) + math.pi / 2
            dx = offset * math.cos(angle)
            dy = offset * math.sin(angle)

            pygame.draw.line(self.screen, (150, 150, 150), (x1, y1), (x2, y2), 2)
            pygame.draw.line(
                self.screen, (150, 150, 150),
                (x1 + dx, y1 + dy), (x2 + dx, y2 + dy), 2
            )
            pygame.draw.line(
                self.screen, (150, 150, 150),
                (x1 - dx, y1 - dy), (x2 - dx, y2 - dy), 2
            )

        # Draw resonance indicator if applicable
        if bond.resonance_energy > 0:
            pygame.draw.circle(
                self.screen,
                (200, 200, 100),
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                5,
                1
            )

    def _draw_ui(self):
        """Draw user interface elements"""
        # Draw UI background
        pygame.draw.rect(
            self.screen,
            (30, 30, 40),
            (0, self.screen_height - 100, self.screen_width, 100)
        )

        # Draw score
        score_text = self.font_medium.render(f"Score: {self.board.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, self.screen_height - 80))

        # Draw level
        level_text = self.font_medium.render(f"Level: {self.board.level}", True, (255, 255, 255))
        self.screen.blit(level_text, (20, self.screen_height - 50))

        # Draw target molecule
        if self.board.target_molecule:
            target_text = self.font_medium.render(
                f"Target: {self.board.target_molecule}",
                True, (100, 255, 100)
            )
            self.screen.blit(target_text, (300, self.screen_height - 80))

        # Draw time remaining
        time_text = self.font_medium.render(
            f"Time: {int(self.board.time_remaining)}s",
            True, (255, 255, 255)
        )
        self.screen.blit(time_text, (300, self.screen_height - 50))

        # Draw selected element
        element_text = self.font_medium.render(
            f"Element: {self.selected_element.symbol}",
            True, self.selected_element.color
        )
        self.screen.blit(element_text, (600, self.screen_height - 80))

        # Draw electron energy
        energy_text = self.font_medium.render(
            f"E-Energy: {self.launcher_energy}",
            True, (255, 255, 0)
        )
        self.screen.blit(energy_text, (600, self.screen_height - 50))

        # Draw cascade mode indicator
        if self.board.cascade_active:
            cascade_text = self.font_medium.render("CASCADE MODE", True, (255, 100, 100))
            self.screen.blit(cascade_text, (900, self.screen_height - 80))

        # Draw controls hint
        controls_text = self.font_small.render(
            "LMB: Place Atom | RMB: Launch Electron | 1-4: Select Element | C: Cascade | TAB: Tutorial",
            True, (150, 150, 150)
        )
        self.screen.blit(controls_text, (20, self.screen_height - 20))

    def _draw_tutorial(self):
        """Draw tutorial overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Tutorial text
        tutorial_texts = [
            "Welcome to Molecular Cascade!",
            "",
            "OBJECTIVE: Create target molecules by manipulating atoms and electrons",
            "",
            "HOW TO PLAY:",
            "1. Left Click - Place atoms on the board",
            "2. Right Click + Drag - Launch electrons to excite atoms",
            "3. Number Keys (1-4) - Select different elements",
            "4. Mouse Wheel - Adjust electron energy level",
            "",
            "CHEMISTRY CONCEPTS:",
            "- Atoms bond when their electron shells overlap",
            "- Stable molecules have complete outer shells (octet rule)",
            "- Excited electrons can trigger chain reactions",
            "- Different molecules have special properties for bonus points",
            "",
            "SCORING:",
            "- Form bonds: 10 points",
            "- Complete molecules: 20 points per atom",
            "- Stability bonus: Up to 50 points",
            "- Special properties: 30 points each",
            "- Cascade reactions: Multiplied scores!",
            "",
            "Press TAB to close this tutorial",
            "Press C to activate CASCADE MODE for chain reactions!"
        ]

        y_offset = 100
        for text in tutorial_texts:
            if text == "":
                y_offset += 20
                continue

            rendered_text = self.font_small.render(text, True, (255, 255, 255))
            text_rect = rendered_text.get_rect(center=(self.screen_width // 2, y_offset))
            self.screen.blit(rendered_text, text_rect)
            y_offset += 30


# Main execution
if __name__ == "__main__":
    game = MolecularCascadeGame()
    game.run()
