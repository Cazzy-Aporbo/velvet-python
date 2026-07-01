"""
STELLAR CONSTELLATION UNIVERSE 
2025
Interactive Star Map with Constellations, Zodiac Signs, and Planetary Motion
Real star coordinates, zodiac constellations, planetary orbits,
constellation mythology, and spectacular cosmic visualizations
Cazzy Aporbo, MS: Where astronomy meets mythology in visual harmony. An Attempt
"""

import random
from collections import defaultdict, deque
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Cosmic Starfield Palette - Deep space with stellar colors
STELLAR_PALETTE = {
    'deep_space': '#000008',            # Deep space black
    'stellar_white': '#FFFFFF',         # White stars (O-type)
    'stellar_blue': '#9BB2FF',          # Blue stars (B-type)
    'stellar_blue_white': '#B7C5FF',    # Blue-white stars (A-type)
    'stellar_yellow_white': '#F8F7FF',  # Yellow-white stars (F-type)
    'stellar_yellow': '#FFF4EA',        # Yellow stars (G-type, like Sun)
    'stellar_orange': '#FFD2A1',        # Orange stars (K-type)
    'stellar_red': '#FFAD51',           # Red stars (M-type)
    'nebula_pink': '#FF69B4',           # Pink nebula
    'nebula_blue': '#4169E1',           # Blue nebula
    'nebula_green': '#32CD32',          # Green nebula
    'constellation_line': '#87CEEB',    # Sky blue constellation lines
    'zodiac_gold': '#FFD700',           # Gold for zodiac signs
    'planet_mercury': '#8C7853',        # Mercury color
    'planet_venus': '#FFC649',          # Venus color
    'planet_mars': '#CD5C5C',           # Mars color
    'planet_jupiter': '#D2691E',        # Jupiter color
    'planet_saturn': '#FAD5A5',         # Saturn color
    'planet_uranus': '#4FD0E7',         # Uranus color
    'planet_neptune': '#4B70DD',        # Neptune color
    'milky_way': '#E6E6FA',            # Milky Way band
    'galaxy_core': '#DDA0DD',          # Galaxy center
    'cosmic_dust': '#696969',          # Cosmic dust clouds
    'moon_silver': '#C0C0C0',          # Moon color
    'aurora_green': '#00FF7F',         # Aurora effects
    'meteor_trail': '#FF4500'          # Meteor colors
}

# Real star data (sample of brightest stars with coordinates)
STAR_CATALOG = {
    'Sirius': {'ra': 101.287, 'dec': -16.716, 'mag': -1.46, 'color': 'stellar_white', 'constellation': 'Canis Major'},
    'Canopus': {'ra': 95.988, 'dec': -52.696, 'mag': -0.74, 'color': 'stellar_yellow_white', 'constellation': 'Carina'},
    'Arcturus': {'ra': 213.915, 'dec': 19.182, 'mag': -0.05, 'color': 'stellar_orange', 'constellation': 'Bootes'},
    'Vega': {'ra': 279.234, 'dec': 38.784, 'mag': 0.03, 'color': 'stellar_blue_white', 'constellation': 'Lyra'},
    'Capella': {'ra': 79.172, 'dec': 45.998, 'mag': 0.08, 'color': 'stellar_yellow', 'constellation': 'Auriga'},
    'Rigel': {'ra': 78.634, 'dec': -8.202, 'mag': 0.13, 'color': 'stellar_blue', 'constellation': 'Orion'},
    'Procyon': {'ra': 114.825, 'dec': 5.225, 'mag': 0.34, 'color': 'stellar_yellow_white', 'constellation': 'Canis Minor'},
    'Betelgeuse': {'ra': 88.793, 'dec': 7.407, 'mag': 0.50, 'color': 'stellar_red', 'constellation': 'Orion'},
    'Achernar': {'ra': 24.430, 'dec': -57.237, 'mag': 0.46, 'color': 'stellar_blue', 'constellation': 'Eridanus'},
    'Hadar': {'ra': 210.956, 'dec': -60.373, 'mag': 0.61, 'color': 'stellar_blue', 'constellation': 'Centaurus'},
    'Altair': {'ra': 297.696, 'dec': 8.868, 'mag': 0.77, 'color': 'stellar_white', 'constellation': 'Aquila'},
    'Aldebaran': {'ra': 68.980, 'dec': 16.509, 'mag': 0.85, 'color': 'stellar_orange', 'constellation': 'Taurus'},
    'Spica': {'ra': 201.298, 'dec': -11.161, 'mag': 0.97, 'color': 'stellar_blue', 'constellation': 'Virgo'},
    'Antares': {'ra': 247.352, 'dec': -26.432, 'mag': 1.09, 'color': 'stellar_red', 'constellation': 'Scorpius'},
    'Pollux': {'ra': 116.329, 'dec': 28.026, 'mag': 1.14, 'color': 'stellar_orange', 'constellation': 'Gemini'},
    'Fomalhaut': {'ra': 344.413, 'dec': -29.622, 'mag': 1.16, 'color': 'stellar_white', 'constellation': 'Piscis Austrinus'},
    'Deneb': {'ra': 310.358, 'dec': 45.280, 'mag': 1.25, 'color': 'stellar_white', 'constellation': 'Cygnus'},
    'Regulus': {'ra': 152.093, 'dec': 11.967, 'mag': 1.35, 'color': 'stellar_blue_white', 'constellation': 'Leo'},
    'Castor': {'ra': 113.650, 'dec': 31.888, 'mag': 1.57, 'color': 'stellar_white', 'constellation': 'Gemini'},
    'Bellatrix': {'ra': 81.283, 'dec': 6.350, 'mag': 1.64, 'color': 'stellar_blue', 'constellation': 'Orion'}
}

# Zodiac constellation data with mythological information
ZODIAC_CONSTELLATIONS = {
    'Aries': {
        'symbol': '♈', 'element': 'Fire', 'dates': 'Mar 21 - Apr 19',
        'mythology': 'The Golden Ram', 'brightest_star': 'Hamal',
        'key_stars': ['Hamal', 'Sheratan', 'Mesarthim'],
        'coordinates': [(30.0, 20.0), (35.0, 25.0), (28.0, 18.0)]
    },
    'Taurus': {
        'symbol': '♉', 'element': 'Earth', 'dates': 'Apr 20 - May 20',
        'mythology': 'The Bull', 'brightest_star': 'Aldebaran',
        'key_stars': ['Aldebaran', 'Elnath', 'Alcyone'],
        'coordinates': [(68.0, 16.0), (81.0, 28.0), (56.0, 24.0)]
    },
    'Gemini': {
        'symbol': '♊', 'element': 'Air', 'dates': 'May 21 - Jun 20',
        'mythology': 'The Twins Castor and Pollux', 'brightest_star': 'Pollux',
        'key_stars': ['Pollux', 'Castor', 'Alhena'],
        'coordinates': [(116.0, 28.0), (113.0, 32.0), (99.0, 16.0)]
    },
    'Cancer': {
        'symbol': '♋', 'element': 'Water', 'dates': 'Jun 21 - Jul 22',
        'mythology': 'The Crab', 'brightest_star': 'Tarf',
        'key_stars': ['Tarf', 'Acubens', 'Al Tarf'],
        'coordinates': [(130.0, 21.0), (134.0, 11.0), (143.0, 9.0)]
    },
    'Leo': {
        'symbol': '♌', 'element': 'Fire', 'dates': 'Jul 23 - Aug 22',
        'mythology': 'The Lion', 'brightest_star': 'Regulus',
        'key_stars': ['Regulus', 'Denebola', 'Algieba'],
        'coordinates': [(152.0, 12.0), (177.0, 14.0), (154.0, 20.0)]
    },
    'Virgo': {
        'symbol': '♍', 'element': 'Earth', 'dates': 'Aug 23 - Sep 22',
        'mythology': 'The Maiden', 'brightest_star': 'Spica',
        'key_stars': ['Spica', 'Zavijava', 'Porrima'],
        'coordinates': [(201.0, -11.0), (188.0, 3.0), (193.0, -1.0)]
    },
    'Libra': {
        'symbol': '♎', 'element': 'Air', 'dates': 'Sep 23 - Oct 22',
        'mythology': 'The Scales', 'brightest_star': 'Zubeneschamali',
        'key_stars': ['Zubeneschamali', 'Zubenelgenubi', 'Brachium'],
        'coordinates': [(229.0, -9.0), (222.0, -16.0), (240.0, -14.0)]
    },
    'Scorpius': {
        'symbol': '♏', 'element': 'Water', 'dates': 'Oct 23 - Nov 21',
        'mythology': 'The Scorpion', 'brightest_star': 'Antares',
        'key_stars': ['Antares', 'Shaula', 'Sargas'],
        'coordinates': [(247.0, -26.0), (263.0, -37.0), (250.0, -43.0)]
    },
    'Sagittarius': {
        'symbol': '♐', 'element': 'Fire', 'dates': 'Nov 22 - Dec 21',
        'mythology': 'The Archer', 'brightest_star': 'Kaus Australis',
        'key_stars': ['Kaus Australis', 'Nunki', 'Ascella'],
        'coordinates': [(276.0, -34.0), (283.0, -26.0), (290.0, -30.0)]
    },
    'Capricornus': {
        'symbol': '♑', 'element': 'Earth', 'dates': 'Dec 22 - Jan 19',
        'mythology': 'The Sea-Goat', 'brightest_star': 'Deneb Algedi',
        'key_stars': ['Deneb Algedi', 'Dabih', 'Nashira'],
        'coordinates': [(322.0, -16.0), (305.0, -14.0), (315.0, -17.0)]
    },
    'Aquarius': {
        'symbol': '♒', 'element': 'Air', 'dates': 'Jan 20 - Feb 18',
        'mythology': 'The Water Bearer', 'brightest_star': 'Sadalsuud',
        'key_stars': ['Sadalsuud', 'Sadalmelik', 'Sadachbia'],
        'coordinates': [(322.0, -5.0), (331.0, -0.3), (334.0, -9.0)]
    },
    'Pisces': {
        'symbol': '♓', 'element': 'Water', 'dates': 'Feb 19 - Mar 20',
        'mythology': 'The Fishes', 'brightest_star': 'Alrescha',
        'key_stars': ['Alrescha', 'Fum al Samakah', 'Revati'],
        'coordinates': [(29.0, 2.0), (349.0, 3.0), (359.0, 7.0)]
    }
}

# Planet orbital data (simplified)
PLANET_DATA = {
    'Mercury': {'period': 88, 'distance': 0.39, 'color': 'planet_mercury', 'size': 4},
    'Venus': {'period': 225, 'distance': 0.72, 'color': 'planet_venus', 'size': 6},
    'Mars': {'period': 687, 'distance': 1.52, 'color': 'planet_mars', 'size': 5},
    'Jupiter': {'period': 4333, 'distance': 5.20, 'color': 'planet_jupiter', 'size': 12},
    'Saturn': {'period': 10759, 'distance': 9.58, 'color': 'planet_saturn', 'size': 10},
    'Uranus': {'period': 30687, 'distance': 19.22, 'color': 'planet_uranus', 'size': 8},
    'Neptune': {'period': 60190, 'distance': 30.05, 'color': 'planet_neptune', 'size': 8}
}

@dataclass
class Star:
    """Individual star with astronomical properties"""

    name: str
    ra: float  # Right ascension in degrees
    dec: float  # Declination in degrees
    magnitude: float  # Apparent magnitude (brightness)
    color: str
    constellation: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    twinkle_phase: float = 0.0

    def __post_init__(self):
        # Convert spherical coordinates to Cartesian for 3D visualization
        ra_rad = np.radians(self.ra)
        dec_rad = np.radians(self.dec)

        # Project onto celestial sphere
        self.x = np.cos(dec_rad) * np.cos(ra_rad)
        self.y = np.cos(dec_rad) * np.sin(ra_rad)
        self.z = np.sin(dec_rad)

        self.twinkle_phase = random.uniform(0, 2*np.pi)

    def get_size(self) -> float:
        """Calculate star size based on magnitude"""
        # Brighter stars (lower magnitude) are larger
        return max(5, 50 * (2.5 ** (-self.magnitude)))

    def get_brightness(self, time: float) -> float:
        """Calculate twinkling brightness"""
        base_brightness = 1.0 / (1 + max(0, self.magnitude))
        twinkle = 0.1 * np.sin(time * 3 + self.twinkle_phase)
        return min(1.0, base_brightness + twinkle)


class Constellation:
    """Constellation with star patterns and mythology"""

    def __init__(self, name: str, stars: list[Star], mythology: str = ""):
        self.name = name
        self.stars = stars
        self.mythology = mythology
        self.lines = self._generate_constellation_lines()
        self.is_zodiac = name in ZODIAC_CONSTELLATIONS
        self.visibility = 1.0

    def _generate_constellation_lines(self) -> list[tuple[Star, Star]]:
        """Generate lines connecting stars to form constellation pattern"""
        lines = []

        # For major constellations, create realistic star patterns
        if self.name == 'Orion':
            # Connect stars to form Orion's distinctive shape
            if len(self.stars) >= 7:
                # Belt stars
                lines.append((self.stars[0], self.stars[1]))
                lines.append((self.stars[1], self.stars[2]))
                # Shoulders and body
                lines.append((self.stars[3], self.stars[0]))
                lines.append((self.stars[4], self.stars[2]))
                # Legs
                lines.append((self.stars[5], self.stars[6]))

        elif self.name == 'Ursa Major':
            # Big Dipper pattern
            if len(self.stars) >= 7:
                for i in range(len(self.stars) - 1):
                    lines.append((self.stars[i], self.stars[i + 1]))

        else:
            # Generic constellation pattern - connect nearby stars
            for i in range(len(self.stars)):
                for j in range(i + 1, min(i + 3, len(self.stars))):
                    lines.append((self.stars[i], self.stars[j]))

        return lines

    def update_visibility(self, time: float, latitude: float = 40.0):
        """Update constellation visibility based on time and location"""
        # Simplified visibility calculation
        # In reality, this would depend on local sidereal time
        visibility_cycle = np.sin(time * 0.1 + hash(self.name) % 100)
        self.visibility = max(0.1, 0.5 + 0.5 * visibility_cycle)


class Planet:
    """Planet with orbital motion"""

    def __init__(self, name: str, data: dict):
        self.name = name
        self.period = data['period']  # Orbital period in days
        self.distance = data['distance']  # Distance from Sun in AU
        self.color = data['color']
        self.size = data['size']
        self.angle = random.uniform(0, 2*np.pi)  # Starting position
        self.x = 0.0
        self.y = 0.0
        self.trail = deque(maxlen=50)  # Orbital trail

    def update_position(self, time: float):
        """Update planet position based on orbital mechanics"""
        # Angular velocity (radians per time unit)
        angular_velocity = 2 * np.pi / (self.period * 0.01)  # Scaled for visualization

        self.angle += angular_velocity

        # Elliptical orbit (simplified as circular)
        self.x = self.distance * 20 * np.cos(self.angle)  # Scaled for visualization
        self.y = self.distance * 20 * np.sin(self.angle)

        # Add to trail
        self.trail.append((self.x, self.y))


class MeteorShower:
    """Meteor shower with radiant point and meteors"""

    def __init__(self, name: str, radiant_ra: float, radiant_dec: float,
                 peak_rate: int = 50):
        self.name = name
        self.radiant_ra = radiant_ra
        self.radiant_dec = radiant_dec
        self.peak_rate = peak_rate
        self.meteors = []
        self.active = False

    def generate_meteors(self, n_meteors: int = 5):
        """Generate new meteors"""
        self.meteors = []

        for _ in range(n_meteors):
            # Meteors appear to radiate from the radiant point
            offset_ra = random.uniform(-30, 30)
            offset_dec = random.uniform(-30, 30)

            start_ra = self.radiant_ra + offset_ra
            start_dec = self.radiant_dec + offset_dec

            # Direction away from radiant
            end_ra = start_ra + random.uniform(-50, 50)
            end_dec = start_dec + random.uniform(-50, 50)

            meteor = {
                'start_ra': start_ra,
                'start_dec': start_dec,
                'end_ra': end_ra,
                'end_dec': end_dec,
                'lifetime': random.uniform(1.0, 3.0),
                'age': 0,
                'brightness': random.uniform(0.5, 1.0)
            }

            self.meteors.append(meteor)

    def update(self, dt: float):
        """Update meteor positions"""
        for meteor in self.meteors[:]:
            meteor['age'] += dt

            if meteor['age'] > meteor['lifetime']:
                self.meteors.remove(meteor)


class StellarUniverseVisualizer:
    """Main visualization system for stellar constellation universe"""

    def __init__(self, figsize: tuple[int, int] = (20, 14)):
        # Setup figure
        self.fig = plt.figure(figsize=figsize, facecolor=STELLAR_PALETTE['deep_space'])
        self.fig.suptitle('Stellar Constellation Universe - Interactive Star Map & Zodiac Guide',
                         fontsize=18, color=STELLAR_PALETTE['stellar_white'], fontweight='bold')

        # Create layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.25, wspace=0.25)

        # Main star map (large central panel)
        self.ax_sky = self.fig.add_subplot(gs[0:2, 0:3], projection='3d')

        # Zodiac wheel (top right)
        self.ax_zodiac = self.fig.add_subplot(gs[0, 3], projection='polar')

        # Planet positions (middle right)
        self.ax_planets = self.fig.add_subplot(gs[1, 3])

        # Constellation info (bottom left)
        self.ax_constellation_info = self.fig.add_subplot(gs[2, 0])

        # Star brightness chart (bottom center-left)
        self.ax_brightness = self.fig.add_subplot(gs[2, 1])

        # Celestial coordinates (bottom center-right)
        self.ax_coordinates = self.fig.add_subplot(gs[2, 2])

        # Meteor shower tracker (bottom right)
        self.ax_meteors = self.fig.add_subplot(gs[2, 3])

        # Style all axes
        self._style_axes()

        # Initialize astronomical objects
        self.stars = []
        self.constellations = []
        self.planets = []
        self.meteor_showers = []

        # Observer location (default: mid-latitude)
        self.observer_latitude = 40.0  # degrees
        self.observer_longitude = -74.0  # degrees (New York)

        # Time and animation state
        self.time = 0
        self.sidereal_time = 0
        self.selected_constellation = None
        self.show_constellation_lines = True
        self.show_zodiac_names = True

        # Initialize universe
        self._create_star_catalog()
        self._create_constellations()
        self._create_solar_system()
        self._create_meteor_showers()

    def _style_axes(self):
        """Style all axes for deep space theme"""
        # Main 3D sky view
        self.ax_sky.set_facecolor(STELLAR_PALETTE['deep_space'])
        self.ax_sky.xaxis.pane.fill = False
        self.ax_sky.yaxis.pane.fill = False
        self.ax_sky.zaxis.pane.fill = False
        self.ax_sky.grid(False)

        # 2D axes
        for ax in [self.ax_planets, self.ax_constellation_info,
                   self.ax_brightness, self.ax_coordinates, self.ax_meteors]:
            ax.set_facecolor(STELLAR_PALETTE['deep_space'])
            for spine in ax.spines.values():
                spine.set_color(STELLAR_PALETTE['stellar_white'])
                spine.set_linewidth(0.5)
            ax.tick_params(colors=STELLAR_PALETTE['stellar_white'], labelsize=8)

        # Zodiac wheel (polar)
        self.ax_zodiac.set_facecolor(STELLAR_PALETTE['deep_space'])
        self.ax_zodiac.grid(True, alpha=0.3, color=STELLAR_PALETTE['zodiac_gold'])

    def _create_star_catalog(self):
        """Create stars from catalog data"""
        for star_name, data in STAR_CATALOG.items():
            star = Star(
                name=star_name,
                ra=data['ra'],
                dec=data['dec'],
                magnitude=data['mag'],
                color=data['color'],
                constellation=data['constellation']
            )
            self.stars.append(star)

        # Add additional random stars for richness
        for _ in range(200):
            star = Star(
                name=f"Star_{random.randint(1000, 9999)}",
                ra=random.uniform(0, 360),
                dec=random.uniform(-90, 90),
                magnitude=random.uniform(2, 6),
                color=random.choice(['stellar_white', 'stellar_blue', 'stellar_yellow',
                                   'stellar_orange', 'stellar_red']),
                constellation="Background"
            )
            self.stars.append(star)

    def _create_constellations(self):
        """Create constellation objects"""
        # Group stars by constellation
        constellation_stars = defaultdict(list)

        for star in self.stars:
            constellation_stars[star.constellation].append(star)

        # Create constellation objects
        for const_name, stars in constellation_stars.items():
            if len(stars) >= 3:  # Only create constellations with enough stars
                mythology = ""
                if const_name in ZODIAC_CONSTELLATIONS:
                    mythology = ZODIAC_CONSTELLATIONS[const_name]['mythology']

                constellation = Constellation(const_name, stars, mythology)
                self.constellations.append(constellation)

    def _create_solar_system(self):
        """Create planets with orbital motion"""
        for planet_name, data in PLANET_DATA.items():
            planet = Planet(planet_name, data)
            self.planets.append(planet)

    def _create_meteor_showers(self):
        """Create meteor shower objects"""
        showers = [
            ('Perseids', 46.0, 58.0, 100),
            ('Geminids', 112.0, 32.0, 120),
            ('Leonids', 152.0, 22.0, 15),
            ('Quadrantids', 230.0, 49.0, 110)
        ]

        for name, ra, dec, rate in showers:
            shower = MeteorShower(name, ra, dec, rate)
            self.meteor_showers.append(shower)

    def update_universe(self, frame: int):
        """Update the entire stellar universe"""
        self.time = frame * 0.05
        self.sidereal_time = (self.time * 0.1) % (2 * np.pi)

        # Update constellations
        for constellation in self.constellations:
            constellation.update_visibility(self.time, self.observer_latitude)

        # Update planets
        for planet in self.planets:
            planet.update_position(self.time)

        # Update meteor showers
        for shower in self.meteor_showers:
            shower.update(0.05)

            # Randomly activate meteor showers
            if random.random() < 0.002:  # 0.2% chance per frame
                shower.active = True
                shower.generate_meteors(random.randint(3, 8))

        # Clear and redraw
        self._clear_axes()
        self._render_universe()

    def _clear_axes(self):
        """Clear all axes"""
        self.ax_sky.clear()
        self.ax_zodiac.clear()
        self.ax_planets.clear()
        self.ax_constellation_info.clear()
        self.ax_brightness.clear()
        self.ax_coordinates.clear()
        self.ax_meteors.clear()

        self._style_axes()

    def _render_universe(self):
        """Render the complete stellar universe"""
        self._render_star_map()
        self._render_zodiac_wheel()
        self._render_solar_system()
        self._render_constellation_info()
        self._render_star_brightness()
        self._render_celestial_coordinates()
        self._render_meteor_tracker()

    def _render_star_map(self):
        """Render main 3D star map"""
        self.ax_sky.set_title('Interactive Star Map - Constellations & Deep Sky Objects',
                             color=STELLAR_PALETTE['stellar_white'], fontsize=14, pad=20)

        # Render stars with realistic colors and sizes
        for star in self.stars:
            size = star.get_size()
            brightness = star.get_brightness(self.time)
            color = STELLAR_PALETTE[star.color]

            # Main star
            self.ax_sky.scatter(star.x, star.y, star.z,
                               s=size, c=color, alpha=brightness,
                               edgecolors='white', linewidth=0.2)

            # Add glow effect for bright stars
            if star.magnitude < 1.0:
                glow_size = size * 2
                self.ax_sky.scatter(star.x, star.y, star.z,
                                   s=glow_size, c=color, alpha=brightness*0.3)

            # Label very bright stars
            if star.magnitude < 0.5:
                self.ax_sky.text(star.x, star.y, star.z + 0.1, star.name,
                                color=STELLAR_PALETTE['stellar_white'], fontsize=8,
                                alpha=0.8)

        # Render constellation lines
        if self.show_constellation_lines:
            for constellation in self.constellations:
                if constellation.visibility > 0.3:
                    for star1, star2 in constellation.lines:
                        line_alpha = constellation.visibility * 0.6

                        if constellation.is_zodiac:
                            line_color = STELLAR_PALETTE['zodiac_gold']
                            line_width = 2
                        else:
                            line_color = STELLAR_PALETTE['constellation_line']
                            line_width = 1

                        self.ax_sky.plot([star1.x, star2.x],
                                        [star1.y, star2.y],
                                        [star1.z, star2.z],
                                        color=line_color, alpha=line_alpha,
                                        linewidth=line_width)

        # Render zodiac constellation labels
        if self.show_zodiac_names:
            for constellation in self.constellations:
                if constellation.is_zodiac and constellation.visibility > 0.5:
                    # Calculate constellation center
                    center_x = np.mean([star.x for star in constellation.stars])
                    center_y = np.mean([star.y for star in constellation.stars])
                    center_z = np.mean([star.z for star in constellation.stars])

                    zodiac_info = ZODIAC_CONSTELLATIONS[constellation.name]
                    label = f"{zodiac_info['symbol']} {constellation.name}"

                    self.ax_sky.text(center_x, center_y, center_z + 0.2, label,
                                    color=STELLAR_PALETTE['zodiac_gold'],
                                    fontsize=10, fontweight='bold',
                                    alpha=constellation.visibility)

        # Render Milky Way band
        milky_way_points = []
        for angle in np.linspace(0, 2*np.pi, 100):
            x = 0.8 * np.cos(angle)
            y = 0.8 * np.sin(angle)
            z = 0.1 * np.sin(angle * 4)  # Warped disk
            milky_way_points.append([x, y, z])

        milky_way_array = np.array(milky_way_points)
        self.ax_sky.plot(milky_way_array[:, 0], milky_way_array[:, 1], milky_way_array[:, 2],
                        color=STELLAR_PALETTE['milky_way'], alpha=0.3, linewidth=3)

        # Render active meteors
        for shower in self.meteor_showers:
            if shower.active:
                for meteor in shower.meteors:
                    if meteor['age'] < meteor['lifetime']:
                        # Convert RA/Dec to 3D coordinates
                        progress = meteor['age'] / meteor['lifetime']

                        ra_current = meteor['start_ra'] + progress * (meteor['end_ra'] - meteor['start_ra'])
                        dec_current = meteor['start_dec'] + progress * (meteor['end_dec'] - meteor['start_dec'])

                        ra_rad = np.radians(ra_current)
                        dec_rad = np.radians(dec_current)

                        x = np.cos(dec_rad) * np.cos(ra_rad)
                        y = np.cos(dec_rad) * np.sin(ra_rad)
                        z = np.sin(dec_rad)

                        # Fade as meteor ages
                        fade = 1 - progress
                        brightness = meteor['brightness'] * fade

                        self.ax_sky.scatter(x, y, z, s=100*brightness,
                                           c=STELLAR_PALETTE['meteor_trail'],
                                           alpha=brightness, marker='*')

        # Add nebulae (decorative)
        for _ in range(5):
            nebula_x = random.uniform(-0.5, 0.5)
            nebula_y = random.uniform(-0.5, 0.5)
            nebula_z = random.uniform(-0.3, 0.3)

            nebula_color = random.choice([STELLAR_PALETTE['nebula_pink'],
                                        STELLAR_PALETTE['nebula_blue'],
                                        STELLAR_PALETTE['nebula_green']])

            self.ax_sky.scatter(nebula_x, nebula_y, nebula_z, s=500,
                               c=nebula_color, alpha=0.2, marker='o')

        # Set 3D limits
        self.ax_sky.set_xlim(-1.2, 1.2)
        self.ax_sky.set_ylim(-1.2, 1.2)
        self.ax_sky.set_zlim(-0.8, 0.8)

        # Remove axis labels
        self.ax_sky.set_xticks([])
        self.ax_sky.set_yticks([])
        self.ax_sky.set_zticks([])

    def _render_zodiac_wheel(self):
        """Render zodiac wheel in polar coordinates"""
        self.ax_zodiac.set_title('Zodiac Wheel', color=STELLAR_PALETTE['stellar_white'],
                                 fontsize=12, pad=20)

        # Draw zodiac signs
        zodiac_names = list(ZODIAC_CONSTELLATIONS.keys())
        n_signs = len(zodiac_names)

        for i, sign_name in enumerate(zodiac_names):
            angle = i * 2 * np.pi / n_signs
            zodiac_data = ZODIAC_CONSTELLATIONS[sign_name]

            # Sign position on wheel
            radius = 0.8

            # Color by element
            element_colors = {
                'Fire': STELLAR_PALETTE['stellar_orange'],
                'Earth': STELLAR_PALETTE['stellar_yellow'],
                'Air': STELLAR_PALETTE['stellar_blue_white'],
                'Water': STELLAR_PALETTE['stellar_blue']
            }

            color = element_colors.get(zodiac_data['element'], STELLAR_PALETTE['zodiac_gold'])

            # Draw sign symbol
            self.ax_zodiac.scatter(angle, radius, s=200, c=color, alpha=0.8, marker='o')

            # Add symbol text
            self.ax_zodiac.text(angle, radius, zodiac_data['symbol'],
                               ha='center', va='center', fontsize=16,
                               color=STELLAR_PALETTE['stellar_white'], fontweight='bold')

            # Add sign name
            self.ax_zodiac.text(angle, radius - 0.15, sign_name,
                               ha='center', va='center', fontsize=8,
                               color=STELLAR_PALETTE['stellar_white'])

        # Current time indicator (simplified)
        current_angle = self.sidereal_time
        self.ax_zodiac.plot([current_angle, current_angle], [0, 1],
                           color=STELLAR_PALETTE['meteor_trail'], linewidth=3, alpha=0.8)

        self.ax_zodiac.set_ylim(0, 1)
        self.ax_zodiac.set_rticks([])
        self.ax_zodiac.set_thetagrids(range(0, 360, 30))

    def _render_solar_system(self):
        """Render solar system with planetary positions"""
        self.ax_planets.set_title('Solar System', color=STELLAR_PALETTE['stellar_white'], fontsize=12)

        # Draw Sun at center
        self.ax_planets.scatter(0, 0, s=200, c=STELLAR_PALETTE['stellar_yellow'],
                               alpha=0.9, marker='o', edgecolors='orange', linewidth=2)

        # Draw planetary orbits and planets
        for planet in self.planets:
            # Orbital path
            orbit_angles = np.linspace(0, 2*np.pi, 100)
            orbit_x = planet.distance * 20 * np.cos(orbit_angles)
            orbit_y = planet.distance * 20 * np.sin(orbit_angles)

            self.ax_planets.plot(orbit_x, orbit_y, color=STELLAR_PALETTE['stellar_white'],
                                alpha=0.3, linewidth=0.5)

            # Planet position
            color = STELLAR_PALETTE[planet.color]
            self.ax_planets.scatter(planet.x, planet.y, s=planet.size*10,
                                   c=color, alpha=0.9, edgecolors='white', linewidth=0.5)

            # Planet trail
            if len(planet.trail) > 1:
                trail_array = np.array(list(planet.trail))
                self.ax_planets.plot(trail_array[:, 0], trail_array[:, 1],
                                    color=color, alpha=0.3, linewidth=1)

            # Planet label
            self.ax_planets.text(planet.x, planet.y + planet.distance*3, planet.name,
                                ha='center', va='bottom', fontsize=8,
                                color=STELLAR_PALETTE['stellar_white'])

        self.ax_planets.set_xlim(-250, 250)
        self.ax_planets.set_ylim(-250, 250)
        self.ax_planets.set_aspect('equal')
        self.ax_planets.set_xticks([])
        self.ax_planets.set_yticks([])

    def _render_constellation_info(self):
        """Render constellation information panel"""
        self.ax_constellation_info.set_title('Constellation Guide',
                                            color=STELLAR_PALETTE['stellar_white'], fontsize=12)

        # Always show constellation info - cycle through zodiac signs faster
        zodiac_names = list(ZODIAC_CONSTELLATIONS.keys())
        current_index = int(self.time * 0.3) % len(zodiac_names)  # Faster cycling
        featured_name = zodiac_names[current_index]
        zodiac_data = ZODIAC_CONSTELLATIONS[featured_name]

        # Create colorful info display
        info_text = f"{zodiac_data['symbol']} {featured_name}\n"
        info_text += f"Element: {zodiac_data['element']}\n"
        info_text += f"Dates: {zodiac_data['dates']}\n"
        info_text += f"Mythology: {zodiac_data['mythology']}\n"
        info_text += f"Brightest: {zodiac_data['brightest_star']}"

        # Background box
        self.ax_constellation_info.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                                          facecolor=STELLAR_PALETTE['deep_space'],
                                                          edgecolor=STELLAR_PALETTE['zodiac_gold'],
                                                          linewidth=2, alpha=0.8))

        self.ax_constellation_info.text(0.05, 0.95, info_text,
                                       transform=self.ax_constellation_info.transAxes,
                                       color=STELLAR_PALETTE['zodiac_gold'],
                                       fontsize=10, verticalalignment='top',
                                       fontweight='bold')

        # Add element color indicator
        element_colors = {
            'Fire': STELLAR_PALETTE['stellar_orange'],
            'Earth': STELLAR_PALETTE['stellar_yellow'],
            'Air': STELLAR_PALETTE['stellar_blue_white'],
            'Water': STELLAR_PALETTE['stellar_blue']
        }

        element_color = element_colors.get(zodiac_data['element'], STELLAR_PALETTE['zodiac_gold'])

        # Large symbol display
        self.ax_constellation_info.text(0.8, 0.8, zodiac_data['symbol'],
                                       transform=self.ax_constellation_info.transAxes,
                                       color=element_color, fontsize=40,
                                       ha='center', va='center', alpha=0.8)

        # Add constellation visibility indicator
        visibility_text = f"Visibility: {'★' * int(5 * random.uniform(0.3, 1.0))}"
        self.ax_constellation_info.text(0.05, 0.05, visibility_text,
                                       transform=self.ax_constellation_info.transAxes,
                                       color=STELLAR_PALETTE['stellar_white'],
                                       fontsize=9)

        self.ax_constellation_info.set_xticks([])
        self.ax_constellation_info.set_yticks([])
        self.ax_constellation_info.set_xlim(0, 1)
        self.ax_constellation_info.set_ylim(0, 1)

    def _render_star_brightness(self):
        """Render star brightness distribution"""
        self.ax_brightness.set_title('Star Magnitudes',
                                    color=STELLAR_PALETTE['stellar_white'], fontsize=12)

        # Magnitude histogram
        magnitudes = [star.magnitude for star in self.stars if star.magnitude < 6]

        bins = np.arange(-2, 7, 0.5)
        hist, bin_edges = np.histogram(magnitudes, bins=bins)

        # Color bars by magnitude
        colors = []
        for mag in bin_edges[:-1]:
            if mag < 0:
                colors.append(STELLAR_PALETTE['stellar_white'])
            elif mag < 1:
                colors.append(STELLAR_PALETTE['stellar_blue_white'])
            elif mag < 2:
                colors.append(STELLAR_PALETTE['stellar_yellow'])
            elif mag < 4:
                colors.append(STELLAR_PALETTE['stellar_orange'])
            else:
                colors.append(STELLAR_PALETTE['stellar_red'])

        bars = self.ax_brightness.bar(bin_edges[:-1], hist, width=0.4,
                                     color=colors, alpha=0.8,
                                     edgecolor=STELLAR_PALETTE['stellar_white'],
                                     linewidth=0.5)

        self.ax_brightness.set_xlabel('Magnitude', color=STELLAR_PALETTE['stellar_white'], fontsize=9)
        self.ax_brightness.set_ylabel('Count', color=STELLAR_PALETTE['stellar_white'], fontsize=9)
        self.ax_brightness.set_xlim(-2, 6)

    def _render_celestial_coordinates(self):
        """Render celestial coordinate system info"""
        self.ax_coordinates.set_title('Celestial Coordinates',
                                     color=STELLAR_PALETTE['stellar_white'], fontsize=12)

        # Show coordinate grid
        ra_lines = np.arange(0, 360, 30)
        dec_lines = np.arange(-90, 91, 30)

        # RA lines (vertical)
        for ra in ra_lines:
            x = [ra/360, ra/360]
            y = [0, 1]
            self.ax_coordinates.plot(x, y, color=STELLAR_PALETTE['constellation_line'],
                                    alpha=0.3, linewidth=0.5)

        # Dec lines (horizontal)
        for dec in dec_lines:
            x = [0, 1]
            y = [(dec + 90)/180, (dec + 90)/180]
            self.ax_coordinates.plot(x, y, color=STELLAR_PALETTE['constellation_line'],
                                    alpha=0.3, linewidth=0.5)

        # Plot some bright stars
        for star in self.stars[:20]:  # Top 20 brightest
            ra_norm = star.ra / 360
            dec_norm = (star.dec + 90) / 180

            size = star.get_size() / 5
            color = STELLAR_PALETTE[star.color]

            self.ax_coordinates.scatter(ra_norm, dec_norm, s=size, c=color, alpha=0.8)

        self.ax_coordinates.set_xlabel('Right Ascension',
                                      color=STELLAR_PALETTE['stellar_white'], fontsize=9)
        self.ax_coordinates.set_ylabel('Declination',
                                      color=STELLAR_PALETTE['stellar_white'], fontsize=9)
        self.ax_coordinates.set_xlim(0, 1)
        self.ax_coordinates.set_ylim(0, 1)

    def _render_meteor_tracker(self):
        """Render meteor shower activity"""
        self.ax_meteors.set_title('Meteor Showers',
                                 color=STELLAR_PALETTE['stellar_white'], fontsize=12)

        # Always show meteor shower data with simulated activity
        shower_info = [
            ('Perseids', 'Aug 17', 100),
            ('Geminids', 'Dec 14', 120),
            ('Leonids', 'Nov 18', 15),
            ('Quadrantids', 'Jan 4', 110)
        ]

        shower_names = [info[0] for info in shower_info]
        peak_dates = [info[1] for info in shower_info]
        base_rates = [info[2] for info in shower_info]

        # Create dynamic activity levels
        activities = []
        colors = []

        for i, (name, peak, rate) in enumerate(shower_info):
            # Simulate varying activity levels
            time_factor = np.sin(self.time * 0.1 + i) * 0.5 + 0.5
            current_activity = int(rate * time_factor * 0.3)  # Scale down for visualization
            activities.append(current_activity)

            # Color based on activity level
            if current_activity > 20:
                colors.append(STELLAR_PALETTE['meteor_trail'])
            elif current_activity > 10:
                colors.append(STELLAR_PALETTE['stellar_orange'])
            else:
                colors.append(STELLAR_PALETTE['stellar_blue_white'])

        # Horizontal bar chart
        bars = self.ax_meteors.barh(range(len(shower_names)), activities,
                                   color=colors, alpha=0.8,
                                   edgecolor=STELLAR_PALETTE['stellar_white'],
                                   linewidth=0.5)

        # Add animated sparkle effects
        for i, activity in enumerate(activities):
            if activity > 5:
                # Add sparkles for active showers
                n_sparkles = min(8, activity // 3)
                for _ in range(n_sparkles):
                    sparkle_x = random.uniform(0, activity)
                    sparkle_y = i + random.uniform(-0.3, 0.3)

                    sparkle_size = random.uniform(10, 30)
                    self.ax_meteors.scatter(sparkle_x, sparkle_y, s=sparkle_size,
                                           c=STELLAR_PALETTE['meteor_trail'],
                                           alpha=random.uniform(0.5, 1.0), marker='*')

        # Labels and formatting
        self.ax_meteors.set_yticks(range(len(shower_names)))
        shower_labels = [f"{name}\n({peak})" for name, peak in zip(shower_names, peak_dates, strict=False)]
        self.ax_meteors.set_yticklabels(shower_labels, fontsize=9)
        self.ax_meteors.set_xlabel('Meteors/Hour',
                                  color=STELLAR_PALETTE['stellar_white'], fontsize=9)

        max_activity = max(activities) if activities else 50
        self.ax_meteors.set_xlim(0, max(50, max_activity + 10))

        # Add activity indicator text
        total_activity = sum(activities)
        activity_level = "High" if total_activity > 80 else "Medium" if total_activity > 40 else "Low"

        self.ax_meteors.text(0.98, 0.02, f"Overall Activity: {activity_level}",
                            transform=self.ax_meteors.transAxes,
                            color=STELLAR_PALETTE['meteor_trail'],
                            fontsize=9, ha='right', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2",
                                    facecolor=STELLAR_PALETTE['deep_space'],
                                    edgecolor=STELLAR_PALETTE['meteor_trail'],
                                    alpha=0.7))

        # Add background grid for better visibility
        self.ax_meteors.grid(True, alpha=0.2, color=STELLAR_PALETTE['stellar_white'])

    def animate(self):
        """Start the stellar universe animation"""
        def update(frame):
            try:
                self.update_universe(frame)
            except Exception as e:
                print(f"Animation error at frame {frame}: {e}")
            return []

        self.anim = animation.FuncAnimation(
            self.fig, update,
            frames=2000,
            interval=100,
            blit=False,
            repeat=True
        )

        plt.show()


def run_stellar_universe():
    """Launch the Stellar Constellation Universe"""
    print("🌟 STELLAR CONSTELLATION UNIVERSE 2025")
    print("Interactive Star Map & Zodiac Guide")
    print()
    print("✨ Features:")
    print("  • Real star coordinates and magnitudes")
    print("  • All 12 zodiac constellations with mythology")
    print("  • Interactive constellation patterns")
    print("  • Planetary orbital motion in real-time")
    print("  • Meteor shower tracking and visualization")
    print("  • Celestial coordinate system")
    print("  • Star brightness and spectral classification")
    print("  • Deep sky objects and nebulae")
    print("  • Milky Way galaxy structure")
    print()
    print("🔭 Navigate the cosmos and explore the night sky...")

    try:
        universe = StellarUniverseVisualizer()
        universe.animate()
    except Exception as e:
        print(f"❌ Error launching stellar universe: {e}")
        print("Please ensure all dependencies are installed")


if __name__ == "__main__":
    run_stellar_universe()
