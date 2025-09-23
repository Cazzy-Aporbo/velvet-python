#!/usr/bin/env python3
"""
Space Anarchy: A Chaotic Space Adventure

Features
- Navigate a map of sectors and planets
- Black hole events with survival challenges
- Force fields requiring an antigravity package to escape
- Warp drive to jump through time and alternate universes
- Asteroid fields with dodge mini-game
- Aliens to meet, befriend, negotiate, or fight
- Inventory, fuel, ship integrity, reputation
- Random encounters, procedural descriptions, emergent chaos
- Several mini puzzles (prime test, quadratic hint, pattern puzzle)
- Self contained, no external libs

Controls: type a command when prompted.
Commands: help, status, scan, jump, travel, land, explore, use <item>,
          inventory, trade, talk, attack, dock, warp, quit
"""

import random
import math
import time
import sys

# Seed randomness for variety
random.seed()

# -------------------------
# Utility functions
# -------------------------

def randint(a, b):
    return random.randint(a, b)

def choose(seq):
    return random.choice(seq)

def now():
    return time.time()

def pause(msg=None, secs=0.5):
    if msg:
        print(msg)
    time.sleep(secs)

def format_uuid():
    return ''.join(random.choice('0123456789abcdef') for _ in range(8))

def is_prime_verbose(n):
    steps = []
    if n < 2:
        steps.append(f"{n} less than 2 not prime")
        return False, steps
    r = int(math.isqrt(n))
    steps.append(f"Check divisors up to √{n} = {r}")
    for i in range(2, r+1):
        if n % i == 0:
            steps.append(f"{i} divides {n} => composite")
            return False, steps
        else:
            steps.append(f"{i} does not divide {n}")
    steps.append(f"No divisors found => {n} prime")
    return True, steps

# Quadratic solver step explanation
def solve_quadratic_verbose(a, b, c):
    steps = []
    steps.append(f"Equation: {a}x^2 + {b}x + {c} = 0")
    if a == 0:
        steps.append("Not a quadratic when a == 0")
        return [], steps
    steps.append(f"Divide by {a} to normalize")
    half_b = b / (2*a)
    steps.append(f"Complete square with (b/2a) = {half_b}")
    disc = b*b - 4*a*c
    steps.append(f"Discriminant D = b^2 - 4ac = {disc}")
    if disc < 0:
        steps.append("Discriminant < 0 => complex roots")
        sqrt_disc = math.sqrt(-disc)
        x1 = (-b / (2*a)) + (sqrt_disc/(2*a))*1j
        x2 = (-b / (2*a)) - (sqrt_disc/(2*a))*1j
    else:
        sqrt_disc = math.sqrt(disc)
        x1 = (-b + sqrt_disc)/(2*a)
        x2 = (-b - sqrt_disc)/(2*a)
    steps.append(f"Roots: x1 = {x1}, x2 = {x2}")
    return [x1, x2], steps

# -------------------------
# Game data classes
# -------------------------

class Player:
    def __init__(self, name):
        self.name = name
        self.fuel = 100.0
        self.hull = 100.0
        self.credits = 200
        self.inventory = {}  # item: count
        self.reputation = 0   # neutral 0, positive friend, negative hostile
        self.location = None  # Sector object
        self.ship_id = format_uuid()
        self.warp_charges = 1
        self.antigrav = False
        self.time_paradox = 0

    def add_item(self, item, count=1):
        self.inventory[item] = self.inventory.get(item, 0) + count
        if item.lower() == 'antigravity package':
            self.antigrav = True

    def use_item(self, item):
        if self.inventory.get(item, 0) <= 0:
            return False
        self.inventory[item] -= 1
        if self.inventory[item] == 0:
            del self.inventory[item]
        if item.lower() == 'antigravity package':
            self.antigrav = False  # consumed
        return True

    def status_text(self):
        inv = ', '.join(f"{k} x{v}" for k,v in self.inventory.items()) or "empty"
        return (f"Pilot {self.name} Ship {self.ship_id}\n"
                f"Location {self.location.name if self.location else 'Unknown'}\n"
                f"Fuel {self.fuel:.1f} Hull {self.hull:.1f} Credits {self.credits}\n"
                f"Warp charges {self.warp_charges} Antigravity {'yes' if self.antigrav else 'no'}\n"
                f"Inventory {inv}\n"
                f"Reputation {self.reputation} Time paradox {self.time_paradox}")

class Planet:
    def __init__(self, name, flavor, danger=0, resources=None, alien=None):
        self.name = name
        self.flavor = flavor
        self.danger = danger
        self.resources = resources or []
        self.alien = alien  # Alien species or None

class AlienSpecies:
    def __init__(self, name, disposition=0):
        self.name = name
        self.disposition = disposition  # positive friendly, negative hostile

    def greet(self):
        if self.disposition > 1:
            return f"{self.name} greets you warmly and offers aid."
        elif self.disposition < -1:
            return f"{self.name} snarls and prepares to defend its territory."
        else:
            return f"{self.name} observes you with cautious curiosity."

class Sector:
    def __init__(self, x, y, name=None, dimensional=False):
        self.x = x
        self.y = y
        self.name = name or f"Sector {x},{y}"
        self.planets = []
        self.dimensional = dimensional  # multidimensional sector changes rules
        self.black_hole = False

    def describe(self):
        lines = [f"Region {self.name} at coordinates ({self.x},{self.y})"]
        if self.dimensional:
            lines.append("Reality seems thin here. Physics behaves oddly.")
        if self.black_hole:
            lines.append("A black hole lurks here. Stay cautious.")
        if not self.planets:
            lines.append("Empty space with a scattering of asteroids.")
        else:
            lines.append("Nearby worlds:")
            for p in self.planets:
                lines.append(f"  {p.name}: {p.flavor}")
        return '\n'.join(lines)

# -------------------------
# Game world generation
# -------------------------
WORLD = {}

def gen_sector(x, y):
    key = (x, y)
    if key in WORLD:
        return WORLD[key]
    name = f"Quadrant {x}:{y}"
    dimensional = random.random() < 0.12
    sector = Sector(x, y, name=name, dimensional=dimensional)
    # chance of black hole
    if random.random() < 0.06:
        sector.black_hole = True
    # generate planets
    for i in range(randint(0, 3)):
        pname = f"{choose(['Aurelia','Kest','Nox','Zhara','Ith','Vel'])}-{format_uuid()[:4]}"
        flavor = choose([
            "icy ocean world",
            "lava scorched plateaus",
            "lush green canopies with floating islands",
            "metallic plains with strange structures",
            "gas giant with shimmering rings"
        ])
        danger = randint(0, 5)
        resources = []
        if random.random() < 0.5:
            resources.append(choose(['ice','rare_minerals','antimatter_ice','biomass']))
        alien = None
        if random.random() < 0.4:
            alien_name = choose(['Trekli','Vorin','Alyth','Xen','Orr'])
            disposition = randint(-3,3)
            alien = AlienSpecies(alien_name, disposition)
        p = Planet(pname, flavor, danger, resources, alien)
        sector.planets.append(p)
    WORLD[key] = sector
    return sector

# Pre generate nearby sectors for starting area
for X in range(-2, 3):
    for Y in range(-2, 3):
        gen_sector(X, Y)

# -------------------------
# Mini games and challenges
# -------------------------

def asteroid_field_mini(player):
    print("You fly into an asteroid field. You must dodge rocks.")
    hits = 0
    dodged = 0
    for i in range(6):
        ev = random.random()
        prompt = f"Roll dodge attempt {i+1}. Type 'left' or 'right' or 'center': "
        choice = input(prompt).strip().lower()
        if choice not in ('left','right','center'):
            print("Invalid move. You get hit.")
            hits += 1
            continue
        move_success = random.random() < 0.6
        if move_success:
            print("You dodge a chunk of rock.")
            dodged += 1
        else:
            print("A shard smashes the hull.")
            hits += 1
    damage = hits * randint(3,8)
    player.hull = max(0, player.hull - damage)
    print(f"Asteroid run complete. Hull damaged by {damage}. Hull now {player.hull:.1f}.")
    if player.hull <= 0:
        print("Ship destroyed by asteroid field.")
        return False
    return True

def black_hole_challenge(player):
    print("Warning: your ship is being pulled into a black hole.")
    if player.antigrav:
        print("Your antigravity package hums and you resist the pull if you act quickly.")
    # simple puzzle: solve a prime check or face damage
    n = randint(50, 200)
    print(f"To power the stabilizer, compute whether {n} is prime. Type 'prime' or 'composite'.")
    ans = input("> ").strip().lower()
    isprime, steps = is_prime_verbose(n)
    correct = ('prime' if isprime else 'composite')
    if ans == correct:
        print("Stabilizers engaged. You escape the black hole's immediate event horizon.")
        # reward: small fuel and reputation
        player.fuel = min(200, player.fuel + 15)
        player.reputation += 1
        print("You gain fuel and reputation.")
        return True
    else:
        print("Stabilizers fail. The black hole strains the hull.")
        damage = randint(20, 60)
        player.hull = max(0, player.hull - damage)
        print(f"Hull lost {damage} integrity. Hull now {player.hull:.1f}.")
        if player.hull <= 0:
            print("The ship collapses into the singularity. You die.")
            return False
        return True

def force_field_event(player):
    print("You are trapped in a shimmering force field with strange energy.")
    if player.antigrav:
        print("Your antigravity package can modulate the field to escape.")
        print("Use antigravity package now? yes/no")
        c = input("> ").strip().lower()
        if c == 'yes' and player.use_item('antigravity package'):
            print("Antigravity engages and you phase through.")
            return True
        else:
            print("You did not or could not use antigravity. Field holds.")
    # otherwise mini puzzle to escape: quadratic
    print("Solve a quick quadratic to open a portal. Provide sum of roots of x^2 - 3x + 2 = 0")
    ans = input("> ").strip()
    try:
        s = float(ans)
        # sum of roots = -b/a = 3
        if abs(s - 3.0) < 1e-6:
            print("Portal opens and you slip through.")
            player.reputation += 1
            return True
        else:
            print("Wrong. The field drains energy.")
            player.fuel = max(0, player.fuel - 20)
            return False
    except:
        print("Invalid answer. The field drains energy.")
        player.fuel = max(0, player.fuel - 20)
        return False

def alien_interaction(player, alien):
    print(alien.greet())
    # options depend on disposition
    if alien.disposition >= 2:
        print("They offer you a trade or knowledge. Options: trade, learn, leave")
    elif alien.disposition <= -2:
        print("They look hostile. Options: fight, flee, attempt_parley")
    else:
        print("They are cautious. Options: talk, trade, leave")
    choice = input("> ").strip().lower()
    if choice == 'trade':
        price = randint(10,80)
        if player.credits >= price:
            player.credits -= price
            item = choose(['antigravity package','rare_minerals','mysterious_module'])
            player.add_item(item)
            print(f"You trade {price} credits for {item}.")
            player.reputation += 1
        else:
            print("Not enough credits. They are annoyed.")
            player.reputation -= 1
    elif choice == 'learn' and alien.disposition >= 2:
        print("They share an insight about time dilation. You gain a warp charge.")
        player.warp_charges += 1
        player.reputation += 2
    elif choice == 'talk':
        # small chat quiz
        print("They ask you a pattern test: what comes next in 2 3 5 7 11 ?")
        ans = input("> ").strip()
        if ans.strip() == '13':
            print("They nod approvingly and give you a small gift.")
            player.add_item('rare_minerals')
            player.reputation += 1
        else:
            print("They are unimpressed.")
            player.reputation -= 1
    elif choice == 'fight':
        print("Combat initiated.")
        # simple fight computed by random and reputation
        outcome = random.random() + (player.reputation * 0.05)
        if outcome > 0.8:
            print("You win the skirmish but take some hull damage.")
            player.hull -= randint(5,30)
            player.credits += randint(10,80)
            player.reputation -= 1
        else:
            print("You lose. You retreat with damage.")
            player.hull -= randint(15,50)
            player.credits = max(0, player.credits - randint(10,50))
            player.reputation -= 2
    elif choice == 'attempt_parley':
        roll = random.random() + (player.reputation * 0.05)
        if roll > 0.6:
            print("Your parley works. They calm down.")
            player.reputation += 1
        else:
            print("They do not accept your parley.")
            player.reputation -= 1
    elif choice == 'flee' or choice == 'leave':
        print("You leave the encounter.")
    else:
        print("They do not understand your action and move away.")
        player.reputation -= 1

# -------------------------
# Core game engine
# -------------------------

class Game:
    def __init__(self, player):
        self.player = player
        # place player at sector 0,0
        self.player.location = gen_sector(0, 0)
        self.turn = 0

    def help(self):
        print("Available commands")
        print("help            Show this help")
        print("status          Show player and ship status")
        print("scan            Describe current sector and nearby worlds")
        print("jump x y        Jump to sector coordinates x y consuming fuel")
        print("travel planet   Travel to planet in current sector by name or index")
        print("land            Land on the planet you selected")
        print("explore         Explore the planet when landed")
        print("use <item>      Use an item from inventory")
        print("inventory       Show inventory")
        print("talk            Talk to alien if present")
        print("attack          Attack hostile alien")
        print("warp            Use warp drive to jump through time/universe")
        print("quit            Exit the game")

    def status(self):
        print(self.player.status_text())

    def scan(self):
        print(self.player.location.describe())
        # local anomalies
        if self.player.location.dimensional:
            print("Sensors are scrambled with strange interference.")
        if self.player.location.black_hole:
            print("Distorted gravity readings. A black hole is nearby.")

    def jump(self, x, y):
        dist = math.hypot(self.player.location.x - x, self.player.location.y - y)
        fuel_cost = max(5.0, dist * 6.0)
        if self.player.fuel < fuel_cost:
            print("Not enough fuel to jump that far")
            return
        self.player.fuel -= fuel_cost
        base_sector = gen_sector(x, y)
        self.player.location = base_sector
        print(f"Jumped to sector {base_sector.name} at ({x},{y}). Fuel -{fuel_cost:.1f}")
        # random immediate encounter
        self.turn += 1
        if base_sector.black_hole and random.random() < 0.5:
            ok = black_hole_challenge(self.player)
            if not ok:
                print("Game over")
                sys.exit(0)
        # random force field
        if random.random() < 0.08:
            ok = force_field_event(self.player)
            if not ok:
                print("You escaped but exhausted. Continue.")
        # dimensional warp side effects
        if base_sector.dimensional and random.random() < 0.5:
            self.player.time_paradox += 1
            print("Reality trembles. You sense time shifted around you.")

    def travel(self, target):
        # find planet by name or index
        sector = self.player.location
        if not sector.planets:
            print("No planets to travel to here")
            return
        found = None
        try:
            idx = int(target) - 1
            if 0 <= idx < len(sector.planets):
                found = sector.planets[idx]
        except:
            for p in sector.planets:
                if p.name.lower().startswith(target.lower()):
                    found = p
                    break
        if not found:
            print("No such planet in this sector")
            return
        print(f"Locking course to planet {found.name}. Approaching...")
        # landing cost: small fuel and potential asteroid event
        cost = 5 + found.danger * 2
        if self.player.fuel < cost:
            print("Not enough fuel to descend")
            return
        self.player.fuel -= cost
        print(f"Descended to {found.name}. Fuel -{cost:.1f}")
        # possible asteroid or alien encounter
        if found.danger >= 3 and random.random() < 0.6:
            ok = asteroid_field_mini(self.player)
            if not ok:
                print("You died in the asteroid field")
                sys.exit(0)
        # set landed planet in temp attribute
        self.player.landed_planet = found
        print(f"You are now above {found.name}. Use 'land' to touch down or 'explore' to remote-scan")

    def land(self):
        planet = getattr(self.player, 'landed_planet', None)
        if not planet:
            print("You are not targeting any planet")
            return
        print(f"You land on {planet.name} with environment {planet.flavor}")
        # gain resources, possible alien
        if planet.resources:
            item = planet.resources[0]
            print(f"You find {item} and salvage it")
            self.player.add_item(item)
            self.player.credits += randint(5, 40)
        if planet.alien:
            print(f"You encounter {planet.alien.name}")
            alien_interaction(self.player, planet.alien)
        # random chance to find antigrav package
        if random.random() < 0.08:
            print("You discover an antigravity package in an old crate")
            self.player.add_item('antigravity package')
        # landing consumes time, may wear hull
        self.player.hull = max(0, self.player.hull - random.uniform(0, 3))
        # clear landed_planet after landing
        delattr(self.player, 'landed_planet')

    def explore(self):
        # remote exploration without landing
        planet = getattr(self.player, 'landed_planet', None)
        if not planet:
            print("No planet targeted to explore remotely")
            return
        print(f"Remote scanning {planet.name}")
        print("Scan results")
        print(f"Environment: {planet.flavor}")
        print(f"Known hazards level {planet.danger}")
        print(f"Potential resources: {', '.join(planet.resources) if planet.resources else 'none'}")
        if planet.alien:
            print(f"Life detected: {planet.alien.name} (disposition {planet.alien.disposition})")
        # small fuel cost
        self.player.fuel = max(0, self.player.fuel - 2.5)

    def use(self, item):
        item = item.strip()
        if not item:
            print("Use what?")
            return
        if item not in self.player.inventory:
            print("You do not have that item")
            return
        if item == 'antigravity package':
            print("Antigravity toggled temporarily to resist gravity anomalies.")
            # using here acts like temporary protection for next hazard
            # do not remove until used in event
            print("Antigravity will be consumed next time it is needed.")
        else:
            print(f"You use the {item}. It has an effect.")
        # real consumption handled when needed

    def inventory(self):
        print("Inventory")
        if not self.player.inventory:
            print("empty")
            return
        for k,v in self.player.inventory.items():
            print(f"{k} x{v}")

    def talk(self):
        # talk to alien on planet if present
        sector = self.player.location
        # find nearest planet with alien
        alien_planet = None
        for p in sector.planets:
            if p.alien:
                alien_planet = p
                break
        if not alien_planet:
            print("No aliens detected here")
            return
        alien_interaction(self.player, alien_planet.alien)

    def attack(self):
        sector = self.player.location
        # find hostile alien
        hostile = None
        for p in sector.planets:
            if p.alien and p.alien.disposition < -1:
                hostile = p.alien
                break
        if not hostile:
            print("No hostile targets in this sector")
            return
        print("You attack the hostile alien")
        alien_interaction(self.player, hostile)

    def warp(self):
        if self.player.warp_charges <= 0:
            print("No warp charges left")
            return
        print("Engaging warp drive. Choose a time jump magnitude: small, medium, large")
        choice = input("> ").strip().lower()
        if choice == 'small':
            chance = 0.9
            paradox = 0
        elif choice == 'medium':
            chance = 0.7
            paradox = 1
        else:
            chance = 0.4
            paradox = 2
        self.player.warp_charges -= 1
        if random.random() < chance:
            # warp success produce new sector with dimensional rules
            x = randint(-10, 10)
            y = randint(-10, 10)
            sector = gen_sector(x, y)
            sector.dimensional = random.random() < 0.5
            self.player.location = sector
            self.player.time_paradox += paradox
            print(f"Warp succeed. Arrived at {sector.name}. Time paradox now {self.player.time_paradox}")
        else:
            print("Warp unstable. You are scattered across nearby micro-sectors and damaged")
            self.player.hull -= randint(5, 30)
            # random small displacement
            dx = randint(-2, 2)
            dy = randint(-2, 2)
            sector = gen_sector(self.player.location.x + dx, self.player.location.y + dy)
            self.player.location = sector
            print(f"You end up at {sector.name} slightly shaken.")

    def run_command(self, raw):
        cmd = raw.strip().split()
        if not cmd:
            return
        c = cmd[0].lower()
        args = cmd[1:]
        if c == 'help':
            self.help()
        elif c == 'status':
            self.status()
        elif c == 'scan':
            self.scan()
        elif c == 'jump':
            if len(args) < 2:
                print("Usage: jump x y")
                return
            try:
                x = int(args[0]); y = int(args[1])
            except:
                print("Coordinates must be integers")
                return
            self.jump(x, y)
        elif c == 'travel':
            if not args:
                print("Usage: travel <planet name or index>")
                return
            self.travel(' '.join(args))
        elif c == 'land':
            self.land()
        elif c == 'explore':
            self.explore()
        elif c == 'use':
            self.use(' '.join(args))
        elif c == 'inventory':
            self.inventory()
        elif c == 'talk':
            self.talk()
        elif c == 'attack':
            self.attack()
        elif c == 'warp':
            self.warp()
        elif c == 'scanmail' or c == 'email':
            # attempt email fetch but do not require credentials here
            print("Email retrieval not configured in this session. Use the dashboard program to link credentials.")
        elif c == 'quit':
            print("Ending your chaotic voyage. Goodbye.")
            sys.exit(0)
        else:
            print("Unknown command. Type help for a list of commands.")

# -------------------------
# Entrypoint
# -------------------------

def main():
    print("Space Anarchy: a chaotic space adventure")
    name = input("Enter your pilot name: ").strip() or "Pilot"
    player = Player(name)
    # starter items
    player.add_item('repair_kit', 2)
    player.add_item('food_rations', 3)
    # 20% chance to find an antigravity package at start
    if random.random() < 0.2:
        player.add_item('antigravity package', 1)
    game = Game(player)
    print("Welcome aboard. Type help for commands. Prepare for unpredictability.")
    # main loop
    while True:
        try:
            cmd = input("\n> ")
            game.run_command(cmd)
            # small passive regen or decay
            if random.random() < 0.05:
                # cosmic event small effect
                delta = random.uniform(-2,2)
                player.fuel = max(0, min(300, player.fuel + delta))
            # check end conditions
            if player.fuel <= 0:
                print("You have run out of fuel. Stranded in space.")
                break
            if player.hull <= 0:
                print("Your ship is destroyed. Game over.")
                break
            # periodic tips or mini events
            if random.random() < 0.03:
                event = choose(['cargo_run','salvage','mysterious_signal'])
                if event == 'cargo_run':
                    print("Tip: a nearby cargo beacon promises credits for a quick salvage. Scan sectors to find it.")
                elif event == 'salvage':
                    print("A drifting pod passes. You can try to chase it with jump and travel.")
                else:
                    print("A mysterious signal pings your radios. Try scanning your sector.")
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

if __name__ == "__main__":
    main()