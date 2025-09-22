import random
import json
import os
import time
from datetime import datetime, timedelta
import hashlib

class Player:
    def __init__(self, name):
        self.name = name
        self.level = 1
        self.xp = 0
        self.xp_to_next_level = 100
        self.current_kingdom = "Variable Valley"
        self.completed_quests = []
        self.achievements = []
        self.streak_days = 0
        self.last_played = None
        self.inventory = {"hints": 3, "skip_tokens": 1}
        self.stats = {"problems_solved": 0, "bugs_fixed": 0, "speed_records": 0}
        
    def gain_xp(self, amount):
        self.xp += amount
        if self.xp >= self.xp_to_next_level:
            self.level_up()
    
    def level_up(self):
        self.level += 1
        self.xp -= self.xp_to_next_level
        self.xp_to_next_level = int(self.xp_to_next_level * 1.2)
        self.inventory["hints"] += 1
        print(f"\n*** LEVEL UP! You are now level {self.level}! ***")
        print(f"You gained an extra hint! Total hints: {self.inventory['hints']}")
        
    def get_title(self):
        if self.level <= 10:
            return "Novice Coder"
        elif self.level <= 25:
            return "Script Apprentice"
        elif self.level <= 40:
            return "Data Alchemist"
        elif self.level <= 60:
            return "Object Sage"
        elif self.level <= 80:
            return "Library Master"
        else:
            return "Python Archmage"

class Quest:
    def __init__(self, title, description, challenge, solution, hint, xp_reward, kingdom):
        self.title = title
        self.description = description
        self.challenge = challenge
        self.solution = solution
        self.hint = hint
        self.xp_reward = xp_reward
        self.kingdom = kingdom
        self.attempts = 0
        self.max_attempts = 3

class CodeQuest:
    def __init__(self):
        self.player = None
        self.current_quest = None
        self.save_file = "codequest_save.json"
        self.kingdoms = {
            "Variable Valley": {"unlocked": True, "level_req": 1},
            "Function Forest": {"unlocked": False, "level_req": 5},
            "Loop Labyrinth": {"unlocked": False, "level_req": 10},
            "Conditional Caverns": {"unlocked": False, "level_req": 15},
            "Data Structure Desert": {"unlocked": False, "level_req": 20},
            "Object-Oriented Oasis": {"unlocked": False, "level_req": 30},
            "Module Mountains": {"unlocked": False, "level_req": 40},
            "Exception Expanse": {"unlocked": False, "level_req": 50},
            "Algorithm Archipelago": {"unlocked": False, "level_req": 60},
            "Framework Fortress": {"unlocked": False, "level_req": 70}
        }
        self.quests = self.initialize_quests()
        
    def initialize_quests(self):
        return {
            "Variable Valley": [
                Quest(
                    "The Sacred Numbers",
                    "The village crystals have lost their power! Store the magical values.",
                    "Create three variables: 'wizard_name' with your name, 'magic_power' with value 100, and 'gold_coins' with value 50. Then print all three.",
                    ["wizard_name", "magic_power", "gold_coins", "print"],
                    "Use the format: variable_name = value",
                    50,
                    "Variable Valley"
                ),
                Quest(
                    "String Enchantment",
                    "The ancient scrolls need to be combined!",
                    "Create two string variables 'first_spell' and 'second_spell', then combine them into 'ultimate_spell' and print it.",
                    ["first_spell", "second_spell", "ultimate_spell", "+"],
                    "Use the + operator to combine strings",
                    75,
                    "Variable Valley"
                ),
                Quest(
                    "Mathematical Magic",
                    "Calculate the power needed to defeat the shadow beast!",
                    "Create variables for base_power=25 and multiplier=4. Calculate total_power and print it.",
                    ["base_power", "multiplier", "total_power", "*"],
                    "Use * for multiplication",
                    100,
                    "Variable Valley"
                )
            ],
            "Function Forest": [
                Quest(
                    "The Greeting Ritual",
                    "Create a magical greeting function for fellow wizards!",
                    "Write a function called 'greet_wizard' that takes a name parameter and returns 'Greetings, [name] the Wise!'",
                    ["def", "greet_wizard", "return", "Greetings"],
                    "def function_name(parameter):",
                    150,
                    "Function Forest"
                ),
                Quest(
                    "Power Calculator",
                    "The wizards need to calculate their combined magical power!",
                    "Create a function 'calculate_power' that takes two numbers and returns their sum multiplied by 2.",
                    ["def", "calculate_power", "return", "*", "+"],
                    "Remember to multiply the sum by 2",
                    200,
                    "Function Forest"
                ),
                Quest(
                    "The Spell Checker",
                    "Validate if a spell word has the right magical length!",
                    "Write a function 'is_valid_spell' that returns True if the spell (string) has more than 5 characters, False otherwise.",
                    ["def", "is_valid_spell", "len", "return", ">"],
                    "Use len() to get string length",
                    250,
                    "Function Forest"
                )
            ],
            "Loop Labyrinth": [
                Quest(
                    "The Counting Curse",
                    "Break the curse by counting from 1 to 10!",
                    "Use a for loop to print numbers from 1 to 10.",
                    ["for", "range", "print", "1", "11"],
                    "range(1, 11) gives you 1 to 10",
                    300,
                    "Loop Labyrinth"
                ),
                Quest(
                    "Potion Ingredients",
                    "Process each ingredient in your magical list!",
                    "Create a list of 4 potion ingredients and use a for loop to print each one with 'Adding: ' prefix.",
                    ["for", "in", "print", "Adding:", "list"],
                    "for ingredient in ingredients:",
                    350,
                    "Loop Labyrinth"
                ),
                Quest(
                    "The Infinite Guardian",
                    "Defeat the guardian by finding the magic number!",
                    "Use a while loop to find the first number greater than 50 that's divisible by 7. Start from 51.",
                    ["while", "%", "==", "0", ">"],
                    "Use % operator to check divisibility",
                    400,
                    "Loop Labyrinth"
                )
            ],
            "Conditional Caverns": [
                Quest(
                    "The Gate Keeper",
                    "Only wizards with enough power can pass!",
                    "Write code that checks if power_level > 100. If true, print 'Access Granted', else 'Access Denied'.",
                    ["if", "else", "power_level", ">", "100"],
                    "if condition: ... else: ...",
                    450,
                    "Conditional Caverns"
                ),
                Quest(
                    "Spell Difficulty Classifier",
                    "Classify spells by their difficulty level!",
                    "Create a function that takes spell_level and returns 'Easy' (1-3), 'Medium' (4-6), or 'Hard' (7+).",
                    ["def", "if", "elif", "else", "<=", ">="],
                    "Use elif for multiple conditions",
                    500,
                    "Conditional Caverns"
                ),
                Quest(
                    "The Weather Oracle",
                    "Predict the best day for magic based on conditions!",
                    "Check if temperature > 20 AND humidity < 60. If both true, print 'Perfect Magic Day!', else 'Stay Inside'.",
                    ["and", "if", "else", ">", "<"],
                    "Use 'and' to combine conditions",
                    550,
                    "Conditional Caverns"
                )
            ],
            "Data Structure Desert": [
                Quest(
                    "The Inventory Master",
                    "Organize your magical items efficiently!",
                    "Create a list of 5 magical items, add 2 more items, then print the total count and the first item.",
                    ["list", "append", "len", "print", "[0]"],
                    "Use append() to add items, len() for count",
                    600,
                    "Data Structure Desert"
                ),
                Quest(
                    "The Spell Dictionary",
                    "Create a magical spell reference book!",
                    "Create a dictionary with 3 spells as keys and their effects as values. Print one spell's effect.",
                    ["dict", "{", "}", ":", "print"],
                    "dictionary = {'key': 'value'}",
                    650,
                    "Data Structure Desert"
                ),
                Quest(
                    "The Unique Artifacts",
                    "Remove duplicate artifacts from your collection!",
                    "Create a list with duplicate items, convert to set to remove duplicates, then back to list and print.",
                    ["list", "set", "duplicates", "unique"],
                    "set() removes duplicates automatically",
                    700,
                    "Data Structure Desert"
                )
            ]
        }
    
    def start_game(self):
        print("=" * 60)
        print("    WELCOME TO CODEQUEST: THE PYTHON ODYSSEY")
        print("=" * 60)
        print("\nEmbark on an epic journey to master Python programming!")
        print("Battle code challenges, level up your skills, and become a Python Archmage!")
        
        if os.path.exists(self.save_file):
            choice = input("\nFound existing save file. Load game? (y/n): ").lower()
            if choice == 'y':
                self.load_game()
            else:
                self.create_new_player()
        else:
            self.create_new_player()
        
        self.check_daily_streak()
        self.main_menu()
    
    def create_new_player(self):
        name = input("\nEnter your wizard name: ").strip()
        if not name:
            name = "Anonymous Wizard"
        self.player = Player(name)
        print(f"\nWelcome, {name}! Your journey begins in the Variable Valley...")
        self.save_game()
    
    def main_menu(self):
        while True:
            self.update_kingdom_access()
            print(f"\n{'='*50}")
            print(f"  {self.player.name} - Level {self.player.level} {self.player.get_title()}")
            print(f"  XP: {self.player.xp}/{self.player.xp_to_next_level}")
            print(f"  Current Kingdom: {self.player.current_kingdom}")
            print(f"  Streak: {self.player.streak_days} days")
            print(f"{'='*50}")
            
            print("\n1. Start Quest")
            print("2. Travel to Kingdom")
            print("3. View Stats")
            print("4. View Achievements")
            print("5. Daily Challenge")
            print("6. Inventory")
            print("7. Save & Quit")
            
            choice = input("\nChoose your action: ").strip()
            
            if choice == '1':
                self.start_quest()
            elif choice == '2':
                self.travel_menu()
            elif choice == '3':
                self.show_stats()
            elif choice == '4':
                self.show_achievements()
            elif choice == '5':
                self.daily_challenge()
            elif choice == '6':
                self.show_inventory()
            elif choice == '7':
                self.save_game()
                print("Game saved! See you next time, brave wizard!")
                break
            else:
                print("Invalid choice! Try again.")
    
    def start_quest(self):
        available_quests = self.get_available_quests()
        if not available_quests:
            print(f"\nNo more quests available in {self.player.current_kingdom}!")
            print("Try traveling to a new kingdom or complete more challenges.")
            return
        
        quest = random.choice(available_quests)
        self.current_quest = quest
        
        print(f"\n{'='*50}")
        print(f"  QUEST: {quest.title}")
        print(f"{'='*50}")
        print(f"\n{quest.description}")
        print(f"\nCHALLENGE:")
        print(f"{quest.challenge}")
        print(f"\nReward: {quest.xp_reward} XP")
        print(f"Attempts remaining: {quest.max_attempts - quest.attempts}")
        
        self.code_challenge()
    
    def code_challenge(self):
        quest = self.current_quest
        
        while quest.attempts < quest.max_attempts:
            print(f"\n{'-'*30}")
            print("Write your Python code (type 'hint' for help, 'skip' to skip):")
            print("Press Enter twice when finished.")
            
            code_lines = []
            while True:
                line = input(">>> " if not code_lines else "... ")
                if line.strip() == "":
                    if code_lines:
                        break
                    else:
                        continue
                elif line.lower() == 'hint':
                    self.use_hint()
                    continue
                elif line.lower() == 'skip':
                    if self.use_skip():
                        return
                    continue
                else:
                    code_lines.append(line)
            
            user_code = '\n'.join(code_lines)
            
            if self.evaluate_code(user_code, quest):
                self.quest_success()
                return
            else:
                quest.attempts += 1
                remaining = quest.max_attempts - quest.attempts
                if remaining > 0:
                    print(f"\nNot quite right! {remaining} attempts remaining.")
                    print("Try again or use a hint!")
                else:
                    print("\nQuest failed! Don't worry, you can try again later.")
                    print("You still gain some experience for trying!")
                    self.player.gain_xp(quest.xp_reward // 4)
                    return
    
    def evaluate_code(self, user_code, quest):
        # Simple evaluation - check if solution keywords are present
        user_code_lower = user_code.lower()
        required_keywords = quest.solution
        
        # Count how many required elements are present
        matches = 0
        for keyword in required_keywords:
            if keyword.lower() in user_code_lower:
                matches += 1
        
        # Need at least 70% of keywords to pass
        success_rate = matches / len(required_keywords)
        
        # Try to execute the code safely (basic check)
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': {
                    'print': print, 'len': len, 'range': range, 
                    'str': str, 'int': int, 'float': float,
                    'list': list, 'dict': dict, 'set': set,
                    'True': True, 'False': False
                }
            }
            
            # Execute in safe environment
            exec(user_code, safe_globals)
            syntax_valid = True
        except:
            syntax_valid = False
            print("Code has syntax errors or uses restricted functions!")
        
        return success_rate >= 0.7 and syntax_valid
    
    def quest_success(self):
        quest = self.current_quest
        print(f"\n*** QUEST COMPLETED! ***")
        print(f"Excellent work! You've mastered: {quest.title}")
        
        # Calculate bonus XP for fewer attempts
        bonus_xp = (quest.max_attempts - quest.attempts) * 25
        total_xp = quest.xp_reward + bonus_xp
        
        self.player.gain_xp(total_xp)
        self.player.completed_quests.append(f"{quest.kingdom}:{quest.title}")
        self.player.stats["problems_solved"] += 1
        
        if bonus_xp > 0:
            print(f"Bonus XP for efficiency: +{bonus_xp}")
        print(f"Total XP gained: {total_xp}")
        
        # Check for achievements
        self.check_achievements()
        self.save_game()
    
    def use_hint(self):
        if self.player.inventory["hints"] > 0:
            self.player.inventory["hints"] -= 1
            print(f"\nHINT: {self.current_quest.hint}")
            print(f"Hints remaining: {self.player.inventory['hints']}")
        else:
            print("\nNo hints available! Gain more by leveling up.")
    
    def use_skip(self):
        if self.player.inventory["skip_tokens"] > 0:
            choice = input("Use skip token? This will complete the quest but give reduced XP (y/n): ")
            if choice.lower() == 'y':
                self.player.inventory["skip_tokens"] -= 1
                self.player.gain_xp(self.current_quest.xp_reward // 2)
                self.player.completed_quests.append(f"{self.current_quest.kingdom}:{self.current_quest.title}")
                print("Quest skipped! You gained partial XP.")
                return True
        else:
            print("No skip tokens available!")
        return False
    
    def get_available_quests(self):
        kingdom_quests = self.quests.get(self.player.current_kingdom, [])
        completed_quest_names = [q.split(':')[1] for q in self.player.completed_quests 
                               if q.startswith(self.player.current_kingdom)]
        
        available = [q for q in kingdom_quests if q.title not in completed_quest_names]
        return available
    
    def travel_menu(self):
        print(f"\n{'='*40}")
        print("    KINGDOM TRAVEL")
        print(f"{'='*40}")
        
        for i, (kingdom, info) in enumerate(self.kingdoms.items(), 1):
            status = "UNLOCKED" if info["unlocked"] else f"LOCKED (Level {info['level_req']} required)"
            current = " <- CURRENT" if kingdom == self.player.current_kingdom else ""
            print(f"{i}. {kingdom} - {status}{current}")
        
        print(f"{len(self.kingdoms) + 1}. Back to Main Menu")
        
        try:
            choice = int(input("\nChoose kingdom: "))
            if choice == len(self.kingdoms) + 1:
                return
            
            kingdom_name = list(self.kingdoms.keys())[choice - 1]
            kingdom_info = self.kingdoms[kingdom_name]
            
            if kingdom_info["unlocked"]:
                self.player.current_kingdom = kingdom_name
                print(f"\nTraveling to {kingdom_name}...")
                print("New adventures await!")
                self.save_game()
            else:
                print(f"\nKingdom locked! Reach level {kingdom_info['level_req']} to unlock.")
        except (ValueError, IndexError):
            print("Invalid choice!")
    
    def update_kingdom_access(self):
        for kingdom, info in self.kingdoms.items():
            if self.player.level >= info["level_req"]:
                if not info["unlocked"]:
                    info["unlocked"] = True
                    print(f"\n*** NEW KINGDOM UNLOCKED: {kingdom}! ***")
    
    def show_stats(self):
        print(f"\n{'='*40}")
        print(f"    {self.player.name}'s STATISTICS")
        print(f"{'='*40}")
        print(f"Level: {self.player.level} ({self.player.get_title()})")
        print(f"XP: {self.player.xp}/{self.player.xp_to_next_level}")
        print(f"Current Kingdom: {self.player.current_kingdom}")
        print(f"Quests Completed: {len(self.player.completed_quests)}")
        print(f"Problems Solved: {self.player.stats['problems_solved']}")
        print(f"Bugs Fixed: {self.player.stats['bugs_fixed']}")
        print(f"Daily Streak: {self.player.streak_days} days")
        print(f"Achievements: {len(self.player.achievements)}")
        
        input("\nPress Enter to continue...")
    
    def show_achievements(self):
        print(f"\n{'='*40}")
        print("    ACHIEVEMENTS")
        print(f"{'='*40}")
        
        if not self.player.achievements:
            print("No achievements yet! Complete quests to earn them.")
        else:
            for achievement in self.player.achievements:
                print(f"* {achievement}")
        
        input("\nPress Enter to continue...")
    
    def show_inventory(self):
        print(f"\n{'='*40}")
        print("    INVENTORY")
        print(f"{'='*40}")
        print(f"Hints: {self.player.inventory['hints']}")
        print(f"Skip Tokens: {self.player.inventory['skip_tokens']}")
        
        input("\nPress Enter to continue...")
    
    def daily_challenge(self):
        print(f"\n{'='*40}")
        print("    DAILY CHALLENGE")
        print(f"{'='*40}")
        
        # Simple daily challenge - changes based on date
        today = datetime.now().strftime("%Y-%m-%d")
        challenge_seed = int(hashlib.md5(today.encode()).hexdigest()[:8], 16)
        random.seed(challenge_seed)
        
        challenges = [
            "Write a function that returns the sum of all even numbers from 1 to 20",
            "Create a list comprehension that squares all odd numbers from 1 to 10",
            "Write code that counts how many vowels are in the string 'Python Programming'",
            "Create a function that checks if a number is prime",
            "Write code that reverses a string without using built-in reverse functions"
        ]
        
        daily_quest = random.choice(challenges)
        print(f"Today's Challenge: {daily_quest}")
        print(f"Reward: 200 XP + Streak Bonus")
        print(f"Current Streak: {self.player.streak_days} days")
        
        if f"daily_{today}" in self.player.completed_quests:
            print("\nYou've already completed today's challenge!")
            print("Come back tomorrow for a new challenge!")
        else:
            choice = input("\nAttempt daily challenge? (y/n): ")
            if choice.lower() == 'y':
                print("\nWrite your solution:")
                code = input(">>> ")
                
                # Simple validation - just check if they wrote something substantial
                if len(code.strip()) > 20:
                    streak_bonus = self.player.streak_days * 10
                    total_xp = 200 + streak_bonus
                    self.player.gain_xp(total_xp)
                    self.player.completed_quests.append(f"daily_{today}")
                    print(f"\nDaily challenge completed! +{total_xp} XP")
                    if streak_bonus > 0:
                        print(f"Streak bonus: +{streak_bonus} XP")
                else:
                    print("\nTry putting more effort into your solution!")
    
    def check_daily_streak(self):
        today = datetime.now().date()
        
        if self.player.last_played:
            last_played = datetime.strptime(self.player.last_played, "%Y-%m-%d").date()
            days_diff = (today - last_played).days
            
            if days_diff == 1:
                self.player.streak_days += 1
                print(f"\nDaily streak continued! {self.player.streak_days} days")
            elif days_diff > 1:
                self.player.streak_days = 1
                print(f"\nStreak reset. Starting fresh at day 1!")
        else:
            self.player.streak_days = 1
        
        self.player.last_played = today.strftime("%Y-%m-%d")
    
    def check_achievements(self):
        new_achievements = []
        
        # First Quest achievement
        if len(self.player.completed_quests) == 1 and "First Steps" not in self.player.achievements:
            new_achievements.append("First Steps - Complete your first quest!")
        
        # Problem Solver achievements
        problems_solved = self.player.stats["problems_solved"]
        if problems_solved >= 10 and "Problem Solver" not in self.player.achievements:
            new_achievements.append("Problem Solver - Solve 10 coding challenges!")
        
        if problems_solved >= 50 and "Code Warrior" not in self.player.achievements:
            new_achievements.append("Code Warrior - Solve 50 coding challenges!")
        
        # Level achievements
        if self.player.level >= 10 and "Rising Star" not in self.player.achievements:
            new_achievements.append("Rising Star - Reach level 10!")
        
        if self.player.level >= 25 and "Python Adept" not in self.player.achievements:
            new_achievements.append("Python Adept - Reach level 25!")
        
        # Streak achievements
        if self.player.streak_days >= 7 and "Dedicated Learner" not in self.player.achievements:
            new_achievements.append("Dedicated Learner - Maintain a 7-day streak!")
        
        # Add new achievements
        for achievement in new_achievements:
            self.player.achievements.append(achievement)
            print(f"\n*** ACHIEVEMENT UNLOCKED: {achievement} ***")
    
    def save_game(self):
        save_data = {
            "name": self.player.name,
            "level": self.player.level,
            "xp": self.player.xp,
            "xp_to_next_level": self.player.xp_to_next_level,
            "current_kingdom": self.player.current_kingdom,
            "completed_quests": self.player.completed_quests,
            "achievements": self.player.achievements,
            "streak_days": self.player.streak_days,
            "last_played": self.player.last_played,
            "inventory": self.player.inventory,
            "stats": self.player.stats,
            "kingdoms": self.kingdoms
        }
        
        try:
            with open(self.save_file, 'w') as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            print(f"Error saving game: {e}")
    
    def load_game(self):
        try:
            with open(self.save_file, 'r') as f:
                save_data = json.load(f)
            
            self.player = Player(save_data["name"])
            self.player.level = save_data["level"]
            self.player.xp = save_data["xp"]
            self.player.xp_to_next_level = save_data["xp_to_next_level"]
            self.player.current_kingdom = save_data["current_kingdom"]
            self.player.completed_quests = save_data["completed_quests"]
            self.player.achievements = save_data["achievements"]
            self.player.streak_days = save_data["streak_days"]
            self.player.last_played = save_data["last_played"]
            self.player.inventory = save_data["inventory"]
            self.player.stats = save_data["stats"]
            self.kingdoms = save_data["kingdoms"]
            
            print(f"\nWelcome back, {self.player.name}!")
            
        except Exception as e:
            print(f"Error loading game: {e}")
            print("Starting new game...")
            self.create_new_player()

# Main execution
if __name__ == "__main__":
    game = CodeQuest()
    game.start_game()

