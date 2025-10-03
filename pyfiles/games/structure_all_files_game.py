"""
PYTHON FILE STRUCTURE TYPING GAME
Dynamic Structure Learning and Speed Typing Challenge

This game allows users to:
- Create their own Python file structures to practice
- Challenge themselves with typing speed and accuracy
- Build muscle memory for professional documentation
- Compete against their own records
- Learn various coding patterns through repetition

Author: Cazzzy 
Version: 2.0.0
Python Requirements: 3.8+
Dependencies: Standard library only
"""

import time
import random
import os
import sys
import textwrap
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import json
import hashlib
import pickle
from pathlib import Path
from collections import deque, defaultdict
import re

class StructureTypingGame:
    """Dynamic typing game for Python file structures"""
    
    def __init__(self):
        self.user_structures = {}
        self.current_structure = None
        self.game_mode = None
        self.score = 0
        self.level = 1
        self.xp = 0
        self.combo_multiplier = 1
        self.power_ups = {
            'time_freeze': 0,
            'hint_peek': 0,
            'error_shield': 0,
            'double_points': 0
        }
        
        # Game statistics
        self.stats = {
            'total_words_typed': 0,
            'total_structures_created': 0,
            'fastest_wpm': 0,
            'longest_combo': 0,
            'total_playtime': 0,
            'favorite_structure': None,
            'achievement_points': 0
        }
        
        # Achievements system
        self.achievements = {
            'first_structure': False,
            'speed_demon': False,  # 100+ WPM
            'perfectionist': False,  # 100% accuracy 5 times
            'marathon_runner': False,  # Play for 1 hour
            'structure_master': False,  # Create 10 structures
            'combo_king': False,  # 10x combo
            'night_owl': False,  # Play after midnight
            'early_bird': False,  # Play before 6 AM
            'variety_pack': False,  # Create 5 different types
            'zen_master': False  # Practice mode for 30 minutes
        }
        
        # Challenge modes
        self.challenges = {
            'LIGHTNING_ROUND': 'Type as many lines as possible in 60 seconds',
            'MEMORY_MATRIX': 'Study for 30 seconds, then type from memory',
            'CASCADE_MODE': 'Lines disappear one by one - type before they vanish',
            'MIRROR_MODE': 'Type the structure backwards',
            'BLIND_MODE': 'Type without seeing what you typed until the end',
            'RHYTHM_MODE': 'Type to maintain a steady rhythm for bonus points'
        }
        
        self.save_file = Path.home() / '.structure_typing_game.save'
        self.load_game_data()
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def typewriter_effect(self, text: str, delay: float = 0.03):
        """Display text with typewriter effect"""
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()
    
    def generate_ascii_banner(self, text: str) -> str:
        """Generate ASCII art style banner"""
        banner_chars = {
            'top': '╔' + '═' * (len(text) + 2) + '╗',
            'middle': '║ ' + text + ' ║',
            'bottom': '╚' + '═' * (len(text) + 2) + '╝'
        }
        return f"{banner_chars['top']}\n{banner_chars['middle']}\n{banner_chars['bottom']}"
    
    def display_welcome(self):
        """Display animated welcome screen"""
        self.clear_screen()
        
        title = "STRUCTURE TYPING ARENA"
        subtitle = "Master Your Python File Architecture"
        
        print("=" * 70)
        print()
        print(self.generate_ascii_banner(title).center(70))
        print()
        print(subtitle.center(70))
        print("=" * 70)
        
        print("\n>>> INITIALIZING GAME ENVIRONMENT...")
        time.sleep(1)
        print(">>> LOADING USER PROFILES...")
        time.sleep(0.5)
        print(">>> CALIBRATING TYPING SENSORS...")
        time.sleep(0.5)
        print(">>> READY!")
        
        print(f"\nWELCOME BACK, CODER!")
        print(f"Level: {self.level} | XP: {self.xp} | Achievement Points: {self.stats['achievement_points']}")
        
        print("\nPress ENTER to dive in...")
        input()
    
    def create_custom_structure(self):
        """Let users create their own Python file structure"""
        self.clear_screen()
        print(self.generate_ascii_banner("STRUCTURE CREATOR"))
        print("\nCraft your own Python file structure to practice!")
        
        structure_name = input("\nName your structure: ").strip().upper()
        if not structure_name:
            structure_name = f"CUSTOM_STRUCTURE_{len(self.user_structures) + 1}"
        
        print("\nChoose structure type:")
        print("1. MODULE - Standard Python module")
        print("2. CLASS - Object-oriented structure")
        print("3. API - Web service structure")
        print("4. SCRIPT - Executable script")
        print("5. PACKAGE - Package with __init__")
        print("6. TEST - Unit test structure")
        print("7. CONFIG - Configuration file")
        print("8. FREESTYLE - Anything goes")
        
        structure_type = input("\nType (1-8): ").strip()
        
        print("\n" + "="*70)
        print("ENTER YOUR STRUCTURE")
        print("="*70)
        print("Type or paste your Python structure below.")
        print("Include docstrings with CAPITALIZED TITLES as you like!")
        print("Type '###END###' on a new line when done.")
        print("-"*70)
        
        lines = []
        line_number = 1
        
        while True:
            try:
                line = input(f"{line_number:3} | ")
                if line.strip() == '###END###':
                    break
                lines.append(line)
                line_number += 1
                
                # Provide live feedback
                if line.strip().startswith('"""') and len(lines) > 1:
                    if lines[1].strip() and lines[1].isupper():
                        print("     [NICE! Capitalized title detected]")
                elif 'Author:' in line:
                    print("     [Author field added]")
                elif 'Version:' in line:
                    print("     [Version field added]")
                elif line.strip().startswith('class '):
                    print("     [Class definition detected]")
                elif line.strip().startswith('def '):
                    print("     [Function definition detected]")
                    
            except KeyboardInterrupt:
                print("\n\n[Structure creation cancelled]")
                return
        
        if not lines:
            print("\nNo structure entered!")
            input("Press ENTER to continue...")
            return
        
        structure_content = '\n'.join(lines)
        
        # Calculate structure complexity
        complexity_score = self._calculate_complexity(structure_content)
        
        # Store the structure
        self.user_structures[structure_name] = {
            'content': structure_content,
            'type': structure_type,
            'created': datetime.now().isoformat(),
            'complexity': complexity_score,
            'practice_count': 0,
            'best_accuracy': 0,
            'best_wpm': 0
        }
        
        self.stats['total_structures_created'] += 1
        self.xp += 50
        
        print("\n" + "="*70)
        print(f"STRUCTURE CREATED: {structure_name}")
        print(f"Complexity Level: {complexity_score}/10")
        print(f"XP Earned: +50")
        
        # Check achievements
        if not self.achievements['first_structure'] and self.stats['total_structures_created'] == 1:
            self.unlock_achievement('first_structure', 'Created your first structure!')
        
        if not self.achievements['structure_master'] and self.stats['total_structures_created'] >= 10:
            self.unlock_achievement('structure_master', 'Structure Master - Created 10 structures!')
        
        print("\nWould you like to practice it now? (y/n): ", end='')
        if input().strip().lower() == 'y':
            self.practice_structure(structure_name)
        else:
            input("\nPress ENTER to continue...")
    
    def _calculate_complexity(self, content: str) -> int:
        """Calculate complexity score of a structure"""
        score = 1
        
        # Check various elements
        if '"""' in content:
            score += 1
        if 'class ' in content:
            score += 2
        if 'def ' in content:
            score += 1
        if 'import ' in content:
            score += 1
        if '@' in content:  # Decorators
            score += 2
        if 'async ' in content:
            score += 2
        if '__init__' in content:
            score += 1
        
        lines = content.split('\n')
        if len(lines) > 50:
            score += 2
        elif len(lines) > 20:
            score += 1
        
        return min(10, score)
    
    def practice_structure(self, structure_name: str = None):
        """Practice typing a specific structure"""
        if not structure_name and not self.user_structures:
            print("\nNo structures available! Create one first.")
            input("Press ENTER to continue...")
            return
        
        if not structure_name:
            # Let user choose
            self.clear_screen()
            print(self.generate_ascii_banner("SELECT STRUCTURE"))
            print("\nAvailable structures:")
            
            for idx, (name, data) in enumerate(self.user_structures.items(), 1):
                print(f"{idx}. {name} (Complexity: {data['complexity']}/10, "
                      f"Practiced: {data['practice_count']} times)")
            
            choice = input("\nEnter number or name: ").strip()
            
            try:
                idx = int(choice) - 1
                structure_name = list(self.user_structures.keys())[idx]
            except:
                structure_name = choice.upper()
        
        if structure_name not in self.user_structures:
            print(f"\nStructure '{structure_name}' not found!")
            input("Press ENTER to continue...")
            return
        
        structure = self.user_structures[structure_name]
        self.current_structure = structure_name
        
        # Choose game mode
        mode = self.select_game_mode()
        
        if mode == 'STANDARD':
            self.standard_typing_challenge(structure_name)
        elif mode == 'LIGHTNING_ROUND':
            self.lightning_round(structure_name)
        elif mode == 'MEMORY_MATRIX':
            self.memory_matrix(structure_name)
        elif mode == 'CASCADE_MODE':
            self.cascade_mode(structure_name)
        elif mode == 'MIRROR_MODE':
            self.mirror_mode(structure_name)
        elif mode == 'BLIND_MODE':
            self.blind_mode(structure_name)
        elif mode == 'RHYTHM_MODE':
            self.rhythm_mode(structure_name)
    
    def select_game_mode(self) -> str:
        """Select game mode for practice"""
        self.clear_screen()
        print(self.generate_ascii_banner("GAME MODE SELECTION"))
        
        print("\n1. STANDARD - Classic typing practice")
        print("2. LIGHTNING ROUND - Speed challenge")
        print("3. MEMORY MATRIX - Memorization test")
        print("4. CASCADE MODE - Disappearing lines")
        print("5. MIRROR MODE - Reverse typing")
        print("6. BLIND MODE - Type without seeing")
        print("7. RHYTHM MODE - Maintain typing rhythm")
        
        choice = input("\nSelect mode (1-7): ").strip()
        
        modes = {
            '1': 'STANDARD',
            '2': 'LIGHTNING_ROUND',
            '3': 'MEMORY_MATRIX',
            '4': 'CASCADE_MODE',
            '5': 'MIRROR_MODE',
            '6': 'BLIND_MODE',
            '7': 'RHYTHM_MODE'
        }
        
        return modes.get(choice, 'STANDARD')
    
    def standard_typing_challenge(self, structure_name: str):
        """Standard typing practice mode"""
        structure = self.user_structures[structure_name]
        content = structure['content']
        
        # Show structure
        self.clear_screen()
        print(self.generate_ascii_banner(f"STUDY: {structure_name}"))
        print("\nMemorize this structure:")
        print("-"*70)
        print(content)
        print("-"*70)
        
        study_time = max(10, min(60, len(content) // 50))
        print(f"\nYou have {study_time} seconds to study...")
        
        for remaining in range(study_time, 0, -1):
            print(f"\rTime remaining: {remaining:2d} seconds", end='')
            time.sleep(1)
        
        # Typing phase
        self.clear_screen()
        print(self.generate_ascii_banner(f"TYPE: {structure_name}"))
        print("\nType the structure from memory!")
        print("Type 'DONE' when finished")
        print("-"*70)
        
        lines = []
        start_time = time.time()
        char_count = 0
        
        while True:
            line = input()
            if line.strip() == 'DONE':
                break
            lines.append(line)
            char_count += len(line)
        
        elapsed_time = time.time() - start_time
        user_input = '\n'.join(lines)
        
        # Calculate metrics
        accuracy = self.calculate_accuracy(content, user_input)
        wpm = (char_count / 5) / (elapsed_time / 60) if elapsed_time > 0 else 0
        
        # Update statistics
        structure['practice_count'] += 1
        if accuracy > structure['best_accuracy']:
            structure['best_accuracy'] = accuracy
        if wpm > structure['best_wpm']:
            structure['best_wpm'] = wpm
        
        self.display_results(structure_name, accuracy, wpm, elapsed_time, user_input, content)
    
    def lightning_round(self, structure_name: str):
        """Speed typing challenge - type as much as possible in limited time"""
        structure = self.user_structures[structure_name]
        content_lines = structure['content'].split('\n')
        
        self.clear_screen()
        print(self.generate_ascii_banner("LIGHTNING ROUND"))
        print(f"\nStructure: {structure_name}")
        print("Type as many lines as you can in 60 seconds!")
        print("Each correct line earns points!")
        print("\nPress ENTER to start...")
        input()
        
        self.clear_screen()
        score = 0
        lines_completed = 0
        current_line_idx = 0
        start_time = time.time()
        time_limit = 60
        
        while time.time() - start_time < time_limit and current_line_idx < len(content_lines):
            remaining = time_limit - (time.time() - start_time)
            
            print(f"\rTime: {remaining:.1f}s | Score: {score} | Line {current_line_idx + 1}/{len(content_lines)}")
            print("-"*70)
            print(f"Type this line:\n>>> {content_lines[current_line_idx]}")
            print("-"*70)
            
            user_line = input(">>> ")
            
            if user_line == content_lines[current_line_idx]:
                points = 100 * self.combo_multiplier
                score += points
                self.combo_multiplier = min(5, self.combo_multiplier + 0.5)
                lines_completed += 1
                print(f"CORRECT! +{points} points (Combo: x{self.combo_multiplier:.1f})")
            else:
                self.combo_multiplier = 1
                print("MISS! Combo reset")
            
            current_line_idx += 1
            time.sleep(0.5)
            self.clear_screen()
        
        print("\n" + "="*70)
        print("LIGHTNING ROUND COMPLETE!")
        print("="*70)
        print(f"Final Score: {score}")
        print(f"Lines Completed: {lines_completed}/{len(content_lines)}")
        print(f"Accuracy: {(lines_completed/len(content_lines)*100):.1f}%")
        
        self.xp += score // 10
        print(f"XP Earned: +{score // 10}")
        
        input("\nPress ENTER to continue...")
    
    def memory_matrix(self, structure_name: str):
        """Memorization challenge - limited study time"""
        structure = self.user_structures[structure_name]
        content = structure['content']
        
        self.clear_screen()
        print(self.generate_ascii_banner("MEMORY MATRIX"))
        print(f"\nStructure: {structure_name}")
        print("You have only 30 seconds to memorize!")
        print("\nPress ENTER when ready...")
        input()
        
        self.clear_screen()
        print("MEMORIZE THIS:")
        print("="*70)
        print(content)
        print("="*70)
        
        for remaining in range(30, 0, -1):
            print(f"\rMemorization time: {remaining:2d} seconds", end='')
            time.sleep(1)
        
        self.clear_screen()
        print("TIME'S UP! Now type from memory!")
        print("-"*70)
        
        lines = []
        start_time = time.time()
        
        while True:
            line = input()
            if line.strip() == 'DONE':
                break
            lines.append(line)
        
        elapsed_time = time.time() - start_time
        user_input = '\n'.join(lines)
        
        accuracy = self.calculate_accuracy(content, user_input)
        
        # Bonus points for memory challenge
        memory_bonus = int(accuracy * 500)
        self.score += memory_bonus
        
        print("\n" + "="*70)
        print("MEMORY MATRIX RESULTS")
        print("="*70)
        print(f"Accuracy: {accuracy*100:.1f}%")
        print(f"Time: {elapsed_time:.1f} seconds")
        print(f"Memory Bonus: +{memory_bonus} points")
        
        input("\nPress ENTER to continue...")
    
    def calculate_accuracy(self, expected: str, actual: str) -> float:
        """Calculate accuracy between expected and actual text"""
        if not actual:
            return 0.0
        
        matcher = SequenceMatcher(None, expected.strip(), actual.strip())
        return matcher.ratio()
    
    def display_results(self, structure_name: str, accuracy: float, 
                       wpm: float, elapsed_time: float, 
                       user_input: str, expected: str):
        """Display detailed results"""
        self.clear_screen()
        print(self.generate_ascii_banner("RESULTS"))
        
        base_score = int(accuracy * 1000)
        speed_bonus = int(wpm * 10)
        total_score = base_score + speed_bonus
        
        print(f"\nStructure: {structure_name}")
        print(f"Accuracy: {accuracy*100:.1f}%")
        print(f"Speed: {wpm:.1f} WPM")
        print(f"Time: {elapsed_time:.1f} seconds")
        
        print("\nSCORE BREAKDOWN:")
        print(f"  Base Score: {base_score}")
        print(f"  Speed Bonus: {speed_bonus}")
        print(f"  TOTAL: {total_score}")
        
        self.score += total_score
        self.xp += total_score // 20
        
        # Check for achievements
        if wpm > 100 and not self.achievements['speed_demon']:
            self.unlock_achievement('speed_demon', 'Speed Demon - 100+ WPM!')
        
        # Update stats
        self.stats['total_words_typed'] += len(user_input.split())
        if wpm > self.stats['fastest_wpm']:
            self.stats['fastest_wpm'] = wpm
        
        print(f"\nTotal Score: {self.score} | XP: {self.xp}")
        
        # Level up check
        if self.xp >= self.level * 1000:
            self.level_up()
        
        input("\nPress ENTER to continue...")
    
    def level_up(self):
        """Handle level up"""
        self.level += 1
        print("\n" + "="*70)
        print(f"LEVEL UP! You are now level {self.level}!")
        print("="*70)
        
        # Rewards
        reward = random.choice(['time_freeze', 'hint_peek', 'error_shield', 'double_points'])
        self.power_ups[reward] += 1
        print(f"Reward: +1 {reward.replace('_', ' ').title()} power-up!")
    
    def unlock_achievement(self, achievement: str, message: str):
        """Unlock an achievement"""
        self.achievements[achievement] = True
        self.stats['achievement_points'] += 100
        
        print("\n" + "="*70)
        print("ACHIEVEMENT UNLOCKED!")
        print(message)
        print("+100 Achievement Points")
        print("="*70)
        time.sleep(2)
    
    def save_game_data(self):
        """Save game progress"""
        save_data = {
            'structures': self.user_structures,
            'stats': self.stats,
            'achievements': self.achievements,
            'score': self.score,
            'level': self.level,
            'xp': self.xp,
            'power_ups': self.power_ups
        }
        
        try:
            with open(self.save_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception as e:
            print(f"Could not save game: {e}")
    
    def load_game_data(self):
        """Load saved game progress"""
        if self.save_file.exists():
            try:
                with open(self.save_file, 'rb') as f:
                    save_data = pickle.load(f)
                    
                self.user_structures = save_data.get('structures', {})
                self.stats = save_data.get('stats', self.stats)
                self.achievements = save_data.get('achievements', self.achievements)
                self.score = save_data.get('score', 0)
                self.level = save_data.get('level', 1)
                self.xp = save_data.get('xp', 0)
                self.power_ups = save_data.get('power_ups', self.power_ups)
            except Exception as e:
                print(f"Could not load save file: {e}")
    
    def cascade_mode(self, structure_name: str):
        """Lines disappear one by one"""
        print("\nCASCADE MODE - Coming soon!")
        input("Press ENTER to continue...")
    
    def mirror_mode(self, structure_name: str):
        """Type the structure backwards"""
        print("\nMIRROR MODE - Coming soon!")
        input("Press ENTER to continue...")
    
    def blind_mode(self, structure_name: str):
        """Type without seeing output"""
        print("\nBLIND MODE - Coming soon!")
        input("Press ENTER to continue...")
    
    def rhythm_mode(self, structure_name: str):
        """Maintain steady typing rhythm"""
        print("\nRHYTHM MODE - Coming soon!")
        input("Press ENTER to continue...")
    
    def view_stats(self):
        """Display player statistics"""
        self.clear_screen()
        print(self.generate_ascii_banner("PLAYER STATISTICS"))
        
        print(f"\nLevel: {self.level}")
        print(f"XP: {self.xp}/{self.level * 1000}")
        print(f"Total Score: {self.score}")
        print(f"Achievement Points: {self.stats['achievement_points']}")
        
        print("\nPERFORMANCE:")
        print(f"  Structures Created: {self.stats['total_structures_created']}")
        print(f"  Words Typed: {self.stats['total_words_typed']}")
        print(f"  Fastest WPM: {self.stats['fastest_wpm']:.1f}")
        print(f"  Longest Combo: {self.stats['longest_combo']}")
        
        print("\nPOWER-UPS:")
        for power, count in self.power_ups.items():
            print(f"  {power.replace('_', ' ').title()}: {count}")
        
        print("\nACHIEVEMENTS:")
        unlocked = sum(1 for v in self.achievements.values() if v)
        print(f"  Unlocked: {unlocked}/{len(self.achievements)}")
        
        for name, unlocked in self.achievements.items():
            status = "[X]" if unlocked else "[ ]"
            print(f"  {status} {name.replace('_', ' ').title()}")
        
        input("\nPress ENTER to continue...")
    
    def main_menu(self) -> str:
        """Display main menu"""
        self.clear_screen()
        print(self.generate_ascii_banner("MAIN MENU"))
        
        print(f"\nLevel {self.level} | {self.xp} XP | Score: {self.score}")
        
        print("\n1. CREATE STRUCTURE - Design your own Python file")
        print("2. PRACTICE - Type existing structures")
        print("3. CHALLENGES - Special game modes")
        print("4. STATISTICS - View your progress")
        print("5. MANAGE STRUCTURES - Edit/delete structures")
        print("6. SAVE & QUIT - Save progress and exit")
        
        return input("\nChoice: ").strip()
    
    def manage_structures(self):
        """Manage created structures"""
        if not self.user_structures:
            print("\nNo structures created yet!")
            input("Press ENTER to continue...")
            return
        
        self.clear_screen()
        print(self.generate_ascii_banner("STRUCTURE MANAGER"))
        
        for idx, (name, data) in enumerate(self.user_structures.items(), 1):
            print(f"\n{idx}. {name}")
            print(f"   Type: {data.get('type', 'Unknown')}")
            print(f"   Complexity: {data['complexity']}/10")
            print(f"   Times Practiced: {data['practice_count']}")
            print(f"   Best Accuracy: {data['best_accuracy']*100:.1f}%")
        
        print("\nOptions: [V]iew, [E]dit, [D]elete, [B]ack")
        choice = input("Choice: ").strip().lower()
        
        if choice == 'v':
            name = input("Structure name to view: ").strip().upper()
            if name in self.user_structures:
                print("\n" + "="*70)
                print(self.user_structures[name]['content'])
                print("="*70)
                input("\nPress ENTER to continue...")
        elif choice == 'd':
            name = input("Structure name to delete: ").strip().upper()
            if name in self.user_structures:
                confirm = input(f"Delete {name}? (y/n): ").strip().lower()
                if confirm == 'y':
                    del self.user_structures[name]
                    print(f"{name} deleted.")
                    time.sleep(1)
    
    def run(self):
        """Main game loop"""
        self.display_welcome()
        
        while True:
            choice = self.main_menu()
            
            if choice == '1':
                self.create_custom_structure()
            elif choice == '2':
                self.practice_structure()
            elif choice == '3':
                print("\nAdvanced challenges coming soon!")
                input("Press ENTER to continue...")
            elif choice == '4':
                self.view_stats()
            elif choice == '5':
                self.manage_structures()
            elif choice == '6':
                self.save_game_data()
                self.clear_screen()
                print(self.generate_ascii_banner("GOODBYE"))
                print(f"\nFinal Stats:")
                print(f"  Level: {self.level}")
                print(f"  Total Score: {self.score}")
                print(f"  Structures Created: {self.stats['total_structures_created']}")
                print("\nProgress saved! See you next time!")
                break
            else:
                print("Invalid choice!")
                time.sleep(1)


def main():
    """Entry point for the game"""
    game = StructureTypingGame()
    try:
        game.run()
    except KeyboardInterrupt:
        game.save_game_data()
        print("\n\nGame saved. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        game.save_game_data()


if __name__ == "__main__":
    main()
