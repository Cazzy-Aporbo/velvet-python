"""
MACHINE MIND 
Pattern Recognition and Algorithmic Thinking Training
A system that teaches you to think like a machine by developing
pattern recognition skills, logical reasoning, and systematic problem-solving
approaches that mirror how computers process information.
"""

import random
import time
import json
import math
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

class ThinkingMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    RECURSIVE = "recursive"
    ITERATIVE = "iterative"
    PROBABILISTIC = "probabilistic"

@dataclass
class PatternAnalysis:
    pattern_type: str
    confidence: float
    evidence: List[str]
    prediction: Any
    reasoning_chain: List[str]

class MachineThinkingGame:
    """
    Core game engine that teaches machine-like thinking patterns through
    progressive challenges that build pattern recognition and logical reasoning.
    """
    
    def __init__(self):
        self.player_name = ""
        self.thinking_level = 1
        self.pattern_recognition_score = 0
        self.logical_reasoning_score = 0
        self.systematic_analysis_score = 0
        self.prediction_accuracy = 0.0
        self.patterns_discovered = set()
        self.thinking_modes_mastered = set()
        self.session_data = []
        
    def initialize_training(self):
        """Initialize the machine thinking training system."""
        print("=" * 80)
        print("MACHINE MIND: ALGORITHMIC THINKING TRAINING SYSTEM")
        print("=" * 80)
        print("\nThis system will teach you to:")
        print("• Recognize patterns like a machine learning algorithm")
        print("• Think systematically through logical steps")
        print("• Process information in parallel streams")
        print("• Make predictions based on incomplete data")
        print("• Optimize solutions through iterative refinement")
        print("• Handle uncertainty with probabilistic reasoning")
        
        self.player_name = input("\nEnter your designation: ").strip()
        
        print(f"\nInitializing training protocol for {self.player_name}...")
        print("Calibrating pattern recognition systems...")
        print("Loading algorithmic thinking modules...")
        print("Training sequence activated.")
        
        input("\nPress Enter to begin Level 1: Sequential Pattern Recognition...")

class Level1_SequentialPatterns:
    """
    Teaches recognition of sequential patterns - the foundation of machine thinking.
    Focuses on identifying rules, predicting next elements, and understanding sequences.
    """
    
    def __init__(self, game_controller):
        self.game = game_controller
        self.level_name = "Sequential Pattern Recognition"
        self.patterns_tested = []
        
    def begin_sequential_training(self):
        """Start sequential pattern recognition training."""
        print(f"\nLEVEL 1: {self.level_name}")
        print("-" * 60)
        print("MACHINE THINKING PRINCIPLE:")
        print("Machines excel at finding patterns in sequences by analyzing")
        print("differences, ratios, and transformations between elements.")
        
        print("\nTRAINING OBJECTIVES:")
        print("• Identify arithmetic and geometric progressions")
        print("• Recognize transformation rules")
        print("• Predict sequence continuations")
        print("• Analyze pattern confidence levels")
        
        challenges = [
            self.arithmetic_sequence_analysis(),
            self.geometric_pattern_detection(),
            self.transformation_rule_discovery(),
            self.complex_sequence_prediction(),
            self.pattern_confidence_assessment()
        ]
        
        score = sum(challenges)
        total = len(challenges)
        
        print(f"\nSEQUENTIAL PATTERN ANALYSIS COMPLETE")
        print(f"Recognition Accuracy: {score}/{total} ({score/total*100:.1f}%)")
        
        if score >= total * 0.7:
            self.game.thinking_modes_mastered.add(ThinkingMode.SEQUENTIAL)
            self.game.pattern_recognition_score += score
            return True
        return False
    
    def arithmetic_sequence_analysis(self):
        """Teach systematic analysis of arithmetic sequences."""
        print("\nCHALLENGE 1: Arithmetic Sequence Analysis")
        print("-" * 45)
        print("MACHINE APPROACH: Calculate differences between consecutive terms")
        
        sequences = [
            ([2, 5, 8, 11, 14], "Simple arithmetic: +3"),
            ([100, 93, 86, 79, 72], "Decreasing arithmetic: -7"),
            ([1, 4, 9, 16, 25], "Perfect squares: n²"),
            ([3, 7, 15, 31, 63], "Powers of 2 minus 1: 2^n - 1")
        ]
        
        correct_predictions = 0
        
        for sequence, explanation in sequences:
            print(f"\nSequence: {sequence}")
            print("SYSTEMATIC ANALYSIS:")
            
            # Show machine-like analysis
            differences = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
            print(f"First differences: {differences}")
            
            if len(set(differences)) == 1:
                print(f"Constant difference detected: {differences[0]}")
                next_val = sequence[-1] + differences[0]
                print(f"Prediction algorithm: last_value + difference = {next_val}")
            else:
                # Check second differences
                second_diff = [differences[i+1] - differences[i] for i in range(len(differences)-1)]
                print(f"Second differences: {second_diff}")
                
                if len(set(second_diff)) == 1:
                    print(f"Constant second difference: {second_diff[0]}")
                    next_diff = differences[-1] + second_diff[0]
                    next_val = sequence[-1] + next_diff
                    print(f"Prediction: {sequence[-1]} + {next_diff} = {next_val}")
                else:
                    # Pattern recognition for special sequences
                    if all(x == i*i for i, x in enumerate(sequence, 1)):
                        next_val = len(sequence) + 1
                        next_val = next_val * next_val
                        print(f"Perfect square pattern detected: next = {len(sequence)+1}² = {next_val}")
                    else:
                        # Check exponential patterns
                        ratios = [sequence[i+1] / sequence[i] if sequence[i] != 0 else 0 for i in range(len(sequence)-1)]
                        print(f"Ratios: {[f'{r:.2f}' for r in ratios]}")
                        
                        # Custom pattern detection
                        next_val = self.detect_complex_pattern(sequence)
            
            user_prediction = input(f"Your prediction for next term: ")
            
            try:
                user_val = int(user_prediction)
                if user_val == next_val:
                    print(f"CORRECT! Pattern: {explanation}")
                    correct_predictions += 1
                else:
                    print(f"INCORRECT. Correct answer: {next_val}")
                    print(f"Pattern: {explanation}")
            except ValueError:
                print(f"Invalid input. Correct answer: {next_val}")
                print(f"Pattern: {explanation}")
        
        return 1 if correct_predictions >= 3 else 0
    
    def detect_complex_pattern(self, sequence):
        """Advanced pattern detection for complex sequences."""
        # Check for exponential patterns like 2^n - 1
        if len(sequence) >= 3:
            # Test if sequence follows 2^n - 1 pattern
            powers_of_2_minus_1 = [2**i - 1 for i in range(2, len(sequence) + 2)]
            if sequence == powers_of_2_minus_1[:len(sequence)]:
                return 2**(len(sequence) + 1) - 1
        
        # Default to linear extrapolation
        if len(sequence) >= 2:
            diff = sequence[-1] - sequence[-2]
            return sequence[-1] + diff
        
        return sequence[-1] if sequence else 0
    
    def geometric_pattern_detection(self):
        """Teach recognition of multiplicative patterns."""
        print("\nCHALLENGE 2: Geometric Pattern Detection")
        print("-" * 45)
        print("MACHINE APPROACH: Calculate ratios between consecutive terms")
        
        sequences = [
            ([2, 6, 18, 54, 162], 3, "Multiply by 3"),
            ([1000, 100, 10, 1], 0.1, "Divide by 10"),
            ([1, -2, 4, -8, 16], -2, "Multiply by -2"),
            ([3, 12, 48, 192], 4, "Multiply by 4")
        ]
        
        correct = 0
        
        for sequence, ratio, description in sequences:
            print(f"\nAnalyzing: {sequence}")
            
            # Calculate ratios systematically
            ratios = []
            for i in range(len(sequence) - 1):
                if sequence[i] != 0:
                    r = sequence[i + 1] / sequence[i]
                    ratios.append(r)
            
            print(f"Calculated ratios: {[f'{r:.2f}' for r in ratios]}")
            
            # Check if ratios are constant
            if len(set(ratios)) == 1 or all(abs(r - ratios[0]) < 0.001 for r in ratios):
                detected_ratio = ratios[0]
                print(f"CONSTANT RATIO DETECTED: {detected_ratio}")
                prediction = sequence[-1] * detected_ratio
                print(f"Next term calculation: {sequence[-1]} × {detected_ratio} = {prediction}")
            else:
                print("No constant ratio found - not a geometric sequence")
                prediction = sequence[-1]  # Fallback
            
            user_answer = input("Predict the next term: ")
            
            try:
                user_val = float(user_answer)
                expected = sequence[-1] * ratio
                if abs(user_val - expected) < 0.1:
                    print(f"CORRECT! {description}")
                    correct += 1
                else:
                    print(f"INCORRECT. Answer: {expected} ({description})")
            except ValueError:
                print(f"Invalid input. Answer: {sequence[-1] * ratio}")
        
        return 1 if correct >= 3 else 0
    
    def transformation_rule_discovery(self):
        """Teach discovery of transformation rules in sequences."""
        print("\nCHALLENGE 3: Transformation Rule Discovery")
        print("-" * 45)
        print("MACHINE APPROACH: Analyze how each term transforms to the next")
        
        transformations = [
            {
                'sequence': [1, 1, 2, 3, 5, 8, 13],
                'rule': 'Fibonacci: each term = sum of previous two',
                'next': 21
            },
            {
                'sequence': [1, 4, 2, 8, 6, 24, 22],
                'rule': 'Alternating: ×4 if odd position, -2 if even position',
                'next': 88
            },
            {
                'sequence': [2, 3, 5, 7, 11, 13, 17],
                'rule': 'Prime numbers sequence',
                'next': 19
            },
            {
                'sequence': [1, 11, 21, 1211, 111221, 312211],
                'rule': 'Look-and-say sequence: describe previous term',
                'next': '13112221'
            }
        ]
        
        correct = 0
        
        for trans in transformations:
            sequence = trans['sequence']
            print(f"\nSequence: {sequence}")
            print("SYSTEMATIC ANALYSIS:")
            
            # Provide analysis framework
            if 'Fibonacci' in trans['rule']:
                print("Testing sum relationships...")
                for i in range(2, len(sequence)):
                    sum_prev_two = sequence[i-2] + sequence[i-1]
                    print(f"{sequence[i-2]} + {sequence[i-1]} = {sum_prev_two} (actual: {sequence[i]})")
                print("PATTERN: Each term equals sum of previous two terms")
                
            elif 'Alternating' in trans['rule']:
                print("Testing position-based transformations...")
                for i in range(1, len(sequence)):
                    if i % 2 == 1:  # Odd position (1-indexed)
                        expected = sequence[i-1] * 4
                        print(f"Position {i}: {sequence[i-1]} × 4 = {expected} (actual: {sequence[i]})")
                    else:  # Even position
                        expected = sequence[i-1] - 2
                        print(f"Position {i}: {sequence[i-1]} - 2 = {expected} (actual: {sequence[i]})")
                
            elif 'Prime' in trans['rule']:
                print("Testing primality...")
                for num in sequence:
                    is_prime = self.is_prime(num)
                    print(f"{num}: {'Prime' if is_prime else 'Not prime'}")
                print("PATTERN: All terms are prime numbers")
                
            elif 'Look-and-say' in trans['rule']:
                print("Testing descriptive transformation...")
                for i in range(1, len(sequence)):
                    prev_str = str(sequence[i-1])
                    description = self.look_and_say(prev_str)
                    print(f"'{prev_str}' described as: '{description}' (actual: '{sequence[i]}')")
            
            user_prediction = input("What is the next term? ")
            
            expected = trans['next']
            if str(user_prediction).strip() == str(expected):
                print(f"CORRECT! Rule: {trans['rule']}")
                correct += 1
            else:
                print(f"INCORRECT. Answer: {expected}")
                print(f"Rule: {trans['rule']}")
        
        return 1 if correct >= 2 else 0
    
    def is_prime(self, n):
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def look_and_say(self, s):
        """Generate look-and-say sequence description."""
        result = ""
        i = 0
        while i < len(s):
            count = 1
            digit = s[i]
            while i + count < len(s) and s[i + count] == digit:
                count += 1
            result += str(count) + digit
            i += count
        return result
    
    def complex_sequence_prediction(self):
        """Advanced sequence prediction with multiple pattern types."""
        print("\nCHALLENGE 4: Complex Sequence Prediction")
        print("-" * 45)
        print("MACHINE APPROACH: Multi-level pattern analysis")
        
        complex_sequences = [
            {
                'data': [1, 2, 4, 7, 11, 16, 22],
                'analysis': 'Differences: +1, +2, +3, +4, +5, +6 (triangular numbers)',
                'next': 29
            },
            {
                'data': [2, 3, 5, 8, 13, 21, 34],
                'analysis': 'Fibonacci sequence starting with 2, 3',
                'next': 55
            },
            {
                'data': [1, 4, 9, 16, 25, 36],
                'analysis': 'Perfect squares: n²',
                'next': 49
            }
        ]
        
        print("You will analyze sequences using systematic machine-like approaches:")
        
        correct = 0
        for seq_data in complex_sequences:
            sequence = seq_data['data']
            print(f"\nSequence: {sequence}")
            
            # Guide through systematic analysis
            print("ANALYSIS FRAMEWORK:")
            print("1. Calculate first differences")
            first_diff = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
            print(f"   First differences: {first_diff}")
            
            print("2. Calculate second differences")
            if len(first_diff) > 1:
                second_diff = [first_diff[i+1] - first_diff[i] for i in range(len(first_diff)-1)]
                print(f"   Second differences: {second_diff}")
            
            print("3. Check for patterns")
            print(f"   Analysis: {seq_data['analysis']}")
            
            prediction = input("Based on this analysis, predict the next term: ")
            
            try:
                if int(prediction) == seq_data['next']:
                    print("CORRECT! Pattern recognition successful.")
                    correct += 1
                else:
                    print(f"INCORRECT. Correct answer: {seq_data['next']}")
            except ValueError:
                print(f"Invalid input. Correct answer: {seq_data['next']}")
        
        return 1 if correct >= 2 else 0
    
    def pattern_confidence_assessment(self):
        """Teach assessment of pattern confidence levels."""
        print("\nCHALLENGE 5: Pattern Confidence Assessment")
        print("-" * 45)
        print("MACHINE APPROACH: Quantify confidence in pattern predictions")
        
        test_cases = [
            {
                'sequence': [2, 4, 6, 8, 10],
                'confidence': 'HIGH',
                'reason': 'Perfect arithmetic progression with constant difference +2'
            },
            {
                'sequence': [1, 3, 6, 10],
                'confidence': 'MEDIUM',
                'reason': 'Triangular numbers, but short sequence'
            },
            {
                'sequence': [1, 2, 4, 8],
                'confidence': 'MEDIUM',
                'reason': 'Could be powers of 2, but only 4 data points'
            },
            {
                'sequence': [5, 7, 12],
                'confidence': 'LOW',
                'reason': 'Too few data points, multiple patterns possible'
            }
        ]
        
        print("Rate the confidence level for pattern predictions:")
        print("HIGH: Very certain about the pattern")
        print("MEDIUM: Reasonably confident but need more data")
        print("LOW: Uncertain, multiple patterns possible")
        
        correct = 0
        for case in test_cases:
            print(f"\nSequence: {case['sequence']}")
            
            # Show analysis
            if len(case['sequence']) >= 2:
                diffs = [case['sequence'][i+1] - case['sequence'][i] for i in range(len(case['sequence'])-1)]
                print(f"Differences: {diffs}")
                
                if len(case['sequence']) >= 3:
                    ratios = [case['sequence'][i+1] / case['sequence'][i] if case['sequence'][i] != 0 else 0 
                             for i in range(len(case['sequence'])-1)]
                    print(f"Ratios: {[f'{r:.2f}' for r in ratios]}")
            
            print(f"Data points available: {len(case['sequence'])}")
            
            confidence = input("Confidence level (HIGH/MEDIUM/LOW): ").strip().upper()
            
            if confidence == case['confidence']:
                print(f"CORRECT! {case['reason']}")
                correct += 1
            else:
                print(f"INCORRECT. Correct: {case['confidence']}")
                print(f"Reason: {case['reason']}")
        
        return 1 if correct >= 3 else 0

class Level2_ParallelProcessing:
    """
    Teaches parallel thinking - processing multiple information streams simultaneously.
    Develops ability to track multiple variables and relationships concurrently.
    """
    
    def __init__(self, game_controller):
        self.game = game_controller
        self.level_name = "Parallel Information Processing"
        
    def begin_parallel_training(self):
        """Start parallel processing training."""
        print(f"\nLEVEL 2: {self.level_name}")
        print("-" * 60)
        print("MACHINE THINKING PRINCIPLE:")
        print("Machines process multiple data streams simultaneously, tracking")
        print("relationships between variables and updating states in parallel.")
        
        print("\nTRAINING OBJECTIVES:")
        print("• Track multiple variables simultaneously")
        print("• Process concurrent data streams")
        print("• Identify cross-stream correlations")
        print("• Maintain state across parallel operations")
        
        challenges = [
            self.multi_variable_tracking(),
            self.concurrent_pattern_analysis(),
            self.state_machine_simulation(),
            self.parallel_optimization(),
            self.cross_correlation_detection()
        ]
        
        score = sum(challenges)
        total = len(challenges)
        
        print(f"\nPARALLEL PROCESSING ANALYSIS COMPLETE")
        print(f"Processing Efficiency: {score}/{total} ({score/total*100:.1f}%)")
        
        if score >= total * 0.6:
            self.game.thinking_modes_mastered.add(ThinkingMode.PARALLEL)
            return True
        return False
    
    def multi_variable_tracking(self):
        """Teach simultaneous tracking of multiple variables."""
        print("\nCHALLENGE 1: Multi-Variable State Tracking")
        print("-" * 45)
        print("MACHINE APPROACH: Maintain separate counters for each variable")
        
        print("\nSCENARIO: Traffic intersection monitoring")
        print("Track: cars_north, cars_south, cars_east, cars_west")
        print("Rules: +1 when car arrives, -1 when car passes")
        
        # Simulate traffic data
        events = [
            ("north", "arrive"), ("south", "arrive"), ("east", "arrive"),
            ("north", "pass"), ("west", "arrive"), ("south", "pass"),
            ("east", "arrive"), ("north", "arrive"), ("west", "pass"),
            ("south", "arrive"), ("east", "pass"), ("north", "pass")
        ]
        
        # Initialize counters
        counters = {"north": 0, "south": 0, "east": 0, "west": 0}
        
        print("\nProcessing events in parallel:")
        print("Event | North | South | East | West | Total")
        print("-" * 45)
        
        for i, (direction, action) in enumerate(events):
            if action == "arrive":
                counters[direction] += 1
            else:  # pass
                counters[direction] -= 1
            
            total = sum(counters.values())
            print(f"{i+1:5d} | {counters['north']:5d} | {counters['south']:5d} | {counters['east']:4d} | {counters['west']:4d} | {total:5d}")
        
        # Test tracking ability
        print(f"\nFinal state: {counters}")
        
        questions = [
            ("Which direction has the most cars waiting?", max(counters, key=counters.get)),
            ("What is the total number of cars at the intersection?", sum(counters.values())),
            ("How many cars are waiting north and south combined?", counters['north'] + counters['south'])
        ]
        
        correct = 0
        for question, answer in questions:
            user_answer = input(f"{question} ")
            if str(user_answer).strip().lower() == str(answer).lower():
                print("CORRECT!")
                correct += 1
            else:
                print(f"INCORRECT. Answer: {answer}")
        
        return 1 if correct >= 2 else 0
    
    def concurrent_pattern_analysis(self):
        """Analyze patterns in multiple data streams simultaneously."""
        print("\nCHALLENGE 2: Concurrent Pattern Analysis")
        print("-" * 45)
        print("MACHINE APPROACH: Process multiple sequences in parallel")
        
        # Multiple sequences to analyze simultaneously
        sequences = {
            'A': [2, 4, 6, 8, 10, 12],
            'B': [1, 4, 9, 16, 25, 36],
            'C': [1, 1, 2, 3, 5, 8],
            'D': [10, 20, 15, 30, 25, 50]
        }
        
        print("Analyzing four sequences simultaneously:")
        for name, seq in sequences.items():
            print(f"Sequence {name}: {seq}")
        
        print("\nPARALLEL ANALYSIS:")
        
        # Analyze each sequence
        analyses = {}
        for name, seq in sequences.items():
            # Calculate differences
            diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
            
            # Determine pattern type
            if len(set(diffs)) == 1:
                pattern = f"Arithmetic (+{diffs[0]})"
            elif all(seq[i] == i*i for i in range(1, len(seq)+1)):
                pattern = "Perfect squares"
            elif len(seq) >= 3 and all(seq[i] == seq[i-1] + seq[i-2] for i in range(2, len(seq))):
                pattern = "Fibonacci"
            else:
                pattern = "Complex/Mixed"
            
            analyses[name] = {
                'differences': diffs,
                'pattern': pattern,
                'next_predicted': self.predict_next(seq, pattern)
            }
        
        # Display parallel analysis results
        print("\nSequence | Pattern Type    | Next Value")
        print("-" * 40)
        for name in ['A', 'B', 'C', 'D']:
            analysis = analyses[name]
            print(f"    {name}    | {analysis['pattern']:15} | {analysis['next_predicted']:10}")
        
        # Test understanding
        questions = [
            ("Which sequence follows an arithmetic progression?", "A"),
            ("Which sequence represents perfect squares?", "B"),
            ("Which sequence is Fibonacci-like?", "C"),
            ("What is the next value in sequence A?", analyses['A']['next_predicted'])
        ]
        
        correct = 0
        for question, answer in questions:
            user_answer = input(f"{question} ")
            if str(user_answer).strip().upper() == str(answer).upper():
                print("CORRECT!")
                correct += 1
            else:
                print(f"INCORRECT. Answer: {answer}")
        
        return 1 if correct >= 3 else 0
    
    def predict_next(self, sequence, pattern_type):
        """Predict next value based on pattern type."""
        if "Arithmetic" in pattern_type:
            diff = sequence[1] - sequence[0]
            return sequence[-1] + diff
        elif "Perfect squares" in pattern_type:
            next_n = len(sequence) + 1
            return next_n * next_n
        elif "Fibonacci" in pattern_type:
            return sequence[-1] + sequence[-2]
        else:
            # Default to last difference
            if len(sequence) >= 2:
                return sequence[-1] + (sequence[-1] - sequence[-2])
            return sequence[-1]
    
    def state_machine_simulation(self):
        """Simulate a state machine with multiple concurrent states."""
        print("\nCHALLENGE 3: State Machine Simulation")
        print("-" * 45)
        print("MACHINE APPROACH: Track state transitions across multiple systems")
        
        print("\nSCENARIO: Smart home automation system")
        print("Systems: Lights, Security, Temperature, Music")
        print("Each system has states and responds to events")
        
        # Define state machines
        systems = {
            'lights': {'state': 'off', 'states': ['off', 'dim', 'bright']},
            'security': {'state': 'armed', 'states': ['armed', 'disarmed', 'alarm']},
            'temperature': {'state': 'auto', 'states': ['auto', 'heat', 'cool', 'off']},
            'music': {'state': 'off', 'states': ['off', 'playing', 'paused']}
        }
        
        # Define state transition rules
        transitions = {
            'lights': {
                'off': {'toggle': 'dim', 'brighten': 'bright'},
                'dim': {'toggle': 'bright', 'off': 'off'},
                'bright': {'toggle': 'off', 'dim': 'dim'}
            },
            'security': {
                'armed': {'disarm': 'disarmed', 'trigger': 'alarm'},
                'disarmed': {'arm': 'armed'},
                'alarm': {'reset': 'disarmed'}
            },
            'temperature': {
                'auto': {'heat': 'heat', 'cool': 'cool', 'off': 'off'},
                'heat': {'auto': 'auto', 'cool': 'cool', 'off': 'off'},
                'cool': {'auto': 'auto', 'heat': 'heat', 'off': 'off'},
                'off': {'auto': 'auto', 'heat': 'heat', 'cool': 'cool'}
            },
            'music': {
                'off': {'play': 'playing'},
                'playing': {'pause': 'paused', 'stop': 'off'},
                'paused': {'play': 'playing', 'stop': 'off'}
            }
        }
        
        print(f"\nInitial states:")
        for system, data in systems.items():
            print(f"{system}: {data['state']}")
        
        # Process events
        events = [
            ('lights', 'toggle'),
            ('music', 'play'),
            ('security', 'disarm'),
            ('temperature', 'heat'),
            ('lights', 'brighten'),
            ('music', 'pause'),
            ('security', 'arm'),
            ('temperature', 'auto')
        ]
        
        print(f"\nProcessing events:")
        print("Event | Lights | Security  | Temperature | Music")
        print("-" * 50)
        
        for system, event in events:
            current_state = systems[system]['state']
            if event in transitions[system][current_state]:
                new_state = transitions[system][current_state][event]
                systems[system]['state'] = new_state
            
            states_display = [systems[s]['state'] for s in ['lights', 'security', 'temperature', 'music']]
            print(f"{system:>6} | {states_display[0]:6} | {states_display[1]:9} | {states_display[2]:11} | {states_display[3]:5}")
        
        # Test final state understanding
        final_states = {system: data['state'] for system, data in systems.items()}
        
        questions = [
            ("What is the final state of the lights?", final_states['lights']),
            ("What is the final state of the security system?", final_states['security']),
            ("How many systems are currently 'off'?", sum(1 for state in final_states.values() if state == 'off'))
        ]
        
        correct = 0
        for question, answer in questions:
            user_answer = input(f"{question} ")
            if str(user_answer).strip().lower() == str(answer).lower():
                print("CORRECT!")
                correct += 1
            else:
                print(f"INCORRECT. Answer: {answer}")
        
        return 1 if correct >= 2 else 0
    
    def parallel_optimization
