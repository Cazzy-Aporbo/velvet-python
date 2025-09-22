"""
RECURSIVE THINKING MASTERY: A Progressive Learning System
========================================================

A comprehensive educational program that teaches recursive thinking through
increasingly complex challenges, code demonstrations, and practical applications.
Each level builds upon previous concepts while introducing new recursive patterns.
"""

import time
import random
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import json

class RecursiveThinkingGame:
    """
    Main game controller that manages progression through recursive thinking concepts.
    Tracks student progress, manages difficulty scaling, and provides detailed feedback.
    """
    
    def __init__(self):
        self.student_name = ""
        self.current_level = 1
        self.max_level = 12
        self.skills_mastered = set()
        self.performance_history = []
        self.current_score = 0
        self.total_challenges_completed = 0
        
        # Skill tree mapping
        self.skill_dependencies = {
            'base_case_recognition': [],
            'simple_recursion': ['base_case_recognition'],
            'tail_recursion': ['simple_recursion'],
            'tree_recursion': ['simple_recursion'],
            'mutual_recursion': ['simple_recursion'],
            'memoization': ['tree_recursion'],
            'backtracking': ['tree_recursion'],
            'divide_conquer': ['tree_recursion'],
            'dynamic_programming': ['memoization'],
            'recursive_data_structures': ['tree_recursion'],
            'advanced_optimization': ['memoization', 'tail_recursion'],
            'recursive_algorithms': ['divide_conquer', 'backtracking']
        }
        
    def initialize_game(self):
        """Set up the learning environment and gather student information."""
        print("=" * 80)
        print("RECURSIVE THINKING MASTERY SYSTEM")
        print("=" * 80)
        print("\nWelcome to an advanced learning system designed to develop")
        print("deep understanding of recursive thinking patterns.")
        print("\nThis system will guide you through:")
        print("- Fundamental recursive concepts")
        print("- Pattern recognition and application")
        print("- Performance optimization techniques")
        print("- Real-world problem solving")
        print("- Advanced algorithmic thinking")
        
        self.student_name = input("\nEnter your name: ").strip()
        
        print(f"\nHello {self.student_name}. Let's begin your journey into recursive thinking.")
        print("Each level will introduce new concepts and challenge your understanding.")
        print("Pay close attention to the code patterns and explanations.")
        
        input("\nPress Enter to start Level 1...")

class Level1_BaseCase:
    """
    Foundation level focusing on understanding base cases and termination conditions.
    This is crucial for preventing infinite recursion and understanding when to stop.
    """
    
    def __init__(self, game_controller):
        self.game = game_controller
        self.level_name = "Base Case Recognition and Termination"
        self.concepts_covered = [
            "Understanding base cases",
            "Preventing infinite recursion", 
            "Identifying termination conditions",
            "Simple recursive structure"
        ]
        
    def teach_base_case_concept(self):
        """Comprehensive explanation of base cases with multiple examples."""
        print(f"\nLEVEL 1: {self.level_name}")
        print("-" * 60)
        print("CORE CONCEPT: Every recursive function needs a stopping condition")
        print("called a BASE CASE. Without it, the function calls itself forever.")
        
        print("\nTHINK OF RECURSION LIKE RUSSIAN DOLLS:")
        print("- Each doll contains a smaller doll (recursive case)")
        print("- The smallest doll has no doll inside (base case)")
        print("- Without the smallest doll, you'd never stop opening them")
        
        input("\nPress Enter to see code examples...")
        
        print("\nEXAMPLE 1: Countdown Function")
        print("=" * 40)
        print("""
def countdown_broken(n):
    print(n)
    countdown_broken(n - 1)  # INFINITE RECURSION - NO BASE CASE!

def countdown_correct(n):
    if n <= 0:               # BASE CASE: Stop when n reaches 0
        print("Done!")
        return
    print(n)                 # RECURSIVE CASE: Do work
    countdown_correct(n - 1) # RECURSIVE CALL: Move toward base case
        """)
        
        # Demonstrate the working version
        print("DEMONSTRATION: countdown_correct(5)")
        self.demonstrate_countdown(5)
        
        print("\nEXAMPLE 2: Factorial Function")
        print("=" * 40)
        print("""
def factorial(n):
    # BASE CASE: factorial of 0 or 1 is 1
    if n <= 1:
        return 1
    
    # RECURSIVE CASE: n! = n * (n-1)!
    return n * factorial(n - 1)
        """)
        
        # Demonstrate factorial with step-by-step breakdown
        print("STEP-BY-STEP: factorial(4)")
        self.demonstrate_factorial_steps(4)
        
        return self.base_case_challenge()
    
    def demonstrate_countdown(self, n):
        """Show countdown execution with detailed tracing."""
        print("EXECUTION TRACE:")
        original_n = n
        depth = 0
        
        while n > 0:
            print(f"{'  ' * depth}countdown_correct({n})")
            print(f"{'  ' * depth}  prints: {n}")
            print(f"{'  ' * depth}  calls: countdown_correct({n-1})")
            n -= 1
            depth += 1
        
        print(f"{'  ' * depth}countdown_correct(0)")
        print(f"{'  ' * depth}  BASE CASE reached: prints 'Done!' and returns")
        
        # Show the actual execution
        print(f"\nACTUAL OUTPUT:")
        for i in range(original_n, 0, -1):
            print(i)
        print("Done!")
    
    def demonstrate_factorial_steps(self, n):
        """Show factorial calculation with complete call stack visualization."""
        print("CALL STACK VISUALIZATION:")
        
        # Build the call stack going down
        calls = []
        current = n
        while current > 1:
            calls.append(current)
            current -= 1
        calls.append(1)
        
        # Show the descent
        print("DESCENT (making recursive calls):")
        for i, call_value in enumerate(calls[:-1]):
            indent = "  " * i
            print(f"{indent}factorial({call_value})")
            print(f"{indent}  needs factorial({call_value - 1})")
        
        # Show base case
        indent = "  " * (len(calls) - 1)
        print(f"{indent}factorial(1)")
        print(f"{indent}  BASE CASE: returns 1")
        
        # Show the ascent with calculations
        print("\nASCENT (returning values):")
        result = 1
        for i in range(len(calls) - 1, -1, -1):
            call_value = calls[i]
            indent = "  " * i
            if call_value == 1:
                print(f"{indent}factorial(1) returns 1")
                result = 1
            else:
                result = call_value * result
                print(f"{indent}factorial({call_value}) returns {call_value} * {result // call_value} = {result}")
        
        print(f"\nFINAL RESULT: factorial({n}) = {result}")
    
    def base_case_challenge(self):
        """Interactive challenge to test base case understanding."""
        print("\nCHALLENGE: Identify the Base Case")
        print("=" * 40)
        
        challenges = [
            {
                'code': '''
def sum_digits(n):
    if n < 10:
        return n
    return (n % 10) + sum_digits(n // 10)
                ''',
                'question': 'What is the base case in this function?',
                'correct': 'n < 10',
                'explanation': 'When n is a single digit (< 10), we return n directly without recursion.'
            },
            {
                'code': '''
def power(base, exp):
    if exp == 0:
        return 1
    return base * power(base, exp - 1)
                ''',
                'question': 'What condition stops the recursion?',
                'correct': 'exp == 0',
                'explanation': 'Any number to the power of 0 equals 1, so we return 1 when exp reaches 0.'
            }
        ]
        
        score = 0
        for i, challenge in enumerate(challenges, 1):
            print(f"\nCHALLENGE {i}:")
            print(challenge['code'])
            print(f"QUESTION: {challenge['question']}")
            
            answer = input("YOUR ANSWER: ").strip()
            
            if challenge['correct'].lower() in answer.lower():
                print("CORRECT!")
                print(f"EXPLANATION: {challenge['explanation']}")
                score += 1
            else:
                print(f"INCORRECT. The correct answer is: {challenge['correct']}")
                print(f"EXPLANATION: {challenge['explanation']}")
        
        print(f"\nLEVEL 1 COMPLETE! Score: {score}/{len(challenges)}")
        
        if score >= len(challenges) // 2:
            self.game.skills_mastered.add('base_case_recognition')
            return True
        else:
            print("Please review the concepts and try again.")
            return False

class Level2_SimpleRecursion:
    """
    Introduction to basic recursive patterns and thinking recursively about problems.
    Covers linear recursion and building intuition for recursive problem solving.
    """
    
    def __init__(self, game_controller):
        self.game = game_controller
        self.level_name = "Simple Recursive Patterns"
        
    def teach_recursive_thinking(self):
        """Develop intuition for thinking recursively about problems."""
        print(f"\nLEVEL 2: {self.level_name}")
        print("-" * 60)
        print("CORE CONCEPT: Recursive thinking means solving a problem by")
        print("solving smaller versions of the same problem.")
        
        print("\nTHE RECURSIVE MINDSET:")
        print("1. Assume the function already works for smaller inputs")
        print("2. Figure out how to use that to solve the current problem")
        print("3. Handle the base case where no recursion is needed")
        
        print("\nEXAMPLE: Finding the length of a string recursively")
        print("=" * 50)
        
        print("""
PROBLEM: How long is the string "HELLO"?

RECURSIVE THINKING:
- If the string is empty, length is 0 (base case)
- Otherwise, length is 1 + length of the rest of the string

def string_length(s):
    # Base case: empty string has length 0
    if len(s) == 0:
        return 0
    
    # Recursive case: 1 + length of remaining string
    return 1 + string_length(s[1:])
        """)
        
        # Demonstrate with detailed tracing
        self.trace_string_length("HELLO")
        
        print("\nEXAMPLE: Reversing a string recursively")
        print("=" * 45)
        
        print("""
PROBLEM: Reverse the string "WORLD"

RECURSIVE THINKING:
- If string has 0 or 1 characters, it's already reversed
- Otherwise, reverse = last character + reverse of the rest

def reverse_string(s):
    # Base case: strings of length 0 or 1 are already reversed
    if len(s) <= 1:
        return s
    
    # Recursive case: last char + reverse of the rest
    return s[-1] + reverse_string(s[:-1])
        """)
        
        self.trace_string_reverse("WORLD")
        
        return self.simple_recursion_challenges()
    
    def trace_string_length(self, s):
        """Detailed execution trace for string length calculation."""
        print(f"\nTRACING: string_length('{s}')")
        print("=" * 40)
        
        original_s = s
        depth = 0
        
        while s:
            indent = "  " * depth
            print(f"{indent}string_length('{s}')")
            print(f"{indent}  String is not empty, so:")
            print(f"{indent}  return 1 + string_length('{s[1:]}')")
            s = s[1:]
            depth += 1
        
        # Base case
        indent = "  " * depth
        print(f"{indent}string_length('')")
        print(f"{indent}  BASE CASE: empty string, return 0")
        
        # Show the return journey
        print(f"\nRETURN JOURNEY:")
        for i in range(depth, -1, -1):
            indent = "  " * i
            if i == depth:
                print(f"{indent}returns 0")
            else:
                remaining_chars = depth - i
                print(f"{indent}returns 1 + {remaining_chars} = {remaining_chars + 1}")
        
        print(f"\nFINAL RESULT: length of '{original_s}' is {len(original_s)}")
    
    def trace_string_reverse(self, s):
        """Detailed execution trace for string reversal."""
        print(f"\nTRACING: reverse_string('{s}')")
        print("=" * 40)
        
        calls = []
        temp_s = s
        
        # Build call sequence
        while len(temp_s) > 1:
            calls.append(temp_s)
            temp_s = temp_s[:-1]
        calls.append(temp_s)
        
        # Show descent
        print("DESCENT:")
        for i, call_s in enumerate(calls[:-1]):
            indent = "  " * i
            print(f"{indent}reverse_string('{call_s}')")
            print(f"{indent}  return '{call_s[-1]}' + reverse_string('{call_s[:-1]}')")
        
        # Base case
        indent = "  " * (len(calls) - 1)
        print(f"{indent}reverse_string('{calls[-1]}')")
        print(f"{indent}  BASE CASE: length <= 1, return '{calls[-1]}'")
        
        # Show ascent
        print(f"\nASCENT:")
        result = calls[-1]
        for i in range(len(calls) - 1, -1, -1):
            indent = "  " * i
            if i == len(calls) - 1:
                print(f"{indent}returns '{result}'")
            else:
                char_to_add = calls[i][-1]
                result = char_to_add + result
                print(f"{indent}returns '{char_to_add}' + '{result[1:]}' = '{result}'")
        
        print(f"\nFINAL RESULT: reverse of '{s}' is '{result}'")
    
    def simple_recursion_challenges(self):
        """Progressive challenges to build recursive thinking skills."""
        print("\nRECURSIVE THINKING CHALLENGES")
        print("=" * 40)
        
        challenges = [
            self.challenge_sum_array(),
            self.challenge_find_maximum(),
            self.challenge_count_occurrences()
        ]
        
        score = sum(challenges)
        total = len(challenges)
        
        print(f"\nLEVEL 2 COMPLETE! Score: {score}/{total}")
        
        if score >= total * 0.7:
            self.game.skills_mastered.add('simple_recursion')
            return True
        else:
            print("Review the recursive thinking patterns and try again.")
            return False
    
    def challenge_sum_array(self):
        """Challenge: Implement recursive array sum."""
        print("\nCHALLENGE 1: Sum all elements in an array recursively")
        print("-" * 50)
        
        print("PROBLEM: Given an array [1, 2, 3, 4, 5], find the sum recursively.")
        print("\nTHINK RECURSIVELY:")
        print("- What's the base case? (empty array)")
        print("- How do you use the sum of a smaller array?")
        
        print("\nYOUR TASK: Complete this function")
        print("""
def recursive_sum(arr):
    # Base case: empty array
    if len(arr) == 0:
        return ____
    
    # Recursive case: first element + sum of rest
    return ____ + recursive_sum(____)
        """)
        
        print("Fill in the blanks:")
        base_case = input("Base case return value: ").strip()
        first_element = input("What to add to the recursive call: ").strip()
        recursive_part = input("What array to pass to recursive call: ").strip()
        
        correct_answers = ['0', 'arr[0]', 'arr[1:]']
        user_answers = [base_case, first_element, recursive_part]
        
        if (base_case == '0' and 
            'arr[0]' in first_element and 
            'arr[1:]' in recursive_part):
            print("EXCELLENT! You understand the pattern.")
            
            # Demonstrate their solution
            def recursive_sum(arr):
                if len(arr) == 0:
                    return 0
                return arr[0] + recursive_sum(arr[1:])
            
            test_array = [1, 2, 3, 4, 5]
            result = recursive_sum(test_array)
            print(f"Testing with {test_array}: sum = {result}")
            return 1
        else:
            print("Not quite right. The correct solution:")
            print("Base case: 0 (sum of empty array)")
            print("Recursive case: arr[0] + recursive_sum(arr[1:])")
            return 0
    
    def challenge_find_maximum(self):
        """Challenge: Find maximum element recursively."""
        print("\nCHALLENGE 2: Find the maximum element in an array")
        print("-" * 50)
        
        print("PROBLEM: Find the largest number in [3, 7, 2, 9, 1] recursively.")
        print("\nHINTS:")
        print("- Base case: array with one element")
        print("- Recursive case: max of first element and max of rest")
        
        print("\nImplement this logic:")
        print("def find_max(arr):")
        
        implementation = input("Write your recursive solution (one line): ").strip()
        
        # Test their understanding with explanation
        print("\nLet me show you the optimal solution:")
        print("""
def find_max(arr):
    # Base case: single element
    if len(arr) == 1:
        return arr[0]
    
    # Recursive case: max of first and max of rest
    return max(arr[0], find_max(arr[1:]))
        """)
        
        # Trace through execution
        test_arr = [3, 7, 2, 9, 1]
        print(f"\nTRACING find_max({test_arr}):")
        self.trace_find_max(test_arr)
        
        if 'max' in implementation.lower() and 'arr[0]' in implementation:
            print("Great job! You got the key insight.")
            return 1
        else:
            print("Study the solution above to understand the pattern.")
            return 0
    
    def trace_find_max(self, arr):
        """Trace the find_max execution."""
        if len(arr) == 1:
            print(f"find_max({arr}) = {arr[0]} (base case)")
            return arr[0]
        
        print(f"find_max({arr})")
        print(f"  = max({arr[0]}, find_max({arr[1:]}))")
        
        rest_max = self.trace_find_max(arr[1:])
        result = max(arr[0], rest_max)
        
        print(f"  = max({arr[0]}, {rest_max}) = {result}")
        return result
    
    def challenge_count_occurrences(self):
        """Challenge: Count occurrences of a value."""
        print("\nCHALLENGE 3: Count occurrences of a value in an array")
        print("-" * 55)
        
        print("PROBLEM: Count how many times 'a' appears in ['a', 'b', 'a', 'c', 'a']")
        
        print("\nYour approach:")
        approach = input("Describe your recursive strategy: ").strip()
        
        print("\nHere's the solution with detailed explanation:")
        print("""
def count_occurrences(arr, target):
    # Base case: empty array
    if len(arr) == 0:
        return 0
    
    # Check first element
    count_first = 1 if arr[0] == target else 0
    
    # Recursive case: count in first + count in rest
    return count_first + count_occurrences(arr[1:], target)
        """)
        
        # Demonstrate
        test_arr = ['a', 'b', 'a', 'c', 'a']
        target = 'a'
        
        def count_occurrences(arr, target):
            if len(arr) == 0:
                return 0
            count_first = 1 if arr[0] == target else 0
            return count_first + count_occurrences(arr[1:], target)
        
        result = count_occurrences(test_arr, target)
        print(f"\nResult: '{target}' appears {result} times in {test_arr}")
        
        if 'first' in approach.lower() or 'element' in approach.lower():
            print("Good thinking! You understand the recursive breakdown.")
            return 1
        else:
            print("The key is to handle the first element, then recurse on the rest.")
            return 0

class Level3_TailRecursion:
    """
    Advanced optimization technique focusing on tail-recursive implementations.
    Teaches how to convert regular recursion to tail recursion for better performance.
    """
    
    def __init__(self, game_controller):
        self.game = game_controller
        self.level_name = "Tail Recursion Optimization"
        
    def teach_tail_recursion_concept(self):
        """Comprehensive explanation of tail recursion and its benefits."""
        print(f"\nLEVEL 3: {self.level_name}")
        print("-" * 60)
        print("CORE CONCEPT: Tail recursion is a special form where the recursive")
        print("call is the last operation in the function. This allows for")
        print("optimization that prevents stack overflow.")
        
        print("\nREGULAR vs TAIL RECURSION:")
        print("=" * 40)
        
        print("REGULAR RECURSION (not tail recursive):")
        print("""
def factorial_regular(n):
    if n <= 1:
        return 1
    return n * factorial_regular(n - 1)  # Multiplication AFTER recursive call
        """)
        
        print("TAIL RECURSION (optimizable):")
        print("""
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)  # Recursive call is LAST
        """)
        
        print("\nWHY TAIL RECURSION MATTERS:")
        print("- Regular recursion: Must remember to multiply after each call returns")
        print("- Tail recursion: All work done before the recursive call")
        print("- Compiler can optimize tail recursion to use constant stack space")
        
        self.demonstrate_stack_difference()
        
        return self.tail_recursion_challenges()
    
    def demonstrate_stack_difference(self):
        """Show the difference in stack usage between regular and tail recursion."""
        print("\nSTACK USAGE COMPARISON:")
        print("=" * 30)
        
        n = 4
        
        print(f"REGULAR RECURSION - factorial_regular({n}):")
        print("CALL STACK GROWS:")
        for i in range(n, 0, -1):
            indent = "  " * (n - i)
            print(f"{indent}factorial_regular({i}) -> needs to remember to multiply by {i}")
        
        print("STACK UNWINDS WITH CALCULATIONS:")
        result = 1
        for i in range(1, n + 1):
            result *= i
            indent = "  " * (n - i)
            print(f"{indent}returns {result}")
        
        print(f"\nTAIL RECURSION - factorial_tail({n}, 1):")
        print("CALL STACK WITH ACCUMULATOR:")
        acc = 1
        for i in range(n, 0, -1):
            print(f"factorial_tail({i}, {acc})")
            acc *= i
            if i > 1:
                print(f"  -> factorial_tail({i-1}, {acc})")
        print(f"  -> returns {acc}")
        
        print("\nKEY INSIGHT: Tail recursion carries the result forward,")
        print("while regular recursion builds up work to do on the way back.")
    
    def tail_recursion_challenges(self):
        """Challenges to convert regular recursion to tail recursion."""
        print("\nTAIL RECURSION CONVERSION CHALLENGES")
        print("=" * 45)
        
        challenges = [
            self.challenge_sum_tail(),
            self.challenge_fibonacci_tail(),
            self.challenge_reverse_tail()
        ]
        
        score = sum(challenges)
        total = len(challenges)
        
        print(f"\nLEVEL 3 COMPLETE! Score: {score}/{total}")
        
        if score >= total * 0.6:
            self.game.skills_mastered.add('tail_recursion')
            return True
        else:
            print("Practice converting regular recursion to tail recursion.")
            return False
    
    def challenge_sum_tail(self):
        """Convert array sum to tail recursive form."""
        print("\nCHALLENGE 1: Convert array sum to tail recursion")
        print("-" * 50)
        
        print("REGULAR RECURSIVE SUM:")
        print("""
def sum_regular(arr):
    if len(arr) == 0:
        return 0
    return arr[0] + sum_regular(arr[1:])  # Addition AFTER recursive call
        """)
        
        print("YOUR TASK: Convert to tail recursion using an accumulator")
        print("def sum_tail(arr, accumulator=0):")
        
        base_case = input("What should the base case return? ").strip()
        recursive_call = input("What's the tail recursive call? (format: sum_tail(?, ?)): ").strip()
        
        print("\nCORRECT TAIL RECURSIVE VERSION:")
        print("""
def sum_tail(arr, accumulator=0):
    if len(arr) == 0:
        return accumulator  # Return accumulated result
    return sum_tail(arr[1:], accumulator + arr[0])  # Add to accumulator
        """)
        
        # Test understanding
        if 'accumulator' in base_case.lower() and 'arr[1:]' in recursive_call:
            print("Excellent! You understand the accumulator pattern.")
            
            # Demonstrate execution
            test_arr = [1, 2, 3, 4]
            print(f"\nTRACING sum_tail({test_arr}, 0):")
            self.trace_tail_sum(test_arr, 0)
            return 1
        else:
            print("Study the pattern: accumulator holds the running result.")
            return 0
    
    def trace_tail_sum(self, arr, acc):
        """Trace tail recursive sum execution."""
        original_arr = arr[:]
        step = 0
        
        print(f"Step {step}: sum_tail({arr}, {acc})")
        
        while arr:
            step += 1
            acc += arr[0]
            arr = arr[1:]
            print(f"Step {step}: sum_tail({arr}, {acc})")
        
        print(f"Base case reached: return {acc}")
        print(f"Final result: sum of {original_arr} = {acc}")
    
    def challenge_fibonacci_tail(self):
        """Convert Fibonacci to tail recursive form."""
        print("\nCHALLENGE 2: Tail recursive Fibonacci")
        print("-" * 40)
        
        print("REGULAR FIBONACCI (very inefficient):")
        print("""
def fib_regular(n):
    if n <= 1:
        return n
    return fib_regular(n-1) + fib_regular(n-2)  # Two recursive calls!
        """)
        
        print("HINT: Use two accumulators to track the last two Fibonacci numbers")
        
        approach = input("Describe your tail recursive approach: ").strip()
        
        print("\nOPTIMAL TAIL RECURSIVE FIBONACCI:")
        print("""
def fib_tail(n, a=0, b=1):
    if n == 0:
        return a
    if n == 1:
        return b
    return fib_tail(n-1, b, a+b)  # Shift the window: (a,b) -> (b, a+b)
        """)
        
        # Demonstrate the efficiency gain
        print("\nEFFICIENCY COMPARISON:")
        print("Regular fib(10) makes 177 function calls")
        print("Tail fib(10) makes 10 function calls")
        
        # Show execution trace
        n = 6
        print(f"\nTRACING fib_tail({n}, 0, 1):")
        self.trace_fib_tail(n, 0, 1)
        
        if 'two' in approach.lower() or 'shift' in approach.lower():
            print("Great insight! You see the sliding window pattern.")
            return 1
        else:
            print("The key is maintaining the last two values as you count down.")
            return 0
    
    def trace_fib_tail(self, n, a, b):
        """Trace tail recursive Fibonacci."""
        step = 0
        print(f"Step {step}: fib_tail({n}, {a}, {b})")
        
        while n > 1:
            step += 1
            n -= 1
            a, b = b, a + b
            print(f"Step {step}: fib_tail({n}, {a}, {b})")
        
        result = b if n == 1 else a
        print(f"Base case: return {result}")
        return result
    
    def challenge_reverse_tail(self):
        """Convert string reversal to tail recursive form."""
        print("\nCHALLENGE 3: Tail recursive string reversal")
        print("-" * 45)
        
        print("REGULAR RECURSIVE REVERSE:")
        print("""
def reverse_regular(s):
    if len(s) <= 1:
        return s
    return s[-1] + reverse_regular(s[:-1])  # Concatenation AFTER call
        """)
        
        solution = input("Write tail recursive version with accumulator: ").strip()
        
        print("\nTAIL RECURSIVE VERSION:")
        print("""
def reverse_tail(s, accumulator=""):
    if len(s) == 0:
        return accumulator
    return reverse_tail(s[1:], s[0] + accumulator)  # Build result forward
        """)
        
        print("\nKEY INSIGHT: Instead of building the result on the way back,
