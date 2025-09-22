"""
🎮 BACKPROPAGATION ADVENTURE GAME 🎮
Learn Neural Networks and Backpropagation Step by Step!
Purpose: Interactive learning experience for backpropagation
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple, Dict

class BackpropagationGame:
    def __init__(self):
        self.player_level = 1
        self.player_xp = 0
        self.player_name = ""
        self.current_lesson = 1
        self.max_lessons = 8
        
    def welcome_screen(self):
        """Welcome the player to the backpropagation adventure!"""
        print("=" * 60)
        print("🎮 WELCOME TO BACKPROPAGATION ADVENTURE! 🎮")
        print("=" * 60)
        print("🧠 Learn how neural networks learn through backpropagation!")
        print("🎯 Complete challenges to level up your understanding!")
        print("📈 Progress from beginner to advanced concepts!")
        print("=" * 60)
        
        self.player_name = input("Enter your adventurer name: ")
        print(f"\nWelcome, {self.player_name}! Let's begin your journey! 🚀\n")
        
    def show_progress(self):
        """Display current progress"""
        print(f"👤 Player: {self.player_name}")
        print(f"⭐ Level: {self.player_level}")
        print(f"🎯 XP: {self.player_xp}")
        print(f"📚 Lesson: {self.current_lesson}/{self.max_lessons}")
        print("-" * 40)

# LEVEL 1: BASIC CONCEPTS
class Level1_BasicConcepts:
    """Introduction to Neural Networks and Forward Pass"""
    
    def __init__(self):
        self.title = "🌟 LEVEL 1: Understanding the Basics"
        
    def teach_neurons(self):
        """Teach what a neuron is"""
        print(self.title)
        print("=" * 50)
        print("🧠 LESSON 1.1: What is a Neuron?")
        print("-" * 30)
        print("A neuron is like a tiny decision maker!")
        print("It takes inputs, processes them, and gives an output.")
        print("\nThink of it like a recipe:")
        print("📥 Inputs: ingredients (x1, x2, x3...)")
        print("⚖️  Weights: how much of each ingredient (w1, w2, w3...)")
        print("🔢 Bias: secret sauce (b)")
        print("📤 Output: final dish (y)")
        print("\nFormula: output = activation(w1*x1 + w2*x2 + ... + b)")
        
        input("\nPress Enter to see this in action...")
        
        # Interactive example
        print("\n🎮 INTERACTIVE EXAMPLE:")
        print("Let's make a simple neuron that decides if you should go outside!")
        
        # Get user inputs
        temperature = float(input("What's the temperature? (0-40°C): "))
        rain_chance = float(input("Rain chance? (0-100%): "))
        
        # Simple neuron
        w1, w2 = 0.1, -0.02  # weights
        bias = -2
        
        decision_score = w1 * temperature + w2 * rain_chance + bias
        
        print(f"\n🧮 Calculation:")
        print(f"Score = {w1} × {temperature} + {w2} × {rain_chance} + {bias}")
        print(f"Score = {decision_score:.2f}")
        
        if decision_score > 0:
            print("🌞 Decision: GO OUTSIDE! ✅")
        else:
            print("🏠 Decision: STAY INSIDE! ❌")
            
        return True

    def teach_forward_pass(self):
        """Teach forward propagation"""
        print("\n🧠 LESSON 1.2: Forward Pass - The Journey Forward")
        print("-" * 50)
        print("Forward pass is like following a recipe step by step:")
        print("1. Take inputs")
        print("2. Multiply by weights")
        print("3. Add bias")
        print("4. Apply activation function")
        print("5. Pass to next layer")
        
        input("\nPress Enter to code a simple forward pass...")
        
        print("\n💻 CODE EXAMPLE:")
        print("
")
        
        # Let them try it
        print("\n🎮 YOUR TURN!")
        inputs = [1.0, 0.5, -0.3]
        weights = [0.2, -0.1, 0.4]
        bias = 0.1
        
        print(f"Inputs: {inputs}")
        print(f"Weights: {weights}")
        print(f"Bias: {bias}")
        
        # Calculate step by step
        weighted_sum = sum(i * w for i, w in zip(inputs, weights))
        print(f"\nStep 1 - Weighted sum: {weighted_sum:.3f}")
        
        weighted_sum += bias
        print(f"Step 2 - Add bias: {weighted_sum:.3f}")
        
        output = 1 / (1 + np.exp(-weighted_sum))
        print(f"Step 3 - Sigmoid activation: {output:.3f}")
        
        return True

# LEVEL 2: THE PROBLEM
class Level2_TheProblem:
    """Understanding why we need backpropagation"""
    
    def __init__(self):
        self.title = "🎯 LEVEL 2: The Learning Problem"
        
    def demonstrate_problem(self):
        """Show why random weights don't work"""
        print(self.title)
        print("=" * 50)
        print("🤔 LESSON 2.1: Why Do We Need to Learn?")
        print("-" * 40)
        
        print("Imagine you're teaching a robot to recognize cats vs dogs...")
        print("With random weights, it's just guessing! 🎲")
        
        input("\nPress Enter to see the problem...")
        
        # Simulate random predictions
        print("\n🤖 ROBOT WITH RANDOM WEIGHTS:")
        animals = ["Cat", "Dog", "Cat", "Dog", "Cat"]
        correct = [1, 0, 1, 0, 1]  # 1 for cat, 0 for dog
        
        print("Animal    | Correct | Robot Says | Result")
        print("-" * 45)
        
        correct_count = 0
        for i, (animal, target) in enumerate(zip(animals, correct)):
            # Random prediction
            prediction = random.choice([0, 1])
            result = "✅" if prediction == target else "❌"
            robot_says = "Cat" if prediction == 1 else "Dog"
            correct_answer = "Cat" if target == 1 else "Dog"
            
            if prediction == target:
                correct_count += 1
                
            print(f"{animal:8} | {correct_answer:7} | {robot_says:8} | {result}")
        
        accuracy = correct_count / len(animals) * 100
        print(f"\nAccuracy: {accuracy:.1f}% 😱")
        print("This is terrible! We need the robot to LEARN!")
        
        return True
    
    def introduce_loss(self):
        """Introduce the concept of loss/error"""
        print("\n🎯 LESSON 2.2: Measuring Mistakes (Loss Function)")
        print("-" * 50)
        
        print("To improve, we need to measure how wrong we are!")
        print("This is called a LOSS FUNCTION or ERROR FUNCTION")
        print("\n📏 Common loss functions:")
        print("• Mean Squared Error (MSE): (predicted - actual)²")
        print("• Cross-entropy: for classification problems")
        
        input("\nPress Enter to calculate some losses...")
        
        print("\n🧮 CALCULATING LOSS:")
        predictions = [0.8, 0.3, 0.9, 0.1, 0.7]
        targets = [1.0, 0.0, 1.0, 0.0, 1.0]
        
        print("Prediction | Target | Error | Squared Error")
        print("-" * 45)
        
        total_loss = 0
        for pred, target in zip(predictions, targets):
            error = pred - target
            squared_error = error ** 2
            total_loss += squared_error
            print(f"{pred:8.1f} | {target:4.1f} | {error:5.1f} | {squared_error:11.3f}")
        
        mse = total_loss / len(predictions)
        print(f"\nMean Squared Error: {mse:.3f}")
        print("💡 Lower loss = better performance!")
        print("🎯 Goal: Adjust weights to minimize loss!")
        
        return True

# LEVEL 3: DERIVATIVES AND GRADIENTS
class Level3_Derivatives:
    """Understanding derivatives for optimization"""
    
    def __init__(self):
        self.title = "📈 LEVEL 3: The Magic of Derivatives"
        
    def teach_derivatives_intuition(self):
        """Teach derivatives with intuition"""
        print(self.title)
        print("=" * 50)
        print("🎢 LESSON 3.1: Derivatives - Finding the Slope")
        print("-" * 45)
        
        print("Imagine you're on a hill and want to find the bottom...")
        print("🏔️  The derivative tells you:")
        print("   • Which direction is downhill (+ or -)")
        print("   • How steep the hill is (magnitude)")
        print("\n🎯 In neural networks:")
        print("   • Hill = Loss function")
        print("   • Bottom = Minimum loss (best weights)")
        print("   • Derivative = Which way to adjust weights")
        
        input("\nPress Enter to see this visually...")
        
        # Create a simple visualization
        x = np.linspace(-3, 3, 100)
        y = x**2  # Simple parabola
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=2, label='Loss = x²')
        
        # Show derivative at different points
        points = [-2, -1, 0, 1, 2]
        for point in points:
            derivative = 2 * point  # derivative of x² is 2x
            plt.arrow(point, point**2, -derivative*0.3, 0, 
                     head_width=0.2, head_length=0.1, fc='red', ec='red')
            plt.plot(point, point**2, 'ro', markersize=8)
            plt.text(point, point**2 + 1, f'slope={derivative}', 
                    ha='center', fontsize=10)
        
        plt.xlabel('Weight Value')
        plt.ylabel('Loss')
        plt.title('🎢 Derivatives Show Us Which Way to Go!')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
        print("🔍 Notice:")
        print("• Negative slope → move right (increase weight)")
        print("• Positive slope → move left (decrease weight)")
        print("• Zero slope → we're at the minimum! 🎯")
        
        return True
    
    def teach_chain_rule(self):
        """Teach the chain rule"""
        print("\n⛓️  LESSON 3.2: The Chain Rule - Connecting the Dots")
        print("-" * 50)
        
        print("The chain rule helps us find derivatives of nested functions!")
        print("Think of it like a chain of cause and effect...")
        print("\n🔗 Example: How does changing weight w affect final loss L?")
        print("w → z → a → L")
        print("weight → weighted_sum → activation → loss")
        print("\nChain rule: dL/dw = (dL/da) × (da/dz) × (dz/dw)")
        
        input("\nPress Enter for a concrete example...")
        
        print("\n🧮 CONCRETE EXAMPLE:")
        print("Let's say we have:")
        print("z = w × x + b    (weighted sum)")
        print("a = sigmoid(z)   (activation)")
        print("L = (a - y)²     (loss)")
        
        # Example values
        w, x, b, y = 0.5, 2.0, 0.1, 1.0
        
        print(f"\nWith w={w}, x={x}, b={b}, target y={y}:")
        
        # Forward pass
        z = w * x + b
        a = 1 / (1 + np.exp(-z))  # sigmoid
        L = (a - y) ** 2
        
        print(f"z = {w} × {x} + {b} = {z}")
        print(f"a = sigmoid({z}) = {a:.3f}")
        print(f"L = ({a:.3f} - {y})² = {L:.3f}")
        
        # Backward pass (derivatives)
        print(f"\n⬅️  BACKWARD PASS (Chain Rule):")
        dL_da = 2 * (a - y)
        da_dz = a * (1 - a)  # sigmoid derivative
        dz_dw = x
        
        print(f"dL/da = 2(a - y) = 2({a:.3f} - {y}) = {dL_da:.3f}")
        print(f"da/dz = a(1-a) = {a:.3f}(1-{a:.3f}) = {da_dz:.3f}")
        print(f"dz/dw = x = {dz_dw}")
        
        dL_dw = dL_da * da_dz * dz_dw
        print(f"\n🎯 Final result:")
        print(f"dL/dw = {dL_da:.3f} × {da_dz:.3f} × {dz_dw} = {dL_dw:.3f}")
        print(f"This tells us to {'decrease' if dL_dw > 0 else 'increase'} the weight!")
        
        return True

# LEVEL 4: SIMPLE BACKPROPAGATION
class Level4_SimpleBackprop:
    """Implement backpropagation for a single neuron"""
    
    def __init__(self):
        self.title = "🔄 LEVEL 4: Your First Backpropagation"
        
    def single_neuron_backprop(self):
        """Implement backprop for one neuron"""
        print(self.title)
        print("=" * 50)
        print("🎯 LESSON 4.1: Backpropagation Step by Step")
        print("-" * 45)
        
        print("Let's implement backpropagation for a single neuron!")
        print("We'll train it to learn a simple pattern...")
        
        input("\nPress Enter to start coding...")
        
        print("\n💻 SINGLE NEURON CLASS:")
        print("
