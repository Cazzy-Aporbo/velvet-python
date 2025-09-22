import math
import matplotlib.pyplot as plt
import numpy as np

def derive_quadratic_formula(a, b, c):
    """
    Step-by-step derivation of quadratic formula using completing the square.
    """
    print("\n=== Step-by-Step Derivation ===")
    print(f"Start with: {a}x² + {b}x + {c} = 0")
    
    # Divide by a
    print(f"Divide both sides by {a} to normalize:")
    print(f"x² + ({b}/{a})x + ({c}/{a}) = 0")
    
    # Move constant to other side
    print(f"Move constant term to the other side:")
    print(f"x² + ({b}/{a})x = {-c/a}")
    
    # Complete the square
    print("Complete the square:")
    half_b = b / (2*a)
    print(f"Add ({half_b})² to both sides")
    print(f"x² + ({b}/{a})x + ({half_b})² = {-c/a} + ({half_b})²")
    
    # Factor left side
    print(f"Factor left side: (x + {half_b})² = {half_b**2 - c/a}")
    
    # Solve for x
    discriminant = b**2 - 4*a*c
    sqrt_disc = math.sqrt(abs(discriminant))
    print(f"Take square root of both sides: x + {half_b} = ±√({discriminant}) / (2a)")
    
    x1 = (-b + sqrt_disc)/(2*a)
    x2 = (-b - sqrt_disc)/(2*a)
    
    print(f"Final roots: x = ({-b} ± √({discriminant}))/({2*a})")
    print(f"x₁ = {x1}, x₂ = {x2}\n")
    
    return x1, x2

def plot_quadratic(a, b, c, x1, x2):
    """
    Visualize the quadratic and its roots.
    """
    x = np.linspace(min(x1, x2)-2, max(x1, x2)+2, 400)
    y = a*x**2 + b*x + c
    plt.figure(figsize=(8,6))
    plt.plot(x, y, label=f"{a}x² + {b}x + {c}")
    plt.axhline(0, color='black')
    plt.scatter([x1, x2], [0, 0], color='red', zorder=5)
    plt.title("QuadLab: Quadratic Roots and Parabola")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main ---
if __name__ == "__main__":
    print("=== QuadLab: Step-by-Step Quadratic Proofs ===")
    a = float(input("Enter coefficient a: "))
    b = float(input("Enter coefficient b: "))
    c = float(input("Enter coefficient c: "))
    
    x1, x2 = derive_quadratic_formula(a, b, c)
    plot_quadratic(a, b, c, x1, x2)