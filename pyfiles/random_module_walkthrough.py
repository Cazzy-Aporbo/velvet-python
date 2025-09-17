# random_module_walkthrough.py
# -----------------------------------------------------------------------------
# Title: Python Random Module — From Beginner to Expert
# Author: Cazandra Aporbo
# Started: February 2023
# Last Updated: August 2025
# Intent: Teach the random module step by step — from basics like randint() to
#         advanced statistical distributions. Every line is explained in a human
#         way, as though I'm walking you through it live.
# Promise: I do my best at explaining randomness in Python
#          the way I wish someone had explained it to me.
# -----------------------------------------------------------------------------

import random

def block(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)

# -----------------------------------------------------------------------------
# SECTION 1 — Core Randomness (seed, state, bits)
# -----------------------------------------------------------------------------

def lesson_random_core():
    block("1) Core Randomness: seed, state, and bits")

    # seed(): lock randomness to make results reproducible
    random.seed(42)  # everyone who runs this gets the same sequence
    print("random.seed(42) locks in the sequence")
    print("random.randint(1,10) ->", random.randint(1,10))

    # getstate() and setstate(): capture and restore the generator state
    state = random.getstate()
    print("State captured ->", type(state))
    print("Next random after capturing state:", random.randint(1,10))
    # restore
    random.setstate(state)
    print("Restored state, same number again:", random.randint(1,10))

    # getrandbits(): generate raw bits
    print("8 random bits as integer:", random.getrandbits(8))

# -----------------------------------------------------------------------------
# SECTION 2 — Everyday Picks (randint, randrange, choice, choices, shuffle, sample)
# -----------------------------------------------------------------------------

def lesson_random_everyday():
    block("2) Everyday Picks: random integers and selections")

    # randrange(): like range() but random pick
    print("random.randrange(1,10,2) ->", random.randrange(1,10,2))

    # randint(): inclusive ends
    print("random.randint(1,100) ->", random.randint(1,100))

    # choice(): one item from a sequence
    fruits = ['apple','banana','cherry']
    print("random.choice(fruits) ->", random.choice(fruits))

    # choices(): sample with replacement, multiple picks
    colors = ['red','green','blue']
    print("random.choices(colors, k=3) ->", random.choices(colors, k=3))

    # shuffle(): reorder a list in place
    nums = [1,2,3,4,5]
    random.shuffle(nums)
    print("After shuffle ->", nums)

    # sample(): pick unique subset
    letters = ['a','b','c','d','e']
    print("random.sample(letters,3) ->", random.sample(letters,3))

# -----------------------------------------------------------------------------
# SECTION 3 — Floats and Continuous Distributions
# -----------------------------------------------------------------------------

def lesson_random_floats():
    block("3) Continuous Randomness: floats and distributions")

    # random(): float in [0,1)
    print("random.random() ->", random.random())

    # uniform(): float in [a,b]
    print("random.uniform(2.0,5.0) ->", random.uniform(2.0,5.0))

    # triangular(): skewed toward midpoint
    print("random.triangular(1,5,3) ->", random.triangular(1,5,3))

# -----------------------------------------------------------------------------
# SECTION 4 — Advanced Distributions (statistical modeling)
# -----------------------------------------------------------------------------

def lesson_random_distributions():
    block("4) Advanced Distributions: modeling with math")

    print("Beta distribution (alpha=2,beta=5):", random.betavariate(2,5))
    print("Exponential distribution (lambda=0.5):", random.expovariate(0.5))
    print("Gamma distribution (alpha=2,beta=3):", random.gammavariate(2,3))
    print("Gaussian (mean=0, sigma=1):", random.gauss(0,1))
    print("Log-normal (mu=0, sigma=1):", random.lognormvariate(0,1))
    print("Normal distribution (mean=0, sigma=1):", random.normalvariate(0,1))
    print("Von Mises (mu=0, kappa=1):", random.vonmisesvariate(0,1))
    print("Pareto (alpha=2.5):", random.paretovariate(2.5))
    print("Weibull (alpha=2, beta=3):", random.weibullvariate(2,3))

# -----------------------------------------------------------------------------
# Master runner
# -----------------------------------------------------------------------------

def run_all_random_lessons():
    lesson_random_core()
    lesson_random_everyday()
    lesson_random_floats()
    lesson_random_distributions()

if __name__ == "__main__":
    run_all_random_lessons()
