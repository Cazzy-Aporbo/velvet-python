from itertools import combinations

# The set of 10 numbers
numbers = [41, 30, 34, 43, 3, 48, 77, 14, 7, 17]

# Function to find all combinations of a given length from the set
def find_combinations(numbers, length):
    return list(combinations(numbers, length))

# Finding combinations for different lengths
combinations_4 = find_combinations(numbers, 4)
combinations_5 = find_combinations(numbers, 5)
combinations_6 = find_combinations(numbers, 6)
combinations_7 = find_combinations(numbers, 7)
combinations_8 = find_combinations(numbers, 8)
combinations_9 = find_combinations(numbers, 9)
combinations_10 = find_combinations(numbers, 10)

# Function to print a few examples of combinations for each length
def print_combination_examples(combinations_dict, num_examples=15):
    for length, combos in combinations_dict.items():
        print(f"Combinations for {length} numbers (showing first {num_examples}):")
        # Print the first few combinations
        for combo in combos[:num_examples]:
            print(combo)
        print("...")  # Indicating there are more combinations
        print()

# Dictionary of all combinations
all_combinations = {
    4: combinations_4,
    5: combinations_5,
    6: combinations_6,
    7: combinations_7,
    8: combinations_8,
    9: combinations_9,
    10: combinations_10
}

# Print examples from each set
print_combination_examples(all_combinations)
