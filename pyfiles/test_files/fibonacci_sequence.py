def fibonacci_sequence(n):
    fib_list = [0, 1]  
    for i in range(2, n):
        fib_list.append(fib_list[i-1] + fib_list[i-2])
    return fib_list

def calculate_ratio(numbers):
    ratio_list = []
    for i in range(len(numbers) - 1, len(numbers) - 6, -1):
        ratio = numbers[i] / numbers[i-1]
        ratio_list.append(ratio)
    return ratio_list

fibonacci_numbers = fibonacci_sequence(12)
print("Fibonacci Numbers:")
print(fibonacci_numbers)

ratio_list = calculate_ratio(fibonacci_numbers)
print("\nRatios of the Last Five Fibonacci Numbers:")
print(ratio_list)

rounded_ratios = [round(ratio, 6) for ratio in ratio_list]
print("\nRounded Ratios (to 6 significant figures):")
print(rounded_ratios)


# Observations
print("\n Here are some bservations:")
print("The ratio list tends to converge towards 1.61803398875.")

