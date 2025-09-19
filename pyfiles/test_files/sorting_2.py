import random
from time import time

# Define the sorting functions (same as before)

# Create a list of n values from 100 to 5000 in 100 increments
n_values = list(range(100, 5100, 100))

# Dictionary to store execution times for each sorting algorithm
times = {'Merge': [], 'Insert': [], 'Bubble': []}

# Print the header
print("N", end="\t")
for n in n_values:
    print(n, end="\t")
print()

# Test each sorting algorithm for different n values
for n in n_values:
    A = [i for i in range(n)]
    random.shuffle(A)

    # Measure execution time for mergeSort
    t1 = time()
    B = mergeSort(A.copy())
    t2 = time()
    mtime = (t2 - t1) * 1000
    times['Merge'].append(mtime)

    # Measure execution time for insertionSort
    random.shuffle(A)
    t1 = time()
    C = insertionSort(A.copy())
    t2 = time()
    itime = (t2 - t1) * 1000
    times['Insert'].append(itime)

    # Measure execution time for bubbleSort
    random.shuffle(A)
    t1 = time()
    D = bubbleSort(A.copy())
    t2 = time()
    btime = (t2 - t1) * 1000
    times['Bubble'].append(btime)

# Print the results
print("Merge", end="\t")
for time_val in times['Merge']:
    print(round(time_val, 1), end="\t")
print()

print("Insert", end="\t")
for time_val in times['Insert']:
    print(round(time_val, 1), end="\t")
print()

print("Bubble", end="\t")
for time_val in times['Bubble']:
    print(round(time_val, 1), end="\t")
print()
