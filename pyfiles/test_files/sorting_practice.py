import random
from time import time

# Define the sorting functions
def mergeSort(L):
    if len(L) < 2:
        return L[:]
    else:
        middle = len(L) // 2
        left = mergeSort(L[:middle])
        right = mergeSort(L[middle:])
        return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    while i < len(left):
        result.append(left[i])
        i += 1
    while j < len(right):
        result.append(right[j])
        j += 1
    return result

def insertionSort(L):
    for i in range(1, len(L)):
        key = L[i]
        j = i-1
        while j >=0 and key < L[j] :
            L[j+1] = L[j]
            j -= 1
        L[j+1] = key
    return L

def bubbleSort(L):
    n = len(L)
    for i in range(n):
        for j in range(0, n-i-1):
            if L[j] > L[j+1] :
                L[j], L[j+1] = L[j+1], L[j]
    return L

# Create a list of n values from 100 to 5000 in 100 increments
n_values = list(range(100, 5100, 100))

# Dictionary to store execution times for each sorting algorithm
times = {'Merge': [], 'Insert': [], 'Bubble': []}

# Print the header
print("N\tMerge\tInsert\tBubble")

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
    C = A.copy()  # Create a copy of the original shuffled list
    t1 = time()
    insertionSort(C)
    t2 = time()
    itime = (t2 - t1) * 1000
    times['Insert'].append(itime)

    # Measure execution time for bubbleSort
    D = A.copy()  # Create a copy of the original shuffled list
    t1 = time()
    bubbleSort(D)
    t2 = time()
    btime = (t2 - t1) * 1000
    times['Bubble'].append(btime)

# Print the results
for i in range(len(n_values)):
    print(f"{n_values[i]}\t{round(times['Merge'][i], 1)}\t{round(times['Insert'][i], 1)}\t{round(times['Bubble'][i], 1)}")


