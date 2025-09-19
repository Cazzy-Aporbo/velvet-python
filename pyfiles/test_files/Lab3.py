import random
import time  # Import the time module

# take a number, p, and return true if it is prime and false otherwise.
def isPrime(p):
    if p <= 1:
        return False
    if p == 2:
        return True
    if p % 2 == 0:
        return False
    for i in range(3, int(p ** 0.5) + 1, 2):
        if p % i == 0:
            return False
    return True

# nBitPrime function to generate a random prime number up to n bits:
def nBitPrime(n):
    while True:
        num = random.randint(2**(n-1), 2**n - 1)
        if isPrime(num):
            return num

P = nBitPrime(20)
Q = nBitPrime(20)
PQ = P * Q

def factor(pq):
    for i in range(2, int(pq ** 0.5) + 1):
        if pq % i == 0:
            P = i
            Q = pq // i
            return P, Q
    return None

# Test
pq = 56
result = factor(pq)
if result is not None:
    P, Q = result
    print(f"P: {P}, Q: {Q}")
else:
    print("No factors found.")

bit_lengths = [15, 16, 17, 18, 19, 20]
bit_length_factorization_times = [] #container

for bit_length in bit_lengths:
    pq = nBitPrime(bit_length) * nBitPrime(bit_length)
    start_time = time.time()
    P, Q = factor(pq)
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    bit_length_factorization_times.append((bit_length, elapsed_time_ms))
    print("Bit Length:", bit_length, "Elapsed Time (ms):", elapsed_time_ms)

from math import exp
# Calculate the estimated time to crack a 1024-bit key
bit_length_1024 = 1024
estimated_time_ms = 2 ** (bit_length_1024 - 15) * 0.08
estimated_time_years = estimated_time_ms / (1000 * 60 * 60 * 24 * 365)  # Convert to years

print("Estimated Time in ms to Crack a 1024-bit Key:", estimated_time_ms)
print("Estimated Time in years to Crack a 1024-bit Key:", estimated_time_years)
