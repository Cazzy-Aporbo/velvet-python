import random
from time import time
# test the running times of three sort implementations: mergeSort, insertionSort, and bubbleSort


#take in a single list of integers as a sole parameter
def mergeSort(L):
    if len(L) < 2:
        # test if statement
        #print("in if base-case reached")
        return L[:]
    else:
        # test recursion
        #print("in else recursive case")
        middle = len(L) // 2
        #print(L) #test
        left = mergeSort(L[:middle])
        right = mergeSort(L[middle:])
        return merge(left, right)

#mergeSort([8, 7, 1, 4, 6, 3, 5, 2]) # test code calling mergeSort
# test code

n = 12
test = [i for i in range(n)]
random.shuffle(test)
print("Original unsorted test list:", test)

# Call mergeSort 
sorted_list = mergeSort(test)
print("Sorted test list:", sorted_list)


def merge(left, right):
    # initializes an empty list called `result`. This list will be used to store the merged
    result = []
    #initialize 2 variables
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



# Pass this list to each sort, and make sure the results come back sorted.
def insertionSort(L):
    for i in range(1, len(L)):
        key = L[i]
        j = i-1
        #print(f"{L=}") #test for-loop for list
        while j >=0 and key < L[j] :
            L[j+1] = L[j]
            j -= 1
        L[j+1] = key
    return L



# On a single bubble pass, you run through all the elements in your list from front to back. 
# At each index you compare its value to the value of the next one
# https://www.geeksforgeeks.org/python-program-for-bubble-sort/
def bubbleSort(L):
    n = len(L)
    for i in range(n):
        #print("test: ", L) # test
        for j in range(0, n-i-1):
            if L[j] > L[j+1] :
                #test for-loop
                #print("inside if")
                L[j], L[j+1] = L[j+1], L[j]
    return L




n = 8
test = [i for i in range(1, n + 1)]  # Create a list of n numbers
random.shuffle(test)
print("Original unsorted test list:", test)

# call test for bubbleSort
bubbleSort(A)



# Pass list to each sort, and make sure the results come back sorted
n = 10  # good testing length
A = [i for i in range(n)]

# test sort 
random.shuffle(A)
print("Original list:", A)

B = mergeSort(A.copy())
print("Merge list sorted:", B)

random.shuffle(A)
C = insertionSort(A.copy())
print("Insertion list sorted:", C)

random.shuffle(A)
D = bubbleSort(A.copy())
print("Bubble list sorted:", D)




# Test the sorting functions using time test
# Note: column names are: "N, Merge, Insert, Bubble"

#given: time all three sorts with n values from 100 to 5000, in 100 increments. 
n_values = list(range(100, 5000, 100))
#container to store time executed
times = {'Merge': [], 'Insert': [], 'Bubble': []}
for n in n_values:
    A = [i for i in range(n)]
    random.shuffle(A)
# To time a sort, call the time() function (from time import time) right before and 
#right after the sort call. Then subtract the values and multiply by 1000 
#to get the results in milliseconds
    t1 = time()
    B = mergeSort(A)
    t2 = time()
    mtime = (t2-t1)*1000 # as instructed
    times['Merge'].append(mtime)

    random.shuffle(A)
    t1 = time()
    B = insertionSort(A)
    t2 = time()
    itime = (t2-t1)*1000
    times['Insert'].append(itime)

    random.shuffle(A)
    t1 = time()
    B = bubbleSort(A)
    t2 = time()
    btime = (t2-t1)*1000
    times['Bubble'].append(btime)





print('N\tMerge\tInsert\tBubble')

for i in range(len(n_values)):
    print(f'{n_values[i]}\t{round(times["Merge"][i], 1)}\t{round(times["Insert"][i], 1)}\t{round(times["Bubble"][i], 1)}')

