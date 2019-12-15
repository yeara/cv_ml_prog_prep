# Useful Reading

https://orrsella.com/2016/05/14/preparing-for-a-facebook-google-software-engineer-interview/

https://medium.com/leetcode-patterns

https://medium.com/algorithms-and-leetcode/want-to-crack-leetcode-problems-easily-dc825e27e423

https://app.codility.com/programmers/lessons/16-greedy_algorithms/

https://github.com/orrsella/soft-eng-interview-prep

https://orrsella.com/2016/05/14/preparing-for-a-facebook-google-software-engineer-interview/

# A Systematic Approach to Leetcode

## Go to Problem Solutions

- Problems with counting the frequency of the words or characters, use a lookup table or a hash map.
- Problems of finding a substring or subarray with specific properties: two pointers and a sliding window.
- Sorted array: binary search
- Pair verification problems: stack

##Dynamic Programming

The idea is to find some optimal substructure, and identify overlapping subproblems. 


### Example Tabuation Problem - Longest Common Substring

For this type of problems, it is common to maintain a scoring array and use it to build up the solution. 
This technique avoids multiple lookups and function calls, less memory. 

Examples: longest increasing subsequence, compute the end of the longest sequence ending at that index.


```
Pseudo-code of LCS
init a[M+1][N+1] two dimensional matrix with [0]
for i in [0,M):
    for j in [0,N+1):
        if A[i]==A[j]:
            a[i+1][j+1]=a[i][j]+1
            result = max(result,a[i+1][j+1])        
        else:
            a[i+1][j+1] = 0
```

Perform exhaustive search, and memoize the search results in a "square" table.
Fill in the table in a gradual manner.

This gives a complexity of O(NM) vs. exhaustive search O((NM)^2)


### Memoization Recipe (Top Down)
1. Check if the the table/memory look up contains the solution
2. If it's a valid solution, return the value
3. If it's a base case, update and return
4. Else split up the problem into sub problems
5. Make the recursive calls
6. Compute and store the current value
7. Return the value

Sometime avoid computing solutions to subproblems that are not needed, i.e. Common Subsequence. 

Can be more intuitive for matrix chain multiplications.


##Backtracking

The idea is to exhaustively search all combinations and discard any paths that do not lead to a solution.

Important numbers: 2^n combination for true/false, choose/don't choose - true for sets of unique elements.

Iterative solution can be modeled as a iteration over bit turn on / off: counter & 1 << j 

Backtracking can use a stack like structure to add an element, exhaustively search that space, and then remove the element. 

### When Do We Switch from DP to Backtracking? 

If we need to return all solutions -> backtracking

If we need to count all solution -> memoization + dp

## Sliding Window

https://medium.com/leetcode-patterns/leetcode-pattern-2-sliding-windows-for-strings-e19af105316b

The idea is to maintain counters for the goal, including the start and end of the current window. 
Use a separate indicator for restrictions/search window criteria, such as number of distinct characters.

Update the window indices as soon as a better solution is found, or current window fails criteria.

## DFS + BFS

DFS - Use a stack.

BFS - Use a queue. 

DFS - Exhaustive search, find all combinations/paths

BFS - Shortest path. For graphs where all edges are the same weight -> shortest path is equiv. to Dijkstra

Imp. tree level printing

# Take Home Messages from Solutions

Hashing - use a bit array and turning bits on and off.

For small digit/chat count - use array counters

For list traversal

### Longest Inc. Sequence

Here the hard part was to formulate the decisions on extending the sequence. 

We need to do three things: 

- if the new element is smaller than all end elements (and end elements also include a list of size 1 for example), add it to the active lists.
- if the new element is smaller than some of the end elements, replace them
- if the new element is larger than all end elements, add it

A very streamlined solution was to store this all in a vector, and user lower_bound to replace the correct element.

### Word Combination in Dictionary

The first hint was that the complexity of n^2 - should think about two loops

Now the difficulty for me was how to cache results. 

The best solution on leetcode was to start from i=0, and grow the test word from j=i-1, to j=0 and test if all positions where a word was marked as ending there. 

The results are stored in a vector<bool>

### Longest Substring w/o Repeating Character

Here the important insight is to cache the occurrence of the letter in the string, this way easy to remove and start a new sliding window or track the start location of the current window

##Stacks

### Queue from two stacks

Using two queues for  a stack: the idea is to use one for the in end, and then reverse it into another stack for the out end

The top() operation is used to amortize the reversal cost.

### Mid Element in Array 

Again, advancing two pointers, at different rate.

Or advance one, and then the other. Technically i think it's the same complexity. 

### Cycle in Array

Here the ideas is to advance two pointers in a different rate, if there is a cycle at the vector they will meet.

## Graphs

### Counting Connected Components

The most efficient solution uses the map as labelling, and deleted an entire island at the time. 

The idea was to iterate over the entire grid AND delete the islands one component as a time.

This is technically BFS.

### Cycle / Duplicate Detection

With two pointers - easy. one moves at x1, the other moves at x2, if they meet we have a cycle. 

how how do we detect the cycle entry?

define $L_1$ is the dist between list/graph start and entry

$L_2$ is the dist between entry and meeting point

slow pointer does $L_1 + L_2$

fast pointer does $2L_1 + 2L_2 + nC $ 

Therefor we know that: 

$$  2L_1 + 2L_2 = L_1 + L_2 + (n)C$$ 

$$  L_1 = (C-L_2) + (n-1)C$$

the distance between the start to the entry to the cycle is equal to the distance between meeting point is equal to the distance between entrance to graph and entrance to cycle

### Snakes and Ladders

Getting indexing correctly is the worsest. 

Remember to return -1; 

Remember to use early exit for Dijikstra

Converting linear indices to tile numbers:

r = n - (li-1) / n - 1;
c = (li-1) % n;
if((n-r-1)%2) c = n - c - 1;

## DP

### 8 Queens

Again, indexing and index tests are a bitch - esp the diagonals (check all diagonals!)

Use the row as a way to keep track of what was already searched

### Count Digits

The task was to count all of the 1's in all of the numbers between 0 and i. 

Here it was pretty simple in the sense that you compute your current power and return the pointer to the first cell in the array where you already stored the computation every time you reach a new power.

### Coin Change

Sorting the coins ensures that we test against the largest coins first.

DP - To do the caching, use an array with the amount [0-amount] and store the minimal amount of coins for this value.


## Misc / General

### Power Set

Power set size is 2^n - iterate over all combinations using an int and choose elements using bit shifts tests.

Testing with bit shift: counter & 1 << j

A neat solution was growing the power set by iterating over previous sets until the correct cardinality for each pass (copy and mult)

### Permutation Set

Cardinality of permutation set - n!

### Unique Combination Sum

Backtracking - test conditions and commit solution when entering the function

Start the search window from the current index, or next, depending on the constraints

# Interesting Problems

https://algorithms.tutorialhorizon.com/colorful-numbers/
