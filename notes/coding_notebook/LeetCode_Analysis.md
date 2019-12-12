# Useful Reading

https://orrsella.com/2016/05/14/preparing-for-a-facebook-google-software-engineer-interview/

https://medium.com/leetcode-patterns

https://medium.com/algorithms-and-leetcode/want-to-crack-leetcode-problems-easily-dc825e27e423

https://app.codility.com/programmers/lessons/16-greedy_algorithms/

https://github.com/orrsella/soft-eng-interview-prep

https://orrsella.com/2016/05/14/preparing-for-a-facebook-google-software-engineer-interview/

# A Systematic Approach to Leetcode

- If there is anything related to counting the frequency of the words or characters then the best data structure would be to use a hash map
- If there is anything related to finding some kind of sub-string then the first data structure that comes to my mind is to have two pointers and try a sliding window approach
- If there is anything related to a sorted array then the first thing that comes to my mind is a binary search
- If there is anything related to closing brackets and opening bracket or any problem having pair verification then we can use stacks

##Dynamic Programming

### Longest Common Substring

```
Pseudo-code of LCS
init a[M+1][N+1] two dimensional matrix with [0]
for i in [0,M):
    for j in [0,N+1):
        if A[i]==A[j]:
            a[i+1][j+1]=a[i][j]+1
            result = max(result,a[i+1][j+1])        else:
            a[i+1][j+1] = 0
```

We have to do exhaustive search, and this is the best way to memoize the search results.

LCS = O(n*m)

Exhaustive Search: O(n*m^2)

### Optimal Substructure 

### Overlapping Subproblems

### Memoization (top down)

check if the the table/memory look up contains the solution

if it's a valid solution, return the value

if it's a base case, update and return

else split up the problem into sub problems

make the recursive calls

compute and store the current value

return the value

Sometime avoid computing solutions to subproblems that are not needed, i.e. Common Subsequence

Can be more intuitive for matrix chain multiplicationa

### Tabulation (bottom up)

Avoids multiple lookups and function calls, less memory

Example:

For longest increasing subseqeunce 

their idea was to compute the end of the longest sequence ending at that index, so you iterate over the  

### Longest increasing subsequence

They represent the matches as a table and compute the scores starting from 0,0

##Backtracking

Here the idea is to exhaustively search all combinations. 

Important numbers: 2^n combination for true/false, choose/don't choose - true for sets with unique elements. 

Iterative solution can be modeled as a iteration over bit turn on / off

counter & 1 << j 

and the backtracking can use a stack like structure to add an element, exhaustively search that space, and then remove the element

When Do We Switch from DP to Backtracking? 

if we need to return all solutions -> backtracking

if we need to count all solution -> memoization + dp

## Sliding Window

https://medium.com/leetcode-patterns/leetcode-pattern-2-sliding-windows-for-strings-e19af105316b

Counters for words/char/digits

Another counter for restrictions such as number of distinct etc characters

## DFS + BFS

Using a stack imp for DFS

Queue for level traverse (BFS)

Stack for DFS

DFS - Exhaustive search, find all combinations/paths

BFS - Shortest path. For graphs where all edges are the same weight -> shortest path is equiv. to Dijkstra

Imp. tree level printing

# Take Home Messages from Solutions

Hashing - use a bit array and turning bits on and off

For small digit/chat count - use array counters

For list traversal - assuming we know 

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

First, indexing is the worsest. 

Second, stake home messages:

remember to return -1; 

remember to use early exit for Dijikstra

That was the function for converting 

r = n - (li-1) / n - 1;
c = (li-1) % n;
if((n-r-1)%2) c = n - c - 1;

## DP

### 8 Queens

Again, indexing and index tests are a bitch - esp the diagonals (check all diagonals!)

Use the row as a way to keep track of what was already searched

## Power Set

The power set is a set of all possible subsets

Power set size is 2^n

Testing with bit shift: 

j = 1 < n

counter & 1 << j

### Count Digits

The task was to count all of the 1's in all of the numbers between 0 and i. 

Here it was pretty simple in the sense that you compute your current power and return the pointer to the first cell in the array where you already stored the computation every time you reach a new power.

### Coin Change

First, sorting the coins ensures that we test against the largest coins first.

Here, to do the caching for the DP we use an array with the amount and store the min amount of coin for this value.

# TODOS

https://algorithms.tutorialhorizon.com/colorful-numbers/