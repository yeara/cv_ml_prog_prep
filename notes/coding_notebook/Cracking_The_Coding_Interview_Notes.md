# Cracking The Coding Interview

## Big O Notation

$\Theta()$ notation - expected upper bound for an algorithm runtime

Usually expected and worst case is similar

Amortized run time.

Loop run times: O(N)

Nested loops: O(N^2)
The actual full run time: 

O(n(n-1)/2) which is the sum of a  1+2+...+(N-1)
The sum is equal to: 
N(N-1)/2

Recursive calls: 
O(branches^depth)
O(2^(n))

## Problem Solving
For each question:

1. Pay attention - Record unique details for the problem (sorted arrays for example, positive numbers, etc)
2. Ask for clarifications
3. Create examples and debug them for special cases
   ​	Not too simple
   ​	Not too beautiful (i.e. not balanced trees)
4. State the brute force solution and explain the drawbacks
5. Optimize the solution - 
   	look for unused info 
   	manual solution
   	solve incorrectly
   	time vs space trade offs
   	pre-compute solutions
6. Pseudo code should be very high level

7. walk through the code in detail
   - re-read any copy and pasted code carefully
   - re-check if else / statements
   - if need to to debug during interview, explain in real time

### When implementing

- Write beautiful code 
- Modularized - write functions like initArray()... 
- Use classes / data structures generously for input / output types
- Pretend you have the data structures that you need
- Test!
- Appropriate code reuse
- Break down the common elements of the problems into functions that can be reused
- Flexible code, no need to assume simplest scenario, unless this makes things very easy

### Optimize and Solve Technique: 

- create a nice and large example
- try to solve it as human and not as a computer
- for example of finding a substring in a long string: 
  - sliding window for which substrings can be computed
  - pages 67-71

### Base case and Build for Recursive Algorithms
(for that number generation thing - )
Write out the output / result for the base cases
Stop at the first interesting case and generalize the solution
(abc) example of generation all permutations

Example for small example work:

Permutations of n elements: n!
iteration 0: "str", ""
subcalls: 
i=0 	rem = "t"
		permutation ("t", "s")

i=1 	rem = "st"
		permutation ("st")
i=2		rem = "str"
		permutation ("str", "r")

## Object Oriented Design 

### Handle Ambiguity

Ask question, important to understand who it is for

- who
- what
- where
- when
- how
- why

### Define Core Objects

For a restaurant:

- table
- server
- party
- host
- etc
- employee

### Understand the Relationships between Objects

- for example: table and party
- each restaurant has one host
- how general should the design be?

### Investigate Actions 

- what are the key actions that objects take and how do they relate to each other? 

## Data Structures

### Linked Lists

Single linked lists: 

The order of the nodes does not depend on their location.
Random access is O(N) (worst case)
Constant time insertion and removal O(1)

Head node 

class Node {
	public:
		Node* next;
		int data;
}

bool appendNode(Node* head, Node* node) {
	
	Node* p = head;
	while (p->next != nullptr) {
		p = p->next;
	}
	
	p->next = node;
	return true;
}


For two linked lists, we can store a single value using the XOR operation: 

link = addr(prev)^addr(next)

And then: 
addr(prev)^link = addr(next)

Cycles: 

Start by having a fast pointer and a slow pointer. 

Finding cycle entry points:

The distance the fast pointer travels is equal to start + #cycles 

### Hash Tables

- Lookup is amortized O(1)
- There might be some clashes
- Implemented as array of linked list + hash function as there might be clashes
  - Hash (key) % length
- Implemented as balanced binary search tree 
  - lookup time of O(1)
  - Less storage 

### Vectors

- Dynamic array - auto resizeable 
- Grow by a factor of 2 usually. 
- O(1) read
- O(n) time to double the array
- O(1) insertion 

### AVL Trees

- Complete - formal definition 

- Node maintains three pointers: 
  - parent, left child, right child
- height of leaf = 0 
- height of node = max(height of children)+1
- We can store the height of every node for free and maintain it in constant operations
- goal - keep the heights of the node small
- depth of null ptrs = -1
- Require the left and right children of every node to differ by at most \pm +-1
- Tree height - log(n)
- Worst case - left subtree has a height of +1 more than every right subtree
  - the total number of nodes in this case is 1 + N_{h-1} + N_{h-2}

#### Insertion + Rotations

1. find the place in the tree for the element 
2. Validate that the AVL property of the tree is valid from this node up 
3. If it's invalid, we solve it by two possible operations: 
   1. Rotate right (subnode), rotate left (node)
   2. Rotate left (node), and move upwards

### Heaps 

The heap rule is: 

- The element contained by each node is greater than or equal to the elements of that node's children.
- The tree is a complete binary tree so that every level except the deepest must contain as many nodes as possible; at the deepest level all the nodes are as far left as possible.

For heaps, insert the element furtherst away from the root and propagate it up by switching its place with its parent? 

[https://www.geeksforgeeks.org/insertion-and-deletion-in-heaps/#:~:targetText=Process%20of%20Deletion%3A,last%20element%20from%20the%20Heap.](https://www.geeksforgeeks.org/insertion-and-deletion-in-heaps/#:~:targetText=Process of Deletion%3A,last element from the Heap.)

#### Adding elements

At the last position at the bottom most level

#### Removing Elements

1. The element that we will be removing will **always** be the root. 
2. To do so, save the value of the root into a variable.
3. Copy the last element in the deepest level to the root. This will be called the out-of-place element.
4. Take this last node out of the tree.
5. While the out-of-place element has a priority that is lower than the one of its children) Swap the out of place element with its highest priority child.
6. Return the value that you saved in step 1.

#### Heap Storage

Since heaps are complete binary trees, we can store them in a vector array. 

The location of the nodes at the i-th levels are at indices: 

### Graph Representation

Nodes

Edges

Directed graphs vs. Undirected graphs

#### Adjacency Matrix

- For undirected graphs - symmetric
- Can represent graph weights
- Easier to imp and follow
- Edge removal takes O(1) time
- Edge queries O(1)
- Consumes a lot of space $O(n^2)$ 
- Adding a vertex to the graph is an expensive operation $O(n^2)$ 

#### Adjacency List

An array with number of cells == number of vertices.

For each vertex, maintain a list of the vertices adjacent

If the edges are weighted, main a list of pairs <edge,weight> 

- Space: O(|V|+|E|), max space usage is O|V^2| is the graph is a clique
- Dynamically adding vertices is easier
- Edge Queries: O(V) - not efficient

## Abstract Data Type 

The abstract data type defines the interface, and the data structure supports or provides the implementation of these operation in the required time complexity: 
- insert, delete
- min
- successor / predecessor 

### Possible Data Structures

- Priority queue
- heap - in place, less pointers
- balanced search tree / AVL

### PQs

Define: 

- insert(x)
- max(x)
- extract_max - return and removes from S 
- increase_key
- decrease_key

### (Max) Heap operations
produces a max heap from an unordered array

max-heapify

# C++11 
https://en.wikipedia.org/wiki/Anonymous_function#C++_(since_C++11)

# Algorithms
## A*

## Dijkstra

Initialize list of vertex distances. Set them all to infinity.

Initialize a list of the previous node for each vertex. Set it to null. 

Add the source to a pq with a distance of 0, and also fix this score in the graph

while !q.empty() {

​	auto node = q.front()

​	q.pop()

​	for (n : node.next) {

​	}	

}

## DFS (Depth First Search)

This is equivalent to traversing a tree in by using a stack. 

## BFS (Breadth First Search)

This is equivalent to traversing a tree using a queue. 

Equiv to traversing a tree one level at a time.

# Sorting

## Bubble sort

`For (int  i = 0; i < length && swapped; i++) { `

`swapped = false;`

​	`for (int j = 0; j < length; j++) {`

​		`if (l[j] > l[j-1]) { l[j] = l[j-1]; `

`	swapped = true;}`

​	`}`

`}`

At every iteration the last element is at the correct location.

Worst base: O(N^2)  comparison and swaps

Best case O(N) comparison and O(1) swaps

Constant space

## Merge Sort

Stable sort

## Quick Sort

The algorithm is made of two functions: partition and quicksort.

Choose a pivot element (usually the middle element in the array)

Move all of the elements larger than the pivot to the right side of the array

Return the location of the final pivot element

The pivot is now at the correct place?

There algorithm is made up of two functions:

partition - choose a pivot element value

sort the array into two sections, where one contains all of the elements larger than and one smaller then 

when iterating, don't forget to check if the idx point the same location, if they're smaller than 0

return their meeting point?

now quick sort both halves

iterate through the array going from left "up" and right downwards and swap all of the elements that are smaller/larger than the pivot 

The counters will meet at the center of the array, where 

quicksort

unstable!

### Heapsort

/TODO
