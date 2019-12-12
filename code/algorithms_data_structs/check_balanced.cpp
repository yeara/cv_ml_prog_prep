// check balanced 

// neg example
// 			0
// 		1		2
// 	3		4
// 5		6		


// pos example
// 			0
// 		1				2
// 	3		4		11		12
// 5		6 7		8		


// A tree is balanced if the height of every two subtree differs by at most 1


// the very stupid option: 

// check the height of each subtree pairs, starting from the root and propagating up the tree
// each node will be visited log n ^2 times? 

// so for each node check if depth(left) == depth right +- 1


#include <iostream>
#include "tree.h"

#ifndef check_balanced_h
#define check_balanced_h

template <typename T>
int depth(TreeNode<T>* node) 
{
	
	if (node == nullptr) return 0;

	int dr = depth(node->right);
	int dl = depth(node->left);

	return 1 + std::max(dr, dl);
}

template <typename T>
bool check_balanced(TreeNode<T>* node) 
{
	if (node == nullptr) return true;

	int dr = depth(node->right);
	int dl = depth(node->left);

	if (std::abs(dr-dl) < 2) {
		return check_balanced(node->left) && check_balanced(node->right);
	}

	return false;
}

#endif