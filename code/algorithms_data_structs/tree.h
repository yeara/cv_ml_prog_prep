#include <iostream>

#ifndef tree_h
#define tree_h

template <typename T> 
class TreeNode {
public:
	T value;
	TreeNode* left; 
	TreeNode* right; 
	TreeNode* parent;

	TreeNode() : left(nullptr), right(nullptr), parent(nullptr)
	{}

	TreeNode(T val, TreeNode* p) : left(nullptr), right(nullptr), parent(p), value(val)
	{}

	TreeNode(T val) : left(nullptr), right(nullptr), parent(nullptr), value(val)
	{}

	void insert_in_order(T val) {
		if (val <= value) {
			if (left == nullptr) {
				left = new TreeNode(val);
			} else {
				left->insert_in_order(val);
			}
		} else {
			if (right == nullptr) {
				right = new TreeNode(val, this);
			}
			else {
				right->insert_in_order(val);
			}
		}
	}
};

template <typename T> class BinaryTree {
public:
	TreeNode<T>* root;
};

template <typename T> 
void inorder_traverse(TreeNode<T>* node) {
	if (node != nullptr) {
		inorder_traverse(node->left);
		visit(node);
		inorder_traverse(node->right);
	}
}

template <typename T> 
void postorder_traverse(TreeNode<T>* node) {
	if (node != nullptr) {
		inorder_traverse(node->left);
		inorder_traverse(node->right);
		visit(node);
	}
}

template <typename T> 
void preorder_traverse(TreeNode<T>* node) {
	if (node != nullptr) {
		visit(node);
		inorder_traverse(node->left);
		inorder_traverse(node->right);
	}
}

template <typename T> 
void visit(TreeNode<T>* node) {
	if (node != nullptr) {
		std::cout << node->value << std::endl;
	}
}

#endif