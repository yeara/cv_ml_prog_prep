#include <iostream>
#include "tree.h"
#include "check_balanced.h"

int main() {
	
	TreeNode<int> root = TreeNode<int>(4);

	std::cout << "Root depth " << depth(&root) << std::endl;

	root.insert_in_order(1);
	std::cout << "(4,1) depth " << depth(&root) << std::endl;

	root.insert_in_order(2);
	std::cout << "(4,1,2) depth " << depth(&root) << std::endl;

	std::cout << check_balanced(&root) << std::endl;

	root.insert_in_order(7);
	std::cout << depth(&root) << std::endl;

	root.insert_in_order(8);
	std::cout << depth(&root) << std::endl;

	std::cout << check_balanced(&root) << std::endl;

	root.insert_in_order(5);
	std::cout << check_balanced(&root) << std::endl;

	root.insert_in_order(6);
	std::cout << check_balanced(&root) << std::endl;

	root.insert_in_order(10);
	root.insert_in_order(11);
	root.insert_in_order(12);
	std::cout << check_balanced(&root) << std::endl;

	return 0;
}