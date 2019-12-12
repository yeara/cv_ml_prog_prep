#include <iostream>
// #include "node.h"
// #include "list.h"
#include "stack.h"

int main() {

	Stack<int> stack;
	stack.push(3);
	stack.push(4);
	stack.push(5);

	while (!stack.is_empty()) {
		std::cout << stack.pop() << std::endl;
	}

	stack.pop();
	
	return 0;
}