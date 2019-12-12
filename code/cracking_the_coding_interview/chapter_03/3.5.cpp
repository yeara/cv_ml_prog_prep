#include <iostream>
#include "stack.h"

template<typename T> 
int count_stack_depth(Stack<T>& s, Stack<T>& b) {
	int d = 0; //depth
	while (!s.is_empty()) {
		T val = s.pop();
		++d;
		b.push(val);
	}

	while(!b.is_empty()) {
		T val = b.pop();
		s.push(val);	
	}

	return d;
}

template<typename T> 
T pop_and_store_max(Stack<T>& s, Stack<T>& b, int unsorted_depth) {
	T max = s.peek();

	for (int i = 0; i < unsorted_depth; i++) {
		b.push(s.pop());
		if (b.peek() > max) {
			max = b.peek();
		}
	}

	std::cout << "Max " << max << " in depth " << unsorted_depth << std::endl;
	return max;
}

template<typename T> 
void push_back_dump_max(const T& max, Stack<T>& b, Stack<T>& s) {
	bool found = false;
	while(!b.is_empty()) {
		T val = b.pop();
		if (val == max && !found) { //first time we hit this value
			//do nothing
			found = true;
			// do not push back the value
		} else {
			s.push(val);
		}
	}
}


template <typename T>
bool sort_stack(Stack<T>& s) {

	Stack<T> b; //buffer
	int unsorted_depth = count_stack_depth(s,b);

	while (unsorted_depth > 0) {
		std::cout << unsorted_depth << std::endl;
		std::cout << "popping" << std::endl;
		T max = pop_and_store_max<T>(s,b,unsorted_depth);
		s.push(max);
		std::cout << "pushing" << std::endl;
		push_back_dump_max<T>(max,b,s);
		std::cout << "finished" << std::endl;
		int d = count_stack_depth(s,b);
		std::cout << "depth " << d << std::endl;
		unsorted_depth--;
	}

	return true;
}


int main() {

	Stack<int> stack;
	stack.push(5);
	stack.push(2);
	stack.push(-1);
	stack.push(4);
	stack.push(7);
	stack.push(7);
	stack.push(6);

	// while(!stack.is_empty()) {
	// 	std::cout << stack.pop() << std::endl;
	// }

	sort_stack(stack);

	while(!stack.is_empty()) {
		std::cout << stack.pop() << std::endl;
	}

	return 0;
}
