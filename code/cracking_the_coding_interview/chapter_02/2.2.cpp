// remove kth to last element
// the idea is to have two runners, k elements apart
/// when the last runner hits the last node, return the second runner

#include "node.h"
#include <iostream>
#include <vector>

template <typename T>
Node<T>* return_kth_last_element(Node<T>* head, int k) {
	
	if (k < 1) { return nullptr; }

	Node<T>* s = head;
	Node<T>* p = head;

	int counter = 0;

	//first get (k-1)-spaces between p,s 
	while (p->next != nullptr && counter < k-1) {
		p = p->next;
		counter++;
	}

	if (p->next == nullptr ) { //end of list 
		if (counter < k-1) { // we did not get enough distance
			return nullptr;
		}
	}

	while (p->next != nullptr) {
		p = p->next;
		s = s->next;
	}

	return s;
}

int main() {
	
	std::vector<char> values = {'A','B','C','D'};
	Node<char>* head = build_list(values);

	std::vector<int> idx = {1,2,4,8,0};

	for (auto val : idx) {
		auto res = return_kth_last_element(head, val);
		std::cout << res << std::endl;
		if (res != nullptr) {
			std::cout << val << res->value << std::endl;
		}
		else {
			std::cout << val << " " << res << std::endl;
		}
	}

	return 0;
}