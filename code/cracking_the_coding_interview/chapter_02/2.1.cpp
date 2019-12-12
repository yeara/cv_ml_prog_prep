#include <iostream>
#include <map>
#include "node.h"

// first, let's do things the silly way
template <typename T>
void rem_duplicates(Node<T>* head) { 

	Node<T>* s = head;
	Node<T>* p = head;

	while (s != nullptr) {
		p = s;
		std::cout << "s " << s->value << std::endl;
		while (p !=nullptr && p->next != nullptr) {
			std::cout << "p " << p->next->value << std::endl;
			if (p->next->value == s->value) {
				Node<T>* temp = p->next;
				p->next = p->next->next;
				delete temp;
			}
			p = p->next;
			std::cout << p << std::endl;
		}
		print_list(head);
		s = s->next;

	}

	return;
}

template <typename T>
void rem_duplicates_with_hash(Node<T>* head) {
		
	std::map<T,int> hash; 

	Node<T>* s = head;

	if (head == nullptr) return;
	if (head->next == nullptr) return;

	while (s != nullptr) {
		if (hash.count(s->value) == 0) {
			hash[s->value] = 1;
		} else {
			hash[s->value]++;
		}
		s = s->next;
	}

	// //now go through hash and remove duplicates from the list:
	// for (auto it = hash.begin(); it != hash.end(); it++) {
	// 	if (it->second > 1) {
	// 		s = head;
	// 		int count = it->second;
	// 		while(s->next != nullptr && count > 1) {
	// 			if (s->next->value == it->first) {
	// 				s->next = s->next->next;
	// 				delete s->next;
	// 			}
	// 			s = s->next;
	// 			}
	// 	}
	// }

	//or a better way would be to iterate through the list and lookup:
	Node<T>* p = head;
	s = head->next;
	while(s != nullptr) {
		if (hash[s->value] > 1) {
			hash[s->next]--;
			p->next = s->next;
			delete s;
		}
		p = s;
		s = s->next;
	}

	return;
}


int main() {

	Node<int>* head = build_list();
	print_list(head);
	rem_duplicates(head);
	print_list(head);

	return 0;
}
