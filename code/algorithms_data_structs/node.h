#include <iostream>

template <typename T> class Node {
public:
	Node() {
		value = 0;
		next = nullptr;
	}

	Node(T data) {
		value = data;
		next = nullptr;
	}

public:
	Node<T>* next; 
	T value;
};

template <typename T> 
void print_list(Node<T>* head) {
	Node<T>* s = head;
	while (s!= nullptr) {
		std::cout << s->value << " ";
		s = s->next;
	}
 	std::cout << std::endl;
}

Node<int>* build_list() {

	int arr[] = {0, 1, 1, 2, -9, -7, 15, 0, 6, 15, -7, 1, 2, 3, 5, 5};
	int listsize = 16;

	Node<int>* head = new Node<int>();
	head->value = arr[0];

	Node<int>* s = head;

	for (int i = 1; i < listsize; i++) {
		Node<int>* n = new Node<int>();
		n->value = arr[i];
		s->next = n;
		s = n;
	}

	return head;
}


template <typename T>
Node<T>* build_list(const std::vector<T>& values) {

	int listsize = values.size();
	if (listsize == 0) return nullptr;

	Node<T>* head = new Node<T>();
	head->value = values[0];

	Node<T>* s = head;

	for (int i = 1; i < listsize; i++) {
		Node<T>* n = new Node<T>();
		n->value = values[i];
		s->next = n;
		s = n;
	}

	return head;
}
