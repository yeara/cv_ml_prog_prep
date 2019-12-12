#include "node.h"

template <typename T> class Stack {
    public:

	Stack() {
		head = nullptr;
	}

	~Stack() {
		while(head != nullptr) {
			pop();
		}
	}

	bool is_empty() const {
		return head == nullptr;
	}

	T pop() {
		if (head == nullptr) {
			throw 0;
		}

		Node<T>* n = head;
		T val = n->value;
		head = n->next;
		delete n;
		return val;
	}

	void push(const T& val) {
		Node<T>* n = new Node<T>(val);
		n->next = head;
		head = n;
	}

	T peek() {
		if (head != nullptr) {
			return head->value;
		}
		else {
			throw -1;
		}
	}

private:
	Node<T>* head;

};