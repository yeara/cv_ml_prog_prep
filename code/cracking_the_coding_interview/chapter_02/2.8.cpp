# loop detection 

return the node at the beginning of the loop

O(N) with a hash map 

The idea is visit all of the nodes in the linked list and mark them as visited using a hash-set. The first time we hit a node that was visited before, we have found the start of the loop.

The address of the nodes are 32 bit unsigned integer, so we can hash that.

The c++ hashmap is std::unordered_map
We also use a set

insert
find
count - for a set has a value of 0,1

the initialization is a bit crap, but for integers we can use the default

class LinkedList<T> {

public:

	LinkedList<T>() : head(nullptr) {}
		
	void insert(const T& val) {
		
		Node* n = new Node(val);

		if (head == nullptr) {
			head = n;
		}
		else {
			Node* p = head;
			while(p->next != nullptr) {
				p = p->next;
			}
			p->next = n;
		}
	}

	bool remove(Node* n) {

		if (n == nullptr) return false;

		Node* p = head;

		while (p->next != n && p->next != nullptr) {
			p = p->next;
		}

		if (p->next == nullptr) {
			return false;
		} 

		p->next = n->next;

		return true;
	}

	Node* get_node(T val) {

		Node* n = head;

		while (n != nullptr) {
			if (n->value == val) return n;
			n = n->next;
		}

		return nullptr;
	}

	private:
		Node* head;
};

class Node<T> { 
public:
	Node* next; 
	T value;
};

// Return the node at the beginning of the first loop
Node* loop_detection(Node* head) {
	Node* p = head;
	std::unordered_set<unsigned_int> visited;

	while (p!=nullptr) {
		if (visited.count((unsigned_int)p) > 0) {
			return p;
		}

		visited.insert((unsigned_int)p);
		p = p->next;
	}

	return nullptr;
}

std::unordered_set<int> visited;


// quick breakdown - go to this node
// copy the data from the next node
// delete the next node

Node* remove_middle_node(Node* n) {
	
	if (n == nullptr) return;

	n->value = n->next->value;
	n->next = n->next->next;

	delete n->next;
}

