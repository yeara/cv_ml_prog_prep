#include <iostream>
#include <string>

const int ALPHABET_SIZE = 128;

// O(N)
void count_appearances(const std::string& str, int* counters) {
	for (auto c : str) {
		counters[(int)c]++; //not safe
	}
}

void init_to_zero(int* arr, const int ALPHABET_SIZE) {
	for (auto i = 0; i < ALPHABET_SIZE; i++) {
		arr[i] = 0;
	}	
}

bool is_permutation(const std::string& str1, const std::string& str2) {
	
	if (str1.size() != str2.size()) {
		return false;
	}
	
	int	c1[ALPHABET_SIZE];
	int	c2[ALPHABET_SIZE];

	init_to_zero(c1, ALPHABET_SIZE);
	init_to_zero(c2, ALPHABET_SIZE);

	count_appearances(str1, c1);
	count_appearances(str2, c2);

	// O(S)
	for (int i = 0; i < ALPHABET_SIZE; i++) {
		if (c1[i] != c2[i]) {
			return false;
		}
	}

	return true;
}

int main() {
    
    auto str1 = "aabbccdd";
    auto str2 = "aabbddce";
    
    auto b = is_permutation(str1, str2);
    std::cout << b << std::endl;
    
    return 0;
}