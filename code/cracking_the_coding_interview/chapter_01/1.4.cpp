#include <string>
#include <iostream>
#include <vector>
#include <map>

bool isPalindromePermutation(const std::string& s) {
	
	std::map<char,int> hash;

	for (auto c : s) {
		if (hash.count(c) == 0) {
			hash[c] = 1;
		}
		else {
			hash[c] += 1;
		}
	}

	bool foundUneven = false;
	for (auto it = hash.begin(); it != hash.end(); it++) {
		if (it->second % 2 != 0) {
			if (!foundUneven) {
				foundUneven = true;
			}
			else {
				return false;
			}
		}
	}

	return true;
}


int main() {
	

	std::string s0 = "cat";
	std::string s1 = "cat tac";
	std::string s2 = "ccaad";
	std::string s3 = "ccaa";
	std::string s4 = "abcd";

	std::vector<std::string> tests = {s0,s1,s2,s3,s4};

	for (auto it : tests) {
		std::cout << it << " " << isPalindromePermutation(it) << std::endl;
	}

	return 0;
}