#include <string>
#include <vector>
#include <iostream>

bool isOneAway(const std::string& s1, const std::string& s2) {
	//the idea is to iterate through both string and count the number of changes
	//if they're the same - no problem
	// if one iterator is needs to be moved one ahead to continue, it's also ok 
	// we can also check first the str count and not even start if they're too far apart

	bool changed = false;

	int l1 = s1.length();
	int l2 = s2.length();

	if (std::abs(l1-l2) > 1) {
		return false;
	}

	//need to check three cases: 

	//replacement
	if (l1 == l2) {
		int i = 0;
		for (int i = 0; i < s2.length(); i++) {
			if (s1[i] != s2[i]) {
				if (changed) return false;
				changed = true;
			}
		}
	}

	// insertion
	bool inserted = false;
	if (l1 == l2+1) {
		for (int i = 0; i < s2.length(); i++) {
			if (s1[i+inserted] != s2[i]) {
				if (inserted) return false;
				inserted = true;
			}
		}
	}

	// deletion
	bool deletion = false;
	if (l1+1 == l2) {
		for (int i = 0; i < s2.length(); i++) {
			if (s1[i] != s2[i+deletion]) {
				if (deletion) return false;
				inserted = true;
			}
		}
	}

	return true;
}

int main() {
	
	std::pair<std::string,std::string> s1 = { "pale", "ple" }; 
	std::pair<std::string,std::string> s2 = { "pales", "pale" }; 
	std::pair<std::string,std::string> s3 = { "bale", "pale" }; 
	std::pair<std::string,std::string> s4 = { "pale", "bake" }; 
	
	std::vector<std::pair<std::string,std::string> > tests = { s1,s2,s3,s4 }; 

	for (auto it : tests) {
		std::cout << it.first << " " << it.second << " " << isOneAway(it.first, it.second) << std::endl;
	}

	return 0;

}