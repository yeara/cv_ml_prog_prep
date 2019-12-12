#include <string>
#include <vector>
#include <iostream>

// idea - iterate through the string and insert the character and the count
// if we exceed the length of the orig string terminate

std::string compress_string(const std::string& s){
	
	if (s.size() < 3) {
		return s;
	}

	std::string n;

	int l = s.length();
	int currcount = 1;

	for (int i = 1; i < l && n.length() < l; i++) {
		if (s[i] == s[i-1]) {
			currcount++;
		} else {
			n += s[i-1] + std::to_string(currcount);
			currcount = 1;
			if (n.length() >= l) {
				return s;
			}
		}
	}

	//now we need to handle the last character:
	n += s[l-1] + std::to_string(currcount);

	return n;

}

int main() {
	
	std::string s1 = "a"; 
	std::string s2 = "aaa"; 
	std::string s3 = "aa"; 
	std::string s4 = "aabbccca"; 
	std::string s5 = "aabbccc"; 
	
	std::vector<std::string> tests = { s1,s2,s3,s4,s5 }; 

	for (auto it : tests) {
		std::cout << it << " " << compress_string(it) << std::endl;
	}

	return 0;

}