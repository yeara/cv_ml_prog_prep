#include <iostream>
#include <string>
#include <vector>

void generate_all_permutations(const std::string& str, std::string pre, std::vector<std::string>& solutions) {

	if (str.length() == 1) {
		std::string s1 = pre + str[0];
		solutions.push_back(s1);
		return;
	}
	else {

		for (int i = 0; i < str.length(); i++) {
			char currchar = str[i];
			if (str.find(currchar) < i) continue;

			std::string newpre = pre;
			newpre += str[i];

			std::string newstr = str.substr(0,i);
			newstr = newstr + str.substr(i+1,str.length()-(i+1));


			generate_all_permutations(newstr, newpre, solutions);
		}
	}
}


int main() {

	std::vector<std::string> solutions; 
	std::string input = "aabba";
	std::string pre = "";

	generate_all_permutations(input, pre, solutions);

	for (int i = 0; i < solutions.size(); i++) {
		std::cout << solutions[i] << std::endl;
	}
}
