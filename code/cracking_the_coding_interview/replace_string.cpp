
# assumptions: the string is long enough
# you know the true length of the string

# inputs: string, length

Brute force solution:
Iterate over the array
When you hit a space, copy all characters two spaces to the "right" and insert the "%20"
Complexity: ~N^2 because at most we have N characters for which we have to copy most of the string (the real complexity is (N + (N-1) + .. + 1) = n/2 * (N+1)

Smarter solution: 
Iterate over the array once and count all spaces. 
Starting from the end of the string, iterate forwards. 
Now backward iteration: move every character by #space + 3 to the right, until you hit a space
Then change the space to %20 and decrease the space counter
Cache the index of the start of the string
Decrease the space counter
Complexity: O(N)


#include <iostream>
#include <string>

void urlify(const int len, const char c, const std::string& rep, std::string& str) {
	
	int sc = 0; //space counter
	int slen = rep.length();

	for (int i = 0; i < len; i++) {
		if (str[i] == c) sc++;
	}

	//safety check?

	int idx = len;
	for (; idx >= 0 && sc > 0; idx--) {
		if (str[idx] != ' ') {
			str[idx+(slen-1)*sc] = str[idx];
		} else {
			str.replace(idx+(slen-1)*sc, slen, rep);
			sc--;
		}
	}
}


int main() {

	auto str = "Mr John Smith        ";
	auto c = ' ';
	std::string rep = "%20";
	int len = 13;

	urilfy(len, c, rep, str);

	std::cout << str << std::endl;

	return 0; 
}