
void urlify(const int len, const char c, const std::string& rep, std::string& str) {
	
	int sc = 0; //space counter
	int slen = rep.length();

	for (int i = 0; i < len; i++) {
		if (str[i] == c) sc++;
	}

	//safety check?
	int idx = len-1;
	for (; idx >= 0 && sc > 0; idx--) {
	    std::cout << str << std::endl;
		if (str[idx] != ' ') {
			str[idx+(slen-1)*sc] = str[idx];
		} else {
			str.replace(idx+(slen-1)*(sc-1), slen, rep);
			sc--;
		}}


int main() {

	std::string str = "Mr John Smith        ";
	auto c = ' ';
	std::string rep = "%20";
	int len = 13;

	urlify(len, c, rep, str);

	std::cout << str << std::endl;

	return 0; 
}