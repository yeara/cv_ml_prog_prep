#include <iostream>

int multiply(int num1, int num2) {
	
	int res = num1;
	int i = 1;

	while ( i < num2 ) {
		if (i*2 == num2) {
			i *= 2;
			res += res;
		}
		else {
			res += multiply(num1, num2 - i);
			i += (num2 - i);
		}
	}

	return res;
}

int main() {
	
	std::cout << "Mutliply 8,3 " << multiply(8,3) << std::endl;
	std::cout << "Mutliply -1,4 " << multiply(-1,4) << std::endl;
	std::cout << "Mutliply 5,13 " << multiply(5,13) << std::endl;
	std::cout << "Mutliply 5,6 " << multiply(5,6) << std::endl;

	return 0;
}
