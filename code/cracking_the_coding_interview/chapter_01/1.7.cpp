// 1.7 rotate matrix  - in place
#include <string>
#include <vector>
#include <iostream>

// assume the input is valid and square
void rotate(std::vector<std::vector<int> > & M) {
	
	// ok, let's do this layer by layer.

	int N = M[0].size(); 

	int nlayers = N / 2; 
	int temp;

	for (int l = 0; l < 1; l++) {
		int first = l;
		int last = N-l-1;

		std::cout << "first " << first << std::endl;
		std::cout << "last " << last << std::endl;

		for (int i = first; i < last; i++) {
			int& top = M[i][first];
			int& right = M[last][i];
			int& bottom = M[last-i][last];
			int& left = M[first][last-i];


			temp = left;
			left = bottom;
			bottom = right;
			right = top;
			top = temp;
		}
	}
}

void print_image(const std::vector<std::vector<int> > & M) {
	
	int n = M[0].size(); 

	for (int i = 0; i < n; i++) {
		for  (int j = 0; j < n; j++) {
			std::cout << M[j][i] << " ";
		}
		std::cout << std::endl;
	}
}

int main() {

	std::vector<int> c0 = { 0, 4, 8, 12};
	std::vector<int> c1 = { 1, 5, 9, 13};
	std::vector<int> c2 = { 2, 6, 10, 14};
	std::vector<int> c3 = { 3, 7, 11, 15};
	
	std::vector<std::vector<int> > image = { c0, c1, c2, c3 }; 

	print_image(image);
	rotate(image);
	print_image(image);

	return 0;

}