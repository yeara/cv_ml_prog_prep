#include <iostream>
#include <vector>

template <typename T>
int binary_search(const std::vector<T>& arr, const T& x) {
	
	int left = 0;
	int right = arr.size()-1;
	int mid;

	while (left <= right) {
		mid = (left + right) / 2;

		if (arr[mid] < x) {
			left = mid + 1;
		}
		else if (arr[mid] > x) {
			right = mid - 1;
		} else {
			return mid;
		}
	}
	
	return -1;
}

int main() {
	
	std::vector<int> arr = {0,1,2,3,15,16,17,19,20,22};

	std::cout << binary_search(arr, 0) << std::endl;
	std::cout << binary_search(arr, 15) << std::endl;
	std::cout << binary_search(arr, 20) << std::endl;
	std::cout << binary_search(arr, 23) << std::endl;
}