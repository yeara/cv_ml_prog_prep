#include <iostream>

template <typename T> 
int partition(T* arr, int low, int high) {
	
	int pivot = arr[(low + high) / 2];

	while (low <= high) {
		while (arr[low] < pivot && low <= high) low++;
		while (arr[high] > pivot && high >= 0) high--;

		if (low <= high) {
			T temp = arr[low];
			arr[low] = arr[high];
			arr[high] = temp;
			low++;
			high--;
		}
	}
	return low;
}

template <typename T> 
void quicksort(T* arr, int low, int high) {
	if (low == high) return; 
	int pi = partition(arr, low, high);
	quicksort(arr, low, pi-1);
	quicksort(arr, pi+1, high);
}


int main() {

	int arr[] = {0,6,5,2,10, 11, 34, 56, 78,12,34,90,-123}; 

	quicksort<int>(arr,0,13);

	for (int i = 0; i < 13; i++) {
		std::cout << arr[i] << " ";
	}

	return 0;
}


