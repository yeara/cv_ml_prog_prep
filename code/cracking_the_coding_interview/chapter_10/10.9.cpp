//  sorted matrix search

// basically, we do some kind of 2d binary search. 

// for every element, we check the middle element of the matrix 
// if it's bigger that that value - we know we need to check the part of the matrix to the bottom right (4)

// if it's smaller - we have three possible sub matrix to check  1 | 2 :
// 															  3 | 4

// 1,2,3

// we check the upper left element for each matrix and decide if it makes sense to proceed in each sub-matrix 


// important to ensure we're doing bound check correctly 

// our stopping criterion: 
// matrix size is 1x1

// the function call would be 

// we need a small class with the bounds:


#include <iostream>

struct idx {
	int row = -1;
	int col = -1;	
};


// ignoring bounds check
bool find_in_matrix(const int(&arr)[4][4], int value, idx start, idx end, idx& res) 
{
	std::cout << "find_in_matrix " << start.row << " " << start.col << " " << end.row << " " << end.col << " " << std::endl;

	
	if (arr[start.col][start.row] > value)  {
		return false;
	}

	idx mid; 
	mid.row = (end.row + start.row) / 2;
	mid.col = (end.col + start.col) / 2;

	int midval = arr[mid.col][mid.row];

	if (midval == value)  { 
		res = mid;
		return true;
	}

	if ((start.row == end.row) && (start.col == end.col) && (midval != value)) {
		return false;
	}

	if (midval > value) {

		//search submatrix 1 
		bool succ = find_in_matrix(arr, value, start, mid, res);

		// search submatrix 2 
		if (!succ) {
			idx start2, end2; 
			start2.row = start.row;
			start2.col = mid.col+1;

			end2.row = mid.row;
			end2.col = end.col;
			succ = find_in_matrix(arr, value, start2, end2, res);
		}

		if (!succ) {

			idx start3, end3; 
			start3.row = mid.row+1;
			start3.col = start.col;

			end3.row = end.row;
			end3.col = mid.col;

			return find_in_matrix(arr, value, start3, end3, res);
		}

		return succ;

	} else {
		//search submatrix 4

		idx start4;
		start4.col = mid.col+1;
		start4.row = mid.row+1;
		return find_in_matrix(arr, value, start4, end, res);
	}
}

int main() {

	int arr[4][4] = {{0,1,7,8},{1,4,8,11},{2,5,9,12},{3,6,10,13}}; 

	idx res, start, end;
	start.row = 0;
	start.col = 0;
	end.row = 3;
	end.col = 3;

	int  value = 8;
	auto succ = find_in_matrix(arr,value,start,end,res);

	std::cout << "Success " << succ << " " << res.row << " " << res.col << " " << std::endl;

	return 0;
}
