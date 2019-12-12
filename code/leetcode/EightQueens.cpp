#include <iostream> 
#include <vector>

const int ROWS = 8;
const int COLS = 8;

bool place_row(int row, const std::vector<int>& cols, std::vector<std::vector<int>>& solutions) 
{
	//placed all of them - WIN! :)
	if (row == ROWS)  {
		solutions.push_back(cols);
		return true;
	}

	// generate all of the valid placements for the row and continue to the next
	// valid placements include all cols that are not in the col, and not in the same diagonals.

	for (int i = 0; i < COLS; i++) {

		bool valid = true;

		// check if i is in cols
		for (int c = 0; c < cols.size() && valid; c++) {
			int currcol = cols[c];
			if (currcol == i) {
				valid = false;
			}

			int rowdiff = row - c;
			int coldiff = i - currcol;

			if(rowdiff == coldiff || rowdiff == -coldiff) {
				valid = false;
			}
		}

		if (valid) { 
			std::vector<int> newcols = cols;
			newcols.push_back(i);
			place_row(row+1,newcols,solutions);
		}
	}

	return false;
}

int main() {
	
	std::vector<std::vector<int>> solutions;
	std::vector<int> cols;

	place_row(0, cols, solutions);

	for (int i = 0; i < solutions.size(); i++) {
		for (int j = 0; j < ROWS; j++) {
			std::cout << solutions[i][j] << " ";
		}
		std::cout << std::endl;
	}

	return 0;
}
