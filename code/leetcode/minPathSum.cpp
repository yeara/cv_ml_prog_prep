class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        
        int rows = grid.size();
        int cols = grid[0].size();
        
        vector<vector<int>> dp(rows, vector<int>(cols,0));
        
        
        dp[0][0] = grid[0][0];
       
        for (int i = 1; i < rows; i++) {
            dp[i][0] = grid[i][0]+dp[i-1][0];
        }
        
        for (int i = 1; i < cols; i++) {
            dp[0][i] = grid[0][i]+dp[0][i-1];
        }
        
        
        for (int r = 1; r < rows; r++) {
            for (int c = 1; c < cols; c++) {
                dp[r][c] = grid[r][c] + min(dp[r-1][c],dp[r][c-1]);
            }
        }
        
        return dp[rows-1][cols-1];
        
    }
};