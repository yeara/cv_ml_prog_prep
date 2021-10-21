class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& A) {
        
        int n = A.size();
        if (n==0) return 0;
        
        vector<vector<int>> dp(n,vector<int>(n,0));
        
        dp[0] = A[0];
        
        for (int row = 1; row < n; row++) {
            for (int col = 0; col < n; col++) {
                int m; 
                
                if (col > 0 && col < n - 1) {
                    m = min(min(dp[row-1][col-1],dp[row-1][col]),dp[row-1][col+1]);
                }
                else if (col == 0) {
                    m = min(dp[row-1][col],dp[row-1][col+1]);
                }
                else {
                    m = min(dp[row-1][col-1],dp[row-1][col]);
                }
            
                dp[row][col] = A[row][col] + m;
            }
        }
        
        for (auto c : dp[n-1] ) {
            cout << c << " ";
        }
    
        return *min_element(dp[n-1].begin(), dp[n-1].end());
        
    }
};