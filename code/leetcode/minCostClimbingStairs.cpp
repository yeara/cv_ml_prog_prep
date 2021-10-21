class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        
        if (cost.size() == 0 ) return 0;
        if (cost.size() == 1 ) return cost[0];
        
        vector<int> dp(cost.size(),0);
        dp[0] = cost[0];
        dp[1] = cost[1];
            
        for (int i = 2; i < cost.size(); i++) {
            dp[i] = min(dp[i-2], dp[i-1]) + cost[i]; 
        }   
        
        int c = cost.size();
        
        return min(dp[c-1],dp[c-2]);
    }
};