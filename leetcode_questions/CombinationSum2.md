class Solution {
public:
    
    vector<vector<int>> combinations;
    vector<int> combination;
        
    
    void combinationSum2(vector<int>& candidates, int target, int start) {
        if (target == 0) {
            combinations.push_back(combination);
            return;
        }
        if (target < 0) return;
        
        for (int i = start; i < candidates.size(); i++) {
            if (i > start && candidates[i] == candidates[i-1]) continue;
            combination.push_back(candidates[i]);
            combinationSum2(candidates,target-candidates[i],i+1);
            combination.pop_back();
        }
    }
    
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {      
        sort(candidates.begin(), candidates.end());
        combinationSum2(candidates,target,0);
        return combinations;        
    }
};