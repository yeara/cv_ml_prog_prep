#include <math.h>
class Solution {
public:
    /*
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> subs;
        vector<int> sub;
        subs.push_back({});
        for (int i = 0; i < nums.size(); i++) {
            int presize = subs.size();
            for (int j = 0; j < presize; j++) {
                vector<int> n = subs[j];
                n.push_back(nums[i]);
                subs.push_back(n);
            }
        }
        
        return subs;
    }*/
    
    vector<vector<int>> powerset;
    vector<int> sub;
    
    void backtrack(vector<int>& nums, int j) {
        
        powerset.push_back(sub);
        
        for (int i = j; i < nums.size(); i++) {
            sub.push_back(nums[i]);
            backtrack(nums,i+1);
            sub.pop_back();
        }
    }
    
    void iterate(const vector<int>& nums) {
        int d = nums.size();
        int setcount = pow(2,nums.size());
        
        for (int i = 0; i < setcount; i++) {
            vector<int> sub;
            //cout << "iter " << i << endl;
            for (int j = 0; j < d; j++) {
                //cout << i << " " << 1 << j << " " << (i & 1 << j) << endl;
                if ( i & 1 << j) {
                    sub.push_back(nums[j]);
                }
            }
            
            powerset.push_back(sub);
        }
    }
    
    vector<vector<int>> subsets(vector<int>& nums) {
        backtrack(nums,0);
        //iterate(nums);
        return powerset;
    }
};