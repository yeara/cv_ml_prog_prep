# Meeting rooms

#Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.

class Solution {
public:
    bool canAttendMeetings(vector<vector<int>>& intervals) {
        
        if (intervals.size() < 2) return true; 
        
        //sorting by inserting into a vector and sorting - n*logn
        vector<pair<int,int>> meets;
        for (auto meet : intervals) {
            meets.push_back({meet[0],meet[1]});
        }
        sort(meets.begin(),meets.end());
        
        //another option would be to add to a set, but then have to keep track
        //of forward/backward iterators
        
        //iterating over a set, imp as binary tree.
        for (int i = 1; i < meets.size(); i++) {
            if (meets[i].first < meets[i-1].second) return false;
        }
        
        return true;
        
    }
};