# CheatSheet

### Snake and Ladders Conversion:

r = n - (li-1) / n - 1;
c = (li-1) % n;
if((n-r-1)%2) c = n - c - 1;

### Power Set

Power set size is 2^n - iterate over all combinations using an int and choose elements using bit shifts tests.

Testing with bit shift: counter & 1 << j

A neat solution was growing the power set by iterating over previous sets until the correct cardinality for each pass (copy and mult)

### Permutation Set

Cardinality of permutation set - n!

### Unique Combination Sum

Backtracking - test conditions and commit solution when entering the function

Start the search window from the current index, or next, depending on the constraints