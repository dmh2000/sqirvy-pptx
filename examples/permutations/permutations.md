To generate all permutations of a list or sequence, one of the most common step-by-step algorithms is recursive backtracking. Here’s a detailed step-by-step approach suitable for both manual understanding and programming implementation:

Recursive Backtracking Algorithm
Start with an Input List

Have a sequence of elements (e.g., [1][2][3]).

Define a Function to Permute

The function takes the current index (start) and recursively swaps every other index (i) from start to the end of the list.

Base Case

If start reaches the end of the list (e.g., start == len(list) - 1), record the current arrangement as a permutation.

Recursive Step (for-loop and swapping)

Loop i from start to len(list) - 1:

Swap elements at indices start and i

Recursively call the function with start + 1

Swap back (backtrack) to restore original ordering before repeating with the next i.

Repeat

This process explores all possible placements for each element position, producing every unique arrangement.