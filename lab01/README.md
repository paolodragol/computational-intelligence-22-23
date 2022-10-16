# Lab 1: Set Covering

### **Problem representation:**
We need the least amount of numers (given *n* lists *S*), that cover all the numbers $[0, N-1]$:

$$
min \sum \limits _{j=1} ^{n} c_{j}x_{j}
$$

where $c_{j}$ is the cost of the given list and 

$$
x_{j} \ for \ list \ S_{j} \ = 
  \begin{cases}
    1 & \text{if $S_{j}$ selected}\\
    0 & \text{if not}\\
  \end{cases} 
$$

The cost of adding a list to the solution is the number of elements in the list. As priority of node expansion we use the length of the list itself. 

### Constraints
Given the problem code, we can see that:
* Lists are of length $[N//5, N//2]$
* The number of lists to explore is between $[N, N*5]$

**Linear programming problem contraints:**
* Every element in the solution space must be covered. *This constraint cannot be guaranteed.*
* Values for $x_{j} \in \{0, 1\}$ 
**NP-hard optimization problem.** 

## **Personal solution**

The solution proposed is based on professor's Squillero *Greedy algorithm* and it aims to implement an **A\* approach** by adding an additional heuristic to the cost considered (the amount of elements in the current state). The *heuristic* function is based on the amount of elements missing from the Solution Space. 

In order to apply such heuristic, we need a Priority Queue and we used the one implemented by professor Squillero in the file `gx_utils.py`. The priority function calculates the number of missing elements in the current state and adds it to the number of elements in it. If priorities are equal, the length of the current state is passed as a second criterion of priority. 

What the algorithm does is that, after checking validity of the current state, it extracts the most promising element (assuming it is the optimal one) from the problem lists and adds it to the current state. At each iteration it recomputes the priorities of all the elements in the problem's remaining lists. Finally, it checks if the added list satisfies the goal condition. 

#### *Note on visited nodes*

In the solution, the amount of visited nodes is the amount of elements considered when executing the verification function (in this algorithm: `covered != goal`), and not the amount of elements for which the priority is computed. If this is wrong, and a node can be considered explored when computing its priority, the number of visited nodes should be multiplied by the length of the problem lists.

## Solution

For the proposed algorithm, the solution results are:

| N    | Cost  | Visited | bloat(%) |
| ---- | ----- | ------- | -------- |
| 5    | 5     | 5       | 0        |
| 10   | 13    | 7       | 30       |
| 20   | 28    | 13      | 40       |
| 100  | 228   | 13      | 128      |
| 500  | 1828  | 27      | 266      |
| 1000 | 4130  | 26      | 313      |


## References

* Greedy algorithm and utils file used: [github.com/squillero](https://github.com/squillero/computational-intelligence/tree/master/2022-23)
* Linear Programming theoretical sorces: [Set Cover Problem (Chapter 2.1, 12), by Tamara Stern](https://math.mit.edu/~goemans/18434S06/setcover-tamara.pdf) and [Set covering problem - Optimization Wiki, by Liang, Alanazi, Al Hamoud ](https://optimization.cbe.cornell.edu/index.php?title=Set_covering_problem#:~:text=The%20set%20covering%20problem%20is,cover\)%20all%20of%20these%20elements.)
