# Lab 3: Nim

#### See [notebook](https://github.com/paolodragol/computational-intelligence-22-23/blob/main/lab3/lab3_nim.ipynb) for problem specification and requirements

## **Task 3.1: An agent using fixed rules based on nim-sum (i.e., an expert system)**

For this task, I used the basic implemetation given by the professor of Nim class. Some additional methods were added, such as display_board and set_winner, but no major changes were made.

Some utility functions were defined:
- calc_nimsum: computes the nimsum of a Nim game at a given state.
- all_ones: boolean function that checks if all the rows of a game are of dimension equal to one.
- game_over: checks if a terminal state has been reached

### Expert Strategy

This is the strategy of an expert player (i.e knows optimal strategy to win and, if possibile, depending on its starting turn, should always win).

- Initially, the bounding of k is taken into account. Some strategy is used but I don't explain it as it was removed from the requirements and may lead to confusion. In further considerations *k* is always `None`. 

- Afterwards, some final scenarios are considered in order to optimize the strategy to some extent.

- Finally, the **nimsum** is considered and used in order to execute the optimal move, if possible. Otherwise, removes one element from the longest row (the idea of this non optimal strategy is to leave as much objects as possible for the opponent to make a mistake).

## **Task 3.2: An agent using evolved rules**

#### Other Hard-Coded Strategies

In order to address the task 3.2, other hard coded strategies are implemented. In the first place, I created two semi-optimal strategies, but are too good to be considered in an evolutionary scenario. I, therefore, considered other options while using the "cooking" function for the board status implemented in class. 

#### 1. Pure Random 

A strategy that makes random choices.

#### 2. Level Zero 

This strategy has the goal to enable the opponent to win, trying to make the opposite of the optimal moves. 

#### 3. Level Two

This strategy on average wins agains *Pure Random*, thus, it is called level two (*Pure Random* is considered "level one").
- It takes into account almost final scenarios (where all rows have one or no elements and only one has more than one element).
- It also wins if only one row is left.
- In other scenarios tries to make some "silly" decisions.

#### 4. Level Three

As level 2 stategy, but:
- Aims to reach the configuration to use "final scenario" conditions (i.e tries to avoid the silly decisions of "Level Two").

#### 5. Expert

The expert strategy of task 3.1.

#### 6. Human

This is a function that aims to enable a human player to play the game.

#### Play match

A simple function that, given two strategies, plays a game in the order the strategies are received as parameters.

### **Evaluation**

In order to take into account the fact that the winning strategy depends on who starts first, this funtion plays `NUM_MATCHES` both starting first and second and returns the evaluation of both initializations. 

### **Evolution**

In order to evolve rules, the most important ones were organized and ordered in a `Rule` class (initially, also the **nimsum** was considere, but later on removed to simulate a more realistic scenario).

#### Make Strategy

This function creates a strategy which can evolve:
- The order in which rules are considered is defined by one of the genome's genes.
- The rules are then considered in such order but fire away depending on other genes.
- If no rule was fired, a "conservative" strategy is used (not evolvable).

#### Fitness and Genetic Operators

I had the idea to use both ES and GA to solve this problem. Due to lack of time I only used ES, but I left these functions hoping to improve the solution or try other approaches.

The fitness is used considering the *Evaluation* metric. 

#### Adaptive (μ+λ)-ES

Finally, this evolutionary strategy used is adapted from the professor's solution to the *rastrigin* problem. 

I tried to evolve the population in a progressive way, from the easiest strategy as an opponent, to the toughest. 

### **Solution**

Evolving the population, the results against the different strategies were:

| Strategy          | Best Fitness  | 
| ----------------- | ------------- |  
| level_zero        | 1.0           |  
| pure_random       | 1.0           |  
| level_two         | 0.725         |
| level_three       | 0.6125        |
| expert_strategy   | 0.0           |


## **Task 3.3: An agent using minmax**

The solution proposed for the *minmax* strategy is completely based on the solution proposed by [Real Python](https://realpython.com/python-minimax-nim/). 

The `nim_utils.py` file imports the already defined functions and classes in the first tasks.  

Some modifications to the solution proposed by the authors were made to try to optimize the problem, given the high dimensionality and branching factor of Nim. 
- As a first step, as suggested, I used `@cache` in order to memorize already calculated steps. 
- Secondly, when calculating the possible new steps, tuples were organized and added to a set in order to avoid considering symmetrical solutions.
- Finally, I defined an empirical *MAX_DEPTH* in order to avoid going too deep in the decision tree, when the tree becomes considerably large. This depth was determined by trying different values and selecting one which provided a good tradeoff between the solution found and the computational cost. It may be improved with a more deterministic approach. 

The **minmax_strategy** aims to use the minmax approach to play the game as in points 3.1 and 3.2. After running the `evaluate` method agains some of the hard-coded strategies with different Nim sizes, the results we obtain are the ones expected:


| Opponent Strategy | Starting (first, second)  | Nim Size                  |
| ----------------- | ------------------------- | ------------------------- |  
| pure_random       | (1.0, 1.0)                | 6                         |
| level_three       | (1.0, 1.0)                | 5                         |
| expert_strategy   | (1.0, 0.0)                | 6                         |
| ----------------- | ------------------------- | ------------------------- |  
| pure_random       | (1.0, 1.0)                | 10                        |
| level_three       | (1.0, 1.0)                | 7                         |
| expert_strategy   | (0.0, 0.0)                | 8                         |


### **References**

* *Nimsum* theory and approaches: [Wikipedia: Nim](https://en.wikipedia.org/wiki/Nim)
* *Adaptive (μ+λ)-ES* was taken from **rastrigin** algorithm of profesor Squillero: [github.com/squillero](https://github.com/squillero/computational-intelligence/blob/master/2021-22/rastrigin.ipynb)
* Collaboration and ideas were shared with: *Giovanni Genna* and *Krzysztof Kleist*. 
