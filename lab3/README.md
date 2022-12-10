# **Lab 3: Nim**

#### See [notebook](https://github.com/paolodragol/computational-intelligence-22-23/blob/main/lab3/lab3_nim.ipynb) for problem specification and requirements

## **Task 3.1: An agent using fixed rules based on nim-sum (i.e., an expert system)**

For this task, I used the basic implemetation given by the professor of Nim class. Some additional methods were added, such as display_board and set_winner, but no major changes were made.

Some utility functions were defined:
- `calc_nimsum`: computes the nimsum of a Nim game at a given state.
- `all_ones`: boolean function that checks if all the rows of a game are of dimension equal to one.
- `game_over`: checks if a terminal state has been reached.

### Expert Strategy

This is the strategy of an expert player (i.e knows optimal strategy to win and, if possibile, depending on its starting turn, should always win).

- Initially, the bounding of k is taken into account. Some strategy is used but I don't explain it as it was removed from the requirements and may lead to confusion. In further considerations *k* is always `None`. 

- Afterwards, some final scenarios are considered in order to optimize the strategy to some extent.

- Finally, the **nimsum** is considered and used in order to execute the optimal move, if possible. Otherwise, removes one element from the longest row (the idea of this non optimal strategy is to leave as much objects as possible for the opponent to make a mistake).

## **Task 3.2: An agent using evolved rules**

#### ***Other Hard-Coded Strategies***

In order to address the task 3.2, other hard coded strategies are implemented. In the first place, I created two semi-optimal strategies, but are too good to be considered in an evolutionary scenario. I, therefore, considered other options while using the "cooking" function for the board status implemented in class with the professor. 

#### 1. *Pure Random*

A strategy that makes random choices.

#### 2. *Level Zero*

This strategy has the goal to enable the opponent to win, trying to make the opposite of the optimal moves. 

#### 3. *Level Two*

This strategy on average wins agains *Pure Random*, thus, it is called level two (*Pure Random* is considered "level one").
- It takes into account almost final scenarios (where all rows have one or no elements and only one has more than one element).
- It also wins if only one row is left.
- In other scenarios tries to make some "silly" decisions.

#### 4. *Level Three*

As level 2 stategy, but:
- Aims to reach the configuration to use "final scenario" conditions (i.e tries to avoid the silly decisions of "Level Two").

#### 5. *Expert*

The expert strategy of task 3.1.

#### 6. *Human*

This is a function that aims to enable a human player to play the game.

#### **Play match**

A simple function that, given two strategies, plays a game in the order the strategies are received as parameters.

### **Evaluation**

In order to take into account the fact that the winning strategy depends on who starts first, this funtion plays `NUM_MATCHES` both starting first and second and returns the evaluation of both initializations. 

### **Evolution**

In order to evolve rules, the most important ones were organized and ordered in a `Rule` class (initially, also the **nimsum** was considere, but later on removed to simulate a more realistic scenario).

#### `Make Strategy`

This function creates a strategy which can evolve:
- The order in which rules are considered is defined by one of the genome's genes.
- The rules are then considered in such order but fire away depending on other genes.
- If no rule was fired, a "conservative" strategy is used (not evolvable).

#### **Fitness and Genetic Operators**

I had the idea to use both ES and GA to solve this problem. Due to lack of time I only used ES, but I left these functions hoping to improve the solution or try other approaches.

The fitness is used considering the *Evaluation* metric. 

#### **Adaptive (μ+λ)-ES**

Finally, this evolutionary strategy used is adapted from the professor's solution to the *rastrigin* problem. 

I tried to evolve the population in a progressive way, from the easiest strategy as an opponent, to the toughest. 

### **Results (tasks 3.1, 3.2)**

Evolving the population, the results against the different strategies were:

| Strategy          | Best Fitness  | 
| ----------------- | ------------- |  
| level_zero        | 1.0           |  
| pure_random       | 1.0           |  
| level_two         | 0.725         |
| level_three       | 0.6125        |
| expert_strategy   | 0.0           |

The results are encouraging and I expected worse values for the metric.

## **Task 3.3: An agent using minmax**

The solution proposed for the *minmax* strategy is completely based on the solution proposed by [Real Python](https://realpython.com/python-minimax-nim/). 

The `nim_utils.py` file imports the already defined functions and classes in the first tasks. The file `lab3_nim_minmax.ipynb` contains the core code for this solution.

Some modifications to the solution proposed by the authors were made to try to optimize the problem, given the high dimensionality and branching factor of Nim. 
- As a first step, as suggested, I used `@cache` in order to memorize already calculated steps. 
- Secondly, when calculating the possible new steps, tuples were organized and added to a set in order to avoid considering symmetrical solutions.
- Finally, I defined an empirical *MAX_DEPTH* in order to avoid going too deep in the decision tree, when the tree becomes considerably large. This depth was determined by trying different values and selecting one which provided a good tradeoff between the solution found and the computational cost. It may be improved with a more deterministic approach. 

### **Results (task 3.3)**

The **minmax_strategy** aims to use the minmax approach to play the game as in points 3.1 and 3.2. After running the `evaluate` method agains some of the hard-coded strategies with different Nim sizes, the results we obtain are the ones expected:


| Opponent Strategy | Starting (first, second)  | Nim Size                  |
| ----------------- | ------------------------- | ------------------------- |  
| pure_random       | (1.0, 1.0)                | 6                         |
| level_three       | (1.0, 1.0)                | 5                         |
| expert_strategy   | (1.0, 0.0)                | 6                         |
|                   |                           |                           |  
| pure_random       | (1.0, 1.0)                | 10                        |
| level_three       | (1.0, 1.0)                | 7                         |
| expert_strategy   | (0.0, 0.0)                | 8                         |


However, the computational cost for this solution is very high. 

## **Task3.4: An agent using reinforcement learning (RL)**

To use a RL approach the solution provided from [Towards Data Science](https://towardsdatascience.com/hands-on-introduction-to-reinforcement-learning-in-python-da07f7aaca88) was adapted to the Nim problem.

The environment and agent implementation are found in `nim_environment.py` and `rl_agent.py` respectively. 

### **Environment**

The class `NimBoard` has the `Nim` class of the previous classes as a basis. Some additional funtions are defined or modified in order to create an RL environment suitable for Nim. 

The most important ones are:
- `construct_allowed_states`: creates all possible states (trying to avoid symmetric ones).
- `get_state_and_reward`: returns current board state (observation, which in this case is the state) and the reward.
- `give_reward`: gives a -1 for any non terminal move and +2 for winning a game, while -2 for a loosing it.

### **Agent**

Also the Agent was adapted to the Nim game. The functions were adapted to the problem's states and actions. Some functions were added to keep track of the results o the agent:

- `update_results`, `get_avg_wins`, `reset_results`: aim at keeping track of how the agent is performing (the average wins is the performance metric considered). 

### **Reinforcement Learning**

The core code for this solution is found in  the  file `lab3_nim_rl.ipynb`. The `reinforcement_learning_nim` function has the goal to return an agent which uses RL in order to play Nim, by receiving a strategy (one of the hard-coded ones seen before) and a Nim size. As an additional parameter, an agent can be received so the agent needs not to learn from scratch. This was implenented taking the following results into account.

### **Results (task 3.4)**

The following approaches were evaluated for Nim size equal to three and five. 

Initially, I ran the approach against all the hard-coded strategies, initializing an agent from scratch: 


| Opponent Strategy | Avg wins over (50, 9950)  | Nim Size                  |
| ----------------- | ------------------------- | ------------------------- |
| level_zero        | (0.784, 0.979)            | 3                         |  
| pure_random       | (0.784, 0.924)            | 3                         |
| level_two         | (0.216, 0.874)            | 3                         |
| level_three       | (0.157, 0.860)            | 3                         |
| expert_strategy   | (0.059, 0.444)            | 3                         |
|                   |                           |                           | 
| level_zero        | (0.784, 0.982)            | 5                         |  
| pure_random       | (0.686, 0.912)            | 5                         |
| level_two         | (0.491, 0.924)            | 5                         |
| level_three       | (0.157, 0.873)            | 5                         |
| expert_strategy   | (0.000, 0.001)            | 5                         |


I noticed that the results were good for almost all of them except the expert stategy, especially in the case where the size is equal to five. I thought that it may be usefult to make the agent improve incrementally as in the task 3.2 approach. Therefore, I tried to pass to the following strategy an the agent received from the preivous RL phase:


| Opponent Strategy | Avg wins over (50, 9950)  | Nim Size                  |
| ----------------- | ------------------------- | ------------------------- |
| level_zero        | (0.824, 0.981)            | 3                         |  
| pure_random       | (0.979, 0.973)            | 3                         |
| level_two         | (0.973, 0.982)            | 3                         |
| level_three       | (0.982, 0.987)            | 3                         |
| expert_strategy   | (0.986, 0.889)            | 3                         |
|                   |                           |                           | 
| level_zero        | (0.803, 0.911)            | 5                         |  
| pure_random       | (0.978, 0.963)            | 5                         |
| level_two         | (0.963, 0.975)            | 5                         |
| level_three       | (0.974, 0.980)            | 5                         |
| expert_strategy   | (0.979, 0.812)            | 5                         |

This results are impressive, in my opinion. I notice some inversion trend in the performance from one strategy to the other which may be seen in the plots, and I do not understand it, but I think it is still remarkably better than before.

I believe the solution proposed can be improved (in many ways...). For instance, considering both starting first and second during the RL phase may be beneficial. Moreover, the lack of improvement using the incremental approach may be analyzed. A more suitable reward approach may also be taken into account. I tried to use an approach that had no idea of the problem or its solutions.  

### **References**

* *Nimsum* theory and approaches: [Wikipedia: Nim](https://en.wikipedia.org/wiki/Nim)
* *Adaptive (μ+λ)-ES* was taken from **rastrigin** algorithm of profesor Squillero: [github.com/squillero](https://github.com/squillero/computational-intelligence/blob/master/2021-22/rastrigin.ipynb)
* The *MinMax* approach: [Real Python](https://realpython.com/python-minimax-nim/)
* *RL* approach modified and corrected by Andrea Calabrese: [Towards Data Science](https://towardsdatascience.com/hands-on-introduction-to-reinforcement-learning-in-python-da07f7aaca88)   
* Collaboration and ideas were shared with: *Giovanni Genna* and *Krzysztof Kleist*. 
