# Lab 2: Set Covering with Genetic Algorithms

#### [See lab1 for problem representation and constraints]

## **Personal solution**

The solution proposed is based on a *Steady State GA*. The number of generations is a fixed number (best solution found was for at most 150 generations). Generations are divided into two, the first 90% are an exploratory phase in which selective pressure is not very high, but there is still some (given that the population size is big). The second stage aims at increasing exploitation, by setting a parent selection with a tournament of *N/2*. 

During exploration, crossover is preferred (still using randomness, but mutating only 30% of the time). The explotation phase tends to prefer mutations with a probability of 80%.

### **Encoding and Representation**

The solution considers a population size which is proportional to the problem size **N**. Each individual is a `True` / `False` list, of length **N** and each element represents the presence or not of a given list in the set of problem lists (these are made unique at the beginning of the algorithm).

#### ***Fitness***

Fitness is condidered as a tuple. The first element is the amount of elements covered by the individual, and the second is the negative of the amount of elements in all the lists of the individual. 


#### ***Parent selection***

Parent selection is used as a tournament and is implemented as explained in the intoduction, with respect to exploration vs exploitation phases.

#### ***Mutations***

Mutations are random changes to one of the elements of an individual. This means that at each mutation, a list is added or removed from a given individual. 


#### ***Crossover***

Different crossovers are meant to be implemented, with the same probability, of about 33% each. The first two are the same, definining a cutting point and taking one half from one parent and the other from the other parent, but differ on which part to take (*this may be useless but improved a bit the solution so I left it there*). The third type executes a uniform crossover, taking, locus by locus, one gene from one parent and one from the other. 


### **Results**

The following results are obtained with a *POPULATION_SIZE=5\*problem_size* (where problem size is the number of unique lists in the problem), *OFFSPRING_SIZE=1000* and *150 generations*, lower numbers are obtained only for low **N** where the algorthims stops if it find optimal solution with number of elements in the individual equal to *N*.

| N    | Cost  | Generations | bloat(%) |
| ---- | ----- | ----------- | -------- |
| 5    | 5     | 1           | 0        |
| 10   | 10    | 4           | 0        |
| 20   | 23    | 150         | 15       |
| 100  | 199   | 150         | 99       |
| 500  | 40449 | 150         | 7989.8   |
| 1000 | 194090| 150         | 19309    |

This solution does not scale to big numbers, as it can be seen for larger numbers of *N*, despite tweaks and adjustments made moslty on the generic operators. Also runnig time grows exponentially. A better solution might be found for a better definition of the *fitness* (I tried some but non were effective).

I readjusted the parameters to get faster solutions for bigger dimensionalities, but, obviously, results were worse. However, it seemed that this dimensionalities had better improvements during the exploitation phase and therefore it was set to 40% (defined by the explore paramenter). With *POPULATION_SIZE=problem_size*, *OFFSPRING_SIZE=500* and *60 generations* the results were:

| N    | Cost  | Generations | bloat(%) |
| ---- | ----- | ----------- | -------- |
| 500  | 81179 | 50          | 16135.8  |
| 1000 | 411562| 50          | 41056.2  |

### **References**

* Inspiration was taken from **one_max** algorithm of profesor Squillero [github.com/squillero](https://github.com/squillero/computational-intelligence/tree/master/2022-23)
* Collaboration and ideas were shared with Giovanni Genna. 
