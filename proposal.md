# Symbolic vs. Generative AI

### Authors' Information

**Jacopo Andrea Bianco (main contact reference)**  
Student ID: 671859 · j.bianco@studenti.unipi.it

**Alessandro Fiumanò**  
Student ID: 674559 · a.fiumano1@studenti.unipi.it

---
## The problem
Our objective is to study how pre-trained LLMs solve search problems, investigating whether they can reproduce the behavior 
of classical search algorithms.
We will prompt an LLM to solve the search problem using a specified algorithm, for instance:
"Solve the following search problem using BFS. Output your solution represented as an ordered sequence of nodes from the
initial state to the goal."
The environment will be FrozenLake from the Gymnasium library. We will provide the LLM with a description of the environment
and allow it to interact with it through function calling.
The available functions will provide the information needed to understand and explore the environment, such as its initial
state, available actions, and state transitions, without exposing any implementation of the search algorithms.

## The algorithms
We will compare our own implementations of classical search algorithms with the solutions produced by the LLM.
For each algorithm, the following procedure will be performed:

1. Implement the algorithm and integrate it with the Gymnasium environment.
2. Validate the correctness of the implementation by comparing its results with those of an existing implementation, where available.
3. Query the LLM to solve the same problem using the specified algorithm, allowing it to explore the environment through function calling.
4. Collect the LLM solution and compare it with the results of our implementation.

*Jacopo Bianco will address DFS and A\*, while Alessandro Fiumanò will address BFS and Uniform Cost Search. Both will
contribute to the script that manages interactions with the selected LLM.*

## Research question
We aim to answer the following research question: how do the solutions produced by LLMs compare with those obtained from
our implementations of classical search algorithms in terms of correctness and optimality?

## Extra
Which search algorithm does an LLM choose when asked to solve a problem without specifying a particular algorithm? Does 
it return a correct / optimal solution?