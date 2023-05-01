# Inferring Implicit Trait Preferences for Task Allocation in Heterogeneous Teams

### Link to Paper: [https://arxiv.org/abs/2302.10817]{https://arxiv.org/abs/2302.10817}
### Advisor: Harish Ravichandar						 

In this research, we address the multi-robot task allocation (MRTA) problems. Within MRTA problems will focus on the coalition formation problem (ST-MR-IA) in which a team of robots needs to be distributed into coalitions (teams) such that multiple concurrent tasks can be successfully carried out [1]. We will be particularly focusing on the coalition formation for heterogeneous teams, where species of robots have different abilities. 

We look at the coalition formation problem as a scheduling problem where tasks are assigned to agents such that tasks are completed. We will subscribe to the trait-based framework where we look at a set of traits that represent the capabilities of the robot agents and the requirements (approximate) of the tasks. The trait-based framework will generalize the agents and tasks, making it easy to introduce new teams without any additional training of the model. We will be reasoning the traits space on the species-level that is the trait composition changes between species of robots instead of each individual robot agent. 

The prior work of extracting the set of approximated strategies to optimize the assignment matrix [1] utilizing the total trait space which can have non-useful information. There are fundamentally two problems because of the approach: 

The approximated strategies extracted from expert demonstrations can have tendencies to propagate traits that are irrelevant to task completion. This can negatively impact the optimizing of assignment matrix .

The time complexity and computations to optimize assignment matrix are higher because of the usage of large trait space of approximated strategies. 

**Our approach satisfies the task requirements by optimizing robot assignments using the trait preferences.** We are trying to satisfy the task requirements using as few traits as possible to reduce the time to optimize the new assignment matrix for a new team. We will be trying to extract the trait preferences by observing the underlying variances and relationships between traits. 

 

A few areas to focus on include: 

1. **Variances**: Looking at the trait space, we can see that the larger variance in a particular trait when tasks are successfully completed shows that a particular trait is less critical to optimize as there are higher chances for it to get fulfilled. The claim is that we need to find the traits with lower variance and give higher preference. The logical flaw for the argument is the natural state of the trait having higher variance. We propose that finding the variance behaviors of the approximated strategies traits and validating if the agent trait space carries the similar variance behaviors will give us a better understanding of the trait. 

2. **Dimensionality**: Focusing on reducing dimensionality will become a critical component of our work as it will help us increase the effectiveness of optimizing the assignment matrix. We will be looking into PCA and trying to see if we can factor out lower variance dimensions and extract the linear relationship from those. The relationship between the traits will be critical in reducing the dimension while preserving their importance. 

The main evaluation will be task completion and computation time when compared to the approach of task allocation without trait preferences. We will also compare our framework with partial implementations of the idea to show each component added to our solution has a vital role. 

We will be dividing the total project into three parts: 

*  Numerical experiment setup 

*  Implementation of ideas 

*  Testing on real world data 

For the numerical experiment setup, we will be focusing on creating a vanilla dataset of expert demonstrations of task allocations for the given agent traits. In two weeks, once the data is ready, we will move on to implementing the ideas of creating the trait preferences and evaluating the performance of the algorithm. The second step will be repeated until the calculated trait preferences are close enough to the expected preferences of the experiment. We will move on to finally implementing the trait-preference algorithm onto a few datasets of task allocation which are available and possibly deploy the model onto a scenario on the Robotarium. 


References: 

[1] Srikanthan, A., & Ravichandar, H. (2021). Resource-Aware Generalization of Heterogeneous Strategies for Coalition Formation. arXiv preprint arXiv:2108.02733. 

