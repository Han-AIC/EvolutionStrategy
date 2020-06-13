# Covariance Matrix Adaption - Evolution Strategy

Making use of CMA for adaptive sigma when generating populations from progenitors.
This experiment will be differentiated from standard implementation by:

* Use of a number of progenitors intentionally divided equally throughout the
parameter search space, in order to produce a number of micro populations which
can independently evolve.

* Choice of either averaging params from the elite group after population has
been evaluated (Original Implementation), or of recombining and perturbing elite group params stochastically.

## Algorithm Outline

1. Along the param space, generate a number of progenitors with binned params.
Perturb params with gaussian noise to break up uniformity (Scale of perturbation is a hyperparam)

2. For each progenitor, generate a population by using each progenitor parameter as the mean of a multivariate gaussian. To clarify; each parameter is sampled from its own multivariate gaussian distribution. Stepsize/sigma determined by covariance matrix of params.

  * We maintain a mean and standard deviation for each parameter for each separate population. Information for each population can be stored separately as a dictionary.
  ![](./readme_imgs/meansigma.jpg)

  * Covariance is used as the step size for the sampling of each param.

  * How do we make use of the information inherent in bad examples? These should serve as warnings of solutions to avoid, it seems wasteful to throw them away.


3. Evalaute each population, take elite group, either average or recombine to produce a new mean for each population's params.

4. Use new means to generate new populations.

5. Repeat 3 and 4 some number of times (Hyperparam).

6. Take the elites from each population, group them by similarity of params (L2 or L1 distance) (Number of groups depends on a hyperparam which will decide threshold of distance needed to be classified as same group)

7. Take the elites of each group, use them to generate new populations. The hope is that the number of groups and valid subpopulations will gradually reduce as convergence nears. The idea being that elite examples will gradually converge closer and closer, or that multiple equally well-performing but substantially different solutions will converge and settle in drastically different areas of param space.

8. Repeat 3-7 until the elites reach an acceptable level of performance.
