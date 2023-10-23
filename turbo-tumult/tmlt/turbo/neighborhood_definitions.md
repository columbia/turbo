
# Notes about neighborhood definitions and DP semantics

In Turbo, we work with Bounded-DP, i.e. the size of the dataset is public and two datasets are neighbors if we can obtain one dataset from the other by *replacing* one row. Meanwhile, Tumult works only with Unbounded-DP and similar variants, where two datasets are neighbors if we can obtain one dataset from the other by *adding or removing* one (or k) row(s).

For $x,x' \in \mathbb{N}^\mathcal{X}$ we note:
- $x \sim_{B} x'$ if $|x| = |x'|$ and there exists $i$ such that for all $j \neq i$, $x_i = x_i'$ (in tuple notation).
- $x \sim_{U} x'$ if $x \Delta x' = 1$ where $\Delta$ denotes the symmetric difference.
- $x \sim_{Uk} x'$ if $x \Delta x' = k$ 

This terminology is similar to https://www.cse.psu.edu/~duk17/papers/nflprivacy.pdf or https://arxiv.org/abs/2207.10635, among others.

## Bounded DP and Unbounded DP

Our Turbo-Tumult implementation uses Pure DP only, so let's focus on this definition.

A mechanism $M_B$ is $\epsilon_B$-Bounded-DP if:
$$\forall x,x' \in \mathbb{N}^\mathcal{X} :  x \sim_{B} x', \forall r \in R, \Pr[M(x) = r] \le e^{\epsilon_B} \Pr[M(x') =r]$$
A mechanism $M_U$ is $\epsilon_U$-Unbounded-DP if:
$$\forall x,x' \in \mathbb{N}^\mathcal{X} : x \sim_{U} x', \forall r \in R, \Pr[M(x) = r] \le e^{\epsilon_U} \Pr[M(x') =r]$$
Remark that given a database $x = (x_1, \dots, x_n)$, replacing one row $x_i$ by $x_i'$ can be achieved by first removing a row $x_i$ (let's note the resulting database $\hat x$) and then adding a new row $x_i'$. Hence we have:

$$\forall x,x' \in \mathbb{N}^\mathcal{X},  x \sim_B x' \implies \exists \hat x : x \sim_U \hat x \wedge \hat x \sim_U x'$$

Thanks to group privacy, this means that an $\epsilon_U$-Unbounded-DP mechanism $M_U$ is also an $\epsilon_B$-Bounded-DP mechanism with $\epsilon_B = 2\epsilon_U$. 

The reverse implication is not true for arbitrary databases $x,x'$, since the RHS can be true for databases that do not have the same number of points.

However, if we fix the database size, then the two sides become equivalent:

$$\forall x,x' \in \mathbb{N}^\mathcal{X} : |x| = |x'|,  x \sim_B x' \iff \exists \hat x : x \sim_U \hat x \wedge \hat x \sim_U x'$$

Indeed, the first direction still holds as before. For the reverse direction, consider $x,x' \in \mathbb{N}^\mathcal{X}$ such as $|x| = |x'| = n$. Moreover, suppose that there exists $\hat x$ such that $x \sim_U \hat x \wedge \hat x \sim_U x'$.
- If $|\hat x| = n+1$, there exists $a \in \mathcal{X}$ such that $\hat x = \{x_1, \dots, x_n, a\}$ in multiset notation. Since $\hat x \sim_U x'$ and $|x'| = n$, we can obtain $x'$ by removing one element from $\hat x$. If that element is $a$, $x' = x$ and in particular $x\sim_B x'$. Otherwise, there exists $i \in [n]$ such that $x' = \{x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n, a\} = \{x_1, \dots, x_{i-1}, a, x_{i+1}, \dots, x_n\}$. Hence is that case we also have $x \sim_B x'$.
- If $|\hat x| = n-1$, the same reasoning holds and gives $x \sim_B x'$.

## Unbounded DP with k rows (UkDP)

 Tumult proposes another neighborhood definition that is closer to what we need, without scaling factors on privacy budgets like in the previous UDP $\implies$ BDP result. `AddMaxRows(max_rows=k)` protects the addition or removal of any set of (at most) $k$ rows. For short, we'll write U2 for `AddMaxRows(max_rows=2)` and note the corresponding relation $x \sim_{U2} x'$. 

First, note that we have:

$$\forall x,x' \in \mathbb{N}^\mathcal{X}, x \sim_{U2} x' \implies \exists \hat x \in  \mathbb{N}^\mathcal{X}: x \sim_{U} \hat x \wedge \hat x \sim_{U} x'$$

where the intermediate database $\hat x$ contains only one additional (or one fewer) row instead of two.

At first sight, it is not clear whether the implication holds in the other direction, because the definition U2 from Tumult's docs is quite ambiguous: "using AddMaxRows(k) will cause Tumult Analytics to hide the addition or removal of up to k rows at once from the corresponding private table." A priori, that does not cover the addition of $\ell$ rows and the removal of $k - \ell$ rows. However, the two definitions are equivalent for standard additive noise mechanisms.

We'll now suppose that $x \sim_{Uk} x' \iff |x \Delta x'| = k$ where $\Delta$ is the symmetric difference. This is more common in the DP literature (e.g. see PINQ). 

Finally, we get the following:

$$\forall x,x' \in \mathbb{N}^\mathcal{X} :  |x| = |x'|, x \sim_{U2} x' \iff x \sim_B x'$$
If we remove the constraint that $x$ and $x'$ have the same number of elements, we only get $U2 \implies B$. 

Thus, if we want to satisfy $\epsilon$-BDP, it is sufficient to satisfy $\epsilon$-U2DP (with the same epsilon). Moreover, any composition of BDP and U2DP mechanisms is BDP.

## Sensitivity of count queries

Our Turbo-Tumult integration only supports count queries, i.e. queries of the form:
```python
query1 = QueryBuilder("citibike").filter("gender = 'male'").count()
```

Since we are using Pure DP, Tumult will use the Laplace mechanism to evaluate these queries. 

For any neighborhood definition, $M: x \mapsto q(x) + Lap(\Delta_q / \epsilon)$ is $\epsilon$-DP, where $\Delta_q$ is the L1 sensitivity of $q$ for that neighborhood definition.

For a counting query $q$, note $\Delta_U, \Delta_B, \Delta_{Uk}$ the sensitivity for the corresponding neighboorhod definitions. We have:
- $\Delta_U = 1$
- $\Delta_B = 1$
- $\Delta_{Uk} = k$

Note that although U2DP implies BDP, an $\epsilon$- U2DP Laplace mechanism will add twice as much noise than an $\epsilon$-BDP Laplace mechanism.

For other queries such as histograms with at least 2 bins, we can have $\Delta_B = 2$ and $\Delta_{U2} = 2$. But this is out of scope for our implementation.
## Implementation considerations

Now, suppose that we want to implement Turbo on top of Tumult. 
The goal is to achieve BDP guarantees at the end (we have no hope of achieving U2DP or UDP without changing Turbo, because Turbo is using BDP mechanisms that rely on the exact dataset size, for instance to express accuracy targets and normalize histograms). We are fine if Tumult uses U2DP mechanisms under the hood, since it implies BDP, so we configure Tumult with  `AddMaxRows(max_rows=2)`.

First, Tumult is in charge of answering queries that Turbo is unable to address. This happens in `session.py`, using the measurement prepared initially.
- All the queries are executed by Tumult unchanged, and give a U2DP budget that is accumulated in the privacy accountant $\epsilon_G$.
- For some queries such as counts (although Turbo should handle those directly), Tumult will add more noise than required for BDP. For other queries like histograms, the noise will be just right. We stay on the safe side by sometimes adding too much noise but always guaranteeing the U2DP and thus BDP guarantees.

Turbo itself uses only two types of DP mechanisms:
- The sparse vector mechanism is implemented manually. 
	- We take the dataset size and an accuracy threshold to compute the right scale $b$ for the internal Laplace, under BDP. We sample the Laplace using Numpy to get a noisy SV threshold. Then, we spend the corresponding $\epsilon_B$ by passing it to Tumult.
	- We use Tumult's accountant to just add up epsilons: $\epsilon_{G} = \epsilon_{U2} + \epsilon_{B}$ where $\epsilon_B$ comes from Turbo calls to the accountant and $\epsilon_{U2}$ comes from Tumult's own calls to the accountant.
	- Even though Tumult originally operates under U2DP, the final DP budget guarantee $\epsilon_G$ should be interpreted under BDP as soon as at least one query used Turbo.
	- The SV initialization happens in `sparse_vectors.py`
- In case of SV failure or Bypass, the PMW-Bypass algorithm relies on a Laplace mechanism to obtain a DP result. We rely on Tumult's Laplace mechanism for that.
	- For simplicity, we follow the PMW-Bypass pseudocode (Alg 1 from the Turbo paper), with the same epsilon for both type of queries. In practice, it is possible to use different values for epsilon (Bypass can accomodate a smaller epsilon) but that complicates the implementation.
	- First, we compute the $\epsilon_{U2} = 4\Delta_{U2} \ln(1/\beta) / (n\alpha)$ necessary to achieve our desired accuracy, where $\Delta$ is the sensitivity of the original query: $\Delta_B = 1$ under BDP and  $\Delta_{U2} = 2$ under U2DP.
	- Then, we compute the transformation and the adjusted budget for a requested budget $\epsilon_B$ by using Tumult's API, in `session.py`:
		- The adjusted budget might be larger than the requested budget because of rounding logic. For simplicity, we ignore the marginal impact of this rounding on accuracy.
		- Since we passed a Pure DP budget, Tumult will prepare a measurement that adds Laplace noise $Lap(\Delta_{U2}/\epsilon_{U2}) = Lap(\Delta_B/\epsilon_B)$ 
		- What we need, in Turbo's pseudocode, is actually to add $Lap(\Delta_B/\epsilon_B)$. We simply execute the measurement prepared by Tumult with $\epsilon_{U2} = 2\epsilon_B$, but only pay $\epsilon_B$. The resulting privacy budget is correct under BDP, since the BDP sensitivity of a count is twice lower than the U2DP sensitivity.
	- Finally, we execute the transformation and spend the budget $\epsilon_B$. 

## General comments

Even though an $\epsilon$-U2DP mechanism is also $\epsilon$-BDP, using an U2DP mechanism  to guarantee BDP might be suboptimal in terms of utility. 

We already saw that U2DP counts have twice the senstivity of BDP counts. 
In other cases, BDP mechanisms can leverage the more restrictive neighborhood relation. For instance, to compute an average in U2DP we need to spend some budget to determine the number of elements, while BDP has access to the true size of the dataset by construction.

BDP and UkDP come with drawbacks, such as the lack of parallel composition, unless additional assumptions hold (e.g., two neighbors differ on the same timestamp).
