---
title: "Replication: GMM in model of consumption and portfolio choices (1982)"
collection: python
permalink: /python/2021-01-27-paper-title-number-1
date: 2021-01-27
citation: 'Arcidiacono, Peter, and Robert A. Miller. &quot;Generalized instrumental variables estimation of nonlinear rational expectations models.&quot; <i>Econometrica  (1982): 1269-1286.</i>'
---

This is part of material when I worked as a graduate TA for General Econometrics (Ph.D level)

The whole point of this exercise is to understand each steps so that you can report estimated parameters and their standard errors. To do
so,

1) We need to derive what is criterion function $J$\
2) Derive analytic population moment\
3) Obtain sample moment\
4) Load data set and estimate parameter

**Step 1) Let’s define criterion function $J(\beta, \alpha)$ with
population moment**:

$$argmax_{\beta, \alpha} \ J(\beta, \alpha) = E ({g}(\beta, \alpha)) ' \ W \ E ({g}(\beta, \alpha))$$

In sample moments:

$$argmax_{\beta, \alpha} \ J(\beta, \alpha) = T \times \bar{g}_T(\beta, \alpha) ' \ W \ \bar{g}_T(\beta, \alpha)$$

where

-   $\beta$ is a discount parameter that we want to estimate.

-   $\alpha$[^1] is a risk aversion parameter that we seek to recover.
    Theory tells us $\alpha \in [0,1]$

-   $\bar{g}_T$ is the sample moment.

-   $W$ is an weighting matrix. (Identity matrix in this problem set)

Equation (1) tells us that if we have sample moment ($ \bar{g}'_T$) and
$W$, we can recover parameter $\in (\beta, \alpha)$ by standard
optimization tool box. Since we already know that we will use $W = I$
(Identity matrix), the next step is to figure it out $E(g)$ with sample
moment $\bar{g}_T$.\

**Step 2) Derive analytic population moment**:

To obtain population moment, we can appeal to either data generating
process, or economic theory. In this problem set, we start with Euler
equation such that

$$E_t [r_{t+1, j} \  MRS_{t+1}] = 1$$

where $j=1,...M $ asset (which is a portfolio), $t=1,..T$ denotes
monthly observations.

Replace MRS with substitution between $c_t$ and $c_{t+1}$ gives us:

$$\begin{aligned}
    E_t \Bigg[ r_{t+1, j} \  \beta \frac{u'(c_{t+1})}{u'(c_t)} \Bigg] &  = 1 \\
    E_t \Bigg[ \beta r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}   -1 \Bigg] &  = 0\end{aligned}$$

Note that we have subscript $t$ over Expectation. $(E_t (\cdot))$ Let
$X^*_t$ denote all information set up to $1,... t$, we can write as
conditional expectation form:

$$E \Bigg[ \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)} -1 \Bigg| X^*_t \Bigg] = 0$$

Note that we drop subscript $t$ anymore $E(\cdot)$.

Then we can use subset $X_t \subset X^*_t$ such that:

$$E \Bigg[ \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1 \Bigg| X_t \Bigg] = 0$$

Our next goal is to convert “conditional expectation” into
“unconditional expectation”.

Claim:

$$E \Bigg[ \Bigg( \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1 \Bigg) \otimes X_t \Bigg] = 0$$

where $\otimes$ represents the kronecker product.

Proof) Take law of iterated expectation over equation (7) gives us[^2]:

$$\underbrace{E \Bigg[ \Bigg( \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1 \Bigg) \otimes X_t \Bigg]}_\text{equation (7)} \ = E \Bigg(   E \Bigg[ \Bigg( \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1 \Bigg) \otimes X_t \Bigg| X_t \Bigg] \Bigg)$$

Appeal to distributive rule, it yields:

$$E \Bigg(   E \Bigg[ \Bigg( \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1 \Bigg) \Bigg| X_t \ \otimes \underbrace{X_t \Big| X_t}_\text{ $X_t \Big| X_t = X_t $} \Bigg] \Bigg) \ =   E \Bigg(  \ \underbrace{E \Bigg[ \Bigg( \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1  \Bigg| X_t \Bigg)}_\text{equation (6)} \ \otimes X_t \Bigg] \Bigg)$$

Note that underbrace is equal to zero by equation (6), which gives us
equation (9) is equal to zero.

Thus,

$$\underbrace{ E \Bigg[ \Bigg( \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1 \Bigg) \otimes X_t \Bigg]}_\text{equation (7)} \ = \underbrace{E \Bigg(  \ \underbrace{E \Bigg[ \Bigg( \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1  \Bigg| X_t \Bigg)}_\text{=0 by equation (6)} \ \otimes X_t \Bigg] \Bigg)   }_\text{equation (9)}   = 0$$

as we desired.

It gives us population moment as:

$$E \Bigg[ \Bigg( \beta  r_{t+1, j} \   \frac{u'(c_{t+1})}{u'(c_t)}  -1 \Bigg) \otimes X_t \Bigg] = 0$$

**Step 3. Derive sample moment**\

In practice, information set ($X_t$) is not necessarily “economically
exogenous”. If we use lagged values from $\frac{c_{t+1}}{c_t}$ and
$r_{t+1, j}$, these would be valid instruments.

Suppose that we use information set as two lagged values for non durable
goods, and 1 year treasury bond. $X_t$ looks like:

$$X_t =  \underbrace{\Bigg[ \frac{c_t}{c_{t-1}}, \ \frac{c_{t-1}}{c_{t-2}},\   r_{t, j}, \ r_{t-1, j}  \Bigg]}_\text{1 * 4}$$

Note that I use one dimensional consumption (non-durable goods) and one
type of return, which is 1 year treasury bond with two lags (first lag,
and second lag).

I will use CRRA utility function as:

$$\begin{aligned}
    u(c_t) &  = \frac{c_t^{1-\alpha}}{1-\alpha} \\
    u'(c_t) & = c_t^{-\alpha}\end{aligned}$$

In a summary, sample moment $\bar{g}_T$ with one dimensional consumption
(non-durable goods), one type of return $(j=1)$, and information with
two lags (consumption and 1 year treasury bonds) will be given by:

$$\underbrace{ \bar{g}_T}_\text{1 * 4} \ = \frac{1}{T} \sum_{t=1}^T \underbrace{ \Bigg( \underbrace{\Big[ \beta \ r_{t+1, j} \Big( \frac{c_{t+1}}{c_t} \Big) ^{-\alpha} -1 \Big]}_\text{1 * 1} \otimes  \Bigg[ \underbrace{ \frac{c_t}{c_{t-1}}, \ \frac{c_{t-1}}{c_{t-2}},  \ r_{t, j},\  r_{t-1, j}  \Bigg] }_\text{1 * 4} \Bigg) }_\text{ 1 $\otimes$ 4 \ = 1 * 4}$$

Criterion function $J$ with $\bar{g}_T $ will be given by:

$$\underbrace{J}_\text{1 * 1} \ = T * \underbrace{\underbrace{\bar{g}'_T}_\text{4*1} \ \underbrace{W}_\text{4*4} \ \underbrace{\bar{g}_T}_\text{1*4}}_\text{1*1}$$

**Normalization**

When I construct $\frac{c_{t+1}}{c_t}$, I use the following
normalization:

$$c_t = \frac{ \text{Personal Consumption Expenditures in Nondurable Goods} * 1,000,000,000 * (100/ \text{Price index in 2019})}{\text{Population} * 1,000}$$

Since Personal Consumption Expenditures in Nondurable Goods expressed as
total consumption with that year, we need to normalize by 1) current
dollar values 2) per capita

[^1]: $\gamma = 1-\alpha$ in the paper, I will keep use $\alpha$ from
    now.

[^2]: I do not use properties that equation (7) is equal to zero
