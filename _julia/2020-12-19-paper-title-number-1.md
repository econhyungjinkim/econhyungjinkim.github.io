---
title: "Replication: Monte Carlo Study in Berry (1994)"
collection: python
permalink: /julia/2020-12-19-paper-title-number-1
---



---
title: "Replication: Monte Carlo Study in Berry (1994)"
collection: python
permalink: /julia/2020-12-19-paper-title-number-1
excerpt: 'This paper is about the number 2. The number 2 is left for future work.'
citation: 'Berry, Steven T. &quot;Estimating discrete-choice models of product differentiation .&quot; <i>The RAND Journal of Economics (1994): 242-262.</i>'
---

# Fixed Coefficients Random Utility (Demand) Estimation

This notebook reviews the estimation and inference of a **linear** random utility model when the agent is facing a finite number of alternatives.

## Introduction
Consider a set of $J+1$ alternatives $\{0,1,2,...,J\}$. The utility that decision maker (DM) $i$ receives from buying projusct $j$ is
$$ u_{ij} = x_{ij}' \beta -\alpha p_j + \xi_j+ \epsilon_{ij}.$$
The Decision Maker maximizes her utility
$$y_i =\arg \min_{j} u_{ij}.$$

We now assume that $\epsilon_{ij}$ are $i.i.d.$ across DMs and across alternatives. In addition, we assume that $\epsilon_{ij}$ are distributed (standard) T1EV. We can write the following Conditional Choice Probabilities (CCP):
$$ Pr(y_i = j) = \frac{e^{x_{ij}'\beta}}{\sum_{k=0}^{J}e^{x_{ik}'\beta}}.$$

**Aggregate, market-level data** In Berry, Levinson, and Pakes (1995) and in many other empirical work following BLP the researcher observes only market-level data. That means that the characteristics vector of the alternatives is not indexed by $i$. The variation in product characteristics are unobserved and get absorbed by the error term $\epsilon$. The choice probabilities become
$$ Pr(y = j|x_j, \xi_j; \beta) \ \text{for} \ j=0,,,J, = \frac{e^{x_{j}'\beta}}{\sum_{k=0}^{J}e^{x_{k}'\beta}}.$$
The left-hand side is simply the market share of product/alternative $j$. We will denote these market shares as $s_0,s_1,...,s_{J-1}$. The CCP above all have the same denominator. Moreover, for identification reasons, we normalize $x_0 = 0$. Therefore,
$$ \frac{s_j}{s_0} = e^{x_j'\beta}.$$

Using Berry's Inversion (1994), and take log for both sides gives us:

$$ \text{ln}(s_j) - \text{ln} (s_0) \ = \delta_j \equiv x_j' \beta - \alpha p_j + \xi_j \ \ \ \ \ \text{(eq} A) $$ 

where
- Mean utility level $\delta_j$ contains the product characteristic $x_j$, $price_j$ and the aggregate error $\xi_j$.
- Econometricians observe aggregate market share for good $j$, outside good, product characteristics $x_j$, price $p_j$.
- $\delta_j$ is uniquely identified directly from a simple algebraic calculation involving market share.
- This is a OLS. (We need instruments for the price!)

# For the rest of this notebook, we will introduce two empirical examples:

## A. Estimate logit-demand using, BLP(1995)'s aggregate market level data.

- A1. Introduction of car data from BLP(1995)
- A2. Data cleaning
- A3. Run linear regression using eq(A)
- A4. Run 2sls using instruments

## B. Monte Carlo Example: estimate logit-demand after solving Nash-Bertrand game

- B1. Data Generating Process
- B2. Obtain (numerically) equilibrium price and market shares
- B3. Regress using OLS / IV (cost shifters, and competitors' product charactersitics as instruments for price. (Table 1, Berry (1994)

### A1.Introduction of car data from BLP(1995)

#### As an empirical study, we will replicate Table 3, as in BLP (1995).

- We obtain product characteristics from the annual issues of the  Automotive News Market Data Book and you can find BLP.csv.

- Data includes the number of cylinders, number of doors, weight, engine displacement, horsepower, length, width, wheelbase, EPA miles per gallon rating  (MPG), and dummy variables for whether the car has front-wheel drive,  automatic transmission, power steering, and air conditioning.
- The data set includes this information on observed products from 1971-1990.
- The price variable is list retail price (in \$1000's). Prices are in 1983 dollars. (We used the Consumer Price Index to  deflate.) 
- The sales variable corresponds to U.S. sales (in 1000's) by nameplate.
- The product characteristics correspond to the characteristics of the base model for the given nameplate.
- To capture the cost of driving, we include milers per dollar (MP\$), calculated as MPG divided by price per gallon. (Notice that MPG and pricer per gallon is provided.)


- In terms of potential market size, there is no formal definition. we used the yearly number of households in the U.S. from Statistical Abstract of the U.S.
- We assume that each model comprises a single firm to avoid a multi-product pricing problem.


```julia
# Query / DataFramesMeta is used for cleaning dataset
# FixedEffectModels is used for running regression
# Distributions, Random, NLsolve are used for Monte Carlo study
using CSV, DataFrames, Query, DataFramesMeta, FixedEffectModels, Distributions, Random, NLsolve
```


```julia
ENV["COLUMNS"],ENV["LINES"] = 350,50  #This is not specific to Julia, it's a Jupyter notebook environment variable

dataset = CSV.read("/Users/jinkim/Dropbox/2020 Summer/dropbox_RA_work/Berry/BLP.csv")
#first(dataset,10)
```

    ┌ Warning: `CSV.read(input; kw...)` is deprecated in favor of `using DataFrames; CSV.read(input, DataFrame; kw...)
    │   caller = read(::String) at CSV.jl:40
    └ @ CSV /Users/jinkim/.julia/packages/CSV/MKemC/src/CSV.jl:40




## Variable name/short description
#### For detailed description, please see BLP(1995) section 7.1. (Data section)

| Variable name    | Description                 | 
|------------------|-----------------------------|
| name             | Car                         |
| id               | Car ID                      |
| ye               | Year                        |
| cy               | Cylinder                    |
| dr               | Number of Doors             | 
| at               | Automatic Transmission      |
| ps               | Power Steering              |
| air              | Air Conditioning            |
| drv              | Front Wheel Drive           |
| p                | Price (in \$ 1000's)        |
| wt               | Weight                      |
| dom              | Domestic                    |
| disp             | Engine Displacement         |
| hp               | Horse Power                 |
| lng              | Length                      |
| wdt              | Width                       |
| wb               | Wheelbase                   |
| mpg              | Miles per Gallon            |
| q                | Quantities                  |
| firmids          | Firm ID                     |
| euro             | Indicator for EURO car      |
| reli             | Rating                      |
| dfi              | Indicator for Digital Fuel Injection       |
| hp2wt            | HP to Weight (ratio)        |
| size             | Length X Width (/1000)      |
| japan            | Japan                       |
| cpi              | CPI                         |
| gasprice         | Gas Price per gallon        |
| nb_hh            | Size of Household (Potential Market Size)          |
| cat              | Size Cateogry               |
| cat              | (Using for nested logit)    |
| door2            | I(door=2)                   |
| door3            | I(door=3)                   |
| door4            | I(door=4)                   |
| door5            | I(door=5)                   |
| sampleweight     | Weights                     |
| mpgd             | Miles per gallon  (imputed from gas prices) |
| dpm              | Dollars per miles (imputed from gas prices) |
| modelid          | Car name                                    |

 



```julia
#summary statistics
describe(dataset,:all)
```




<table class="data-frame"><thead><tr><th></th><th>variable</th><th>mean</th><th>std</th><th>min</th><th>q25</th><th>median</th><th>q75</th><th>max</th><th>nunique</th><th>nmissing</th><th>first</th><th>last</th><th>eltype</th></tr><tr><th></th><th>Symbol</th><th>Union…</th><th>Union…</th><th>Any</th><th>Union…</th><th>Union…</th><th>Union…</th><th>Any</th><th>Union…</th><th>Nothing</th><th>Any</th><th>Any</th><th>DataType</th></tr></thead><tbody><p>38 rows × 13 columns</p><tr><th>1</th><td>name</td><td></td><td></td><td>ACINTE</td><td></td><td></td><td></td><td>YUYUGO</td><td>542</td><td></td><td>ACINTE</td><td>YUYUGO</td><td>String</td></tr><tr><th>2</th><td>id</td><td>2560.89</td><td>1517.55</td><td>129</td><td>1309.0</td><td>2325.0</td><td>3927.0</td><td>5592</td><td></td><td></td><td>3735</td><td>4506</td><td>Int64</td></tr><tr><th>3</th><td>year</td><td>1981.54</td><td>5.74082</td><td>1971</td><td>1977.0</td><td>1982.0</td><td>1987.0</td><td>1990</td><td></td><td></td><td>1986</td><td>1989</td><td>Int64</td></tr><tr><th>4</th><td>cy</td><td>5.3207</td><td>1.55712</td><td>0</td><td>4.0</td><td>4.0</td><td>6.0</td><td>12</td><td></td><td></td><td>4</td><td>4</td><td>Int64</td></tr><tr><th>5</th><td>dr</td><td>3.29409</td><td>0.965154</td><td>2</td><td>2.0</td><td>4.0</td><td>4.0</td><td>5</td><td></td><td></td><td>3</td><td>2</td><td>Int64</td></tr><tr><th>6</th><td>at</td><td>0.326116</td><td>0.468896</td><td>0</td><td>0.0</td><td>0.0</td><td>1.0</td><td>1</td><td></td><td></td><td>0</td><td>0</td><td>Int64</td></tr><tr><th>7</th><td>ps</td><td>0.533153</td><td>0.499012</td><td>0</td><td>0.0</td><td>1.0</td><td>1.0</td><td>1</td><td></td><td></td><td>1</td><td>0</td><td>Int64</td></tr><tr><th>8</th><td>air</td><td>0.241768</td><td>0.428251</td><td>0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1</td><td></td><td></td><td>0</td><td>0</td><td>Int64</td></tr><tr><th>9</th><td>drv</td><td>0.354984</td><td>0.478616</td><td>0</td><td>0.0</td><td>0.0</td><td>1.0</td><td>1</td><td></td><td></td><td>1</td><td>1</td><td>Int64</td></tr><tr><th>10</th><td>p</td><td>11.7614</td><td>8.64378</td><td>3.39327</td><td>6.71375</td><td>8.72865</td><td>13.0741</td><td>68.5968</td><td></td><td></td><td>8.48358</td><td>3.50726</td><td>Float64</td></tr><tr><th>11</th><td>wt</td><td>2930.47</td><td>722.366</td><td>1445</td><td>2375.0</td><td>2861.0</td><td>3383.0</td><td>5362</td><td></td><td></td><td>2249</td><td>1832</td><td>Int64</td></tr><tr><th>12</th><td>dom</td><td>0.589986</td><td>0.491947</td><td>0</td><td>0.0</td><td>1.0</td><td>1.0</td><td>1</td><td></td><td></td><td>0</td><td>0</td><td>Int64</td></tr><tr><th>13</th><td>disp</td><td>177.746</td><td>102.032</td><td>1.0</td><td>109.0</td><td>151.0</td><td>231.0</td><td>500.0</td><td></td><td></td><td>97.0</td><td>68.0</td><td>Float64</td></tr><tr><th>14</th><td>hp</td><td>117.005</td><td>46.6881</td><td>39</td><td>88.0</td><td>105.0</td><td>140.0</td><td>365</td><td></td><td></td><td>113</td><td>52</td><td>Int64</td></tr><tr><th>15</th><td>lng</td><td>186.705</td><td>20.0657</td><td>139.0</td><td>172.2</td><td>185.0</td><td>200.0</td><td>236.0</td><td></td><td></td><td>168.5</td><td>139.0</td><td>Float64</td></tr><tr><th>16</th><td>wdt</td><td>69.6557</td><td>5.29795</td><td>53.0</td><td>65.9</td><td>69.0</td><td>73.0</td><td>81.0</td><td></td><td></td><td>65.6</td><td>60.7</td><td>Float64</td></tr><tr><th>17</th><td>wb</td><td>104.616</td><td>9.817</td><td>14.3</td><td>97.0</td><td>103.4</td><td>110.8</td><td>212.1</td><td></td><td></td><td>96.5</td><td>84.6</td><td>Float64</td></tr><tr><th>18</th><td>mpg</td><td>20.9964</td><td>5.8107</td><td>9.13</td><td>17.0</td><td>20.0</td><td>25.0</td><td>53.0</td><td></td><td></td><td>27.0</td><td>28.0</td><td>Float64</td></tr><tr><th>19</th><td>q</td><td>78.804</td><td>89.0799</td><td>0.049</td><td>15.603</td><td>47.35</td><td>109.002</td><td>646.526</td><td></td><td></td><td>27.807</td><td>10.5</td><td>Float64</td></tr><tr><th>20</th><td>firmids</td><td>13.7438</td><td>6.25909</td><td>1</td><td>8.0</td><td>16.0</td><td>19.0</td><td>26</td><td></td><td></td><td>3</td><td>23</td><td>Int64</td></tr><tr><th>21</th><td>euro</td><td>0.23816</td><td>0.426053</td><td>0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1</td><td></td><td></td><td>0</td><td>1</td><td>Int64</td></tr><tr><th>22</th><td>reli</td><td>3.0433</td><td>1.29108</td><td>1</td><td>2.0</td><td>3.0</td><td>4.0</td><td>5</td><td></td><td></td><td>5</td><td>1</td><td>Int64</td></tr><tr><th>23</th><td>dfi</td><td>0.0135318</td><td>0.115563</td><td>0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1</td><td></td><td></td><td>0</td><td>0</td><td>Int64</td></tr><tr><th>24</th><td>hp2wt</td><td>0.394375</td><td>0.0966429</td><td>0.170455</td><td>0.336585</td><td>0.375049</td><td>0.427509</td><td>0.947581</td><td></td><td></td><td>0.502446</td><td>0.283843</td><td>Float64</td></tr><tr><th>25</th><td>size</td><td>1.31016</td><td>0.237637</td><td>0.756</td><td>1.13128</td><td>1.26983</td><td>1.4527</td><td>1.888</td><td></td><td></td><td>1.10536</td><td>0.84373</td><td>Float64</td></tr><tr><th>26</th><td>japan</td><td>0.171854</td><td>0.377338</td><td>0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1</td><td></td><td></td><td>1</td><td>0</td><td>Int64</td></tr><tr><th>27</th><td>cpi</td><td>88.3488</td><td>28.8772</td><td>40.5</td><td>60.6</td><td>96.5</td><td>113.6</td><td>130.7</td><td></td><td></td><td>109.6</td><td>124.0</td><td>Float64</td></tr><tr><th>28</th><td>gasprice</td><td>1.02727</td><td>0.206509</td><td>0.785362</td><td>0.826794</td><td>1.01788</td><td>1.13523</td><td>1.47121</td><td></td><td></td><td>0.826794</td><td>0.810081</td><td>Float64</td></tr><tr><th>29</th><td>nb_hh</td><td>81539.0</td><td>8770.71</td><td>64778</td><td>74142.0</td><td>83527.0</td><td>89479.0</td><td>93347</td><td></td><td></td><td>88458</td><td>92830</td><td>Int64</td></tr><tr><th>30</th><td>cat</td><td></td><td></td><td>compact</td><td></td><td></td><td></td><td>midsize</td><td>3</td><td></td><td>compact</td><td>compact</td><td>String</td></tr><tr><th>31</th><td>door2</td><td>0.341903</td><td>0.474454</td><td>0</td><td>0.0</td><td>0.0</td><td>1.0</td><td>1</td><td></td><td></td><td>0</td><td>1</td><td>Int64</td></tr><tr><th>32</th><td>door3</td><td>0.0419486</td><td>0.200517</td><td>0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1</td><td></td><td></td><td>1</td><td>0</td><td>Int64</td></tr><tr><th>33</th><td>door4</td><td>0.596301</td><td>0.490749</td><td>0</td><td>0.0</td><td>1.0</td><td>1.0</td><td>1</td><td></td><td></td><td>0</td><td>0</td><td>Int64</td></tr><tr><th>34</th><td>door5</td><td>0.0198466</td><td>0.139505</td><td>0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1</td><td></td><td></td><td>0</td><td>0</td><td>Int64</td></tr><tr><th>35</th><td>sampleweight</td><td>78804.0</td><td>89079.9</td><td>49</td><td>15603.0</td><td>47350.0</td><td>109002.0</td><td>646526</td><td></td><td></td><td>27807</td><td>10500</td><td>Int64</td></tr><tr><th>36</th><td>mpgd</td><td>21.1248</td><td>6.94301</td><td>8.61178</td><td>15.9838</td><td>20.5059</td><td>24.8007</td><td>65.4256</td><td></td><td></td><td>32.6562</td><td>34.5645</td><td>Float64</td></tr><tr><th>37</th><td>dpm</td><td>0.052389</td><td>0.0167817</td><td>0.0152845</td><td>0.0403215</td><td>0.0487664</td><td>0.0625634</td><td>0.11612</td><td></td><td></td><td>0.030622</td><td>0.0289315</td><td>Float64</td></tr><tr><th>38</th><td>model</td><td></td><td></td><td>ACINTE1986</td><td></td><td></td><td></td><td>YUYUGO1989</td><td>2172</td><td></td><td>ACINTE1986</td><td>YUYUGO1989</td><td>String</td></tr></tbody></table>




```julia
### Replicate Table 1: Summary Stats
using Statistics

#Add or substrct column names as needed:
cnames = [:year,:cy,:dr,:at,:ps,:air,:drv,:p,:wt,:dom,:disp,:hp,:lng,:wdt,:wb,:mpg]
aggregate(dataset[!,cnames], [:year], mean)
```

    ┌ Warning: `aggregate(d, cols, f, sort=false, skipmissing=false)` is deprecated. Instead use combine(groupby(d, cols, sort=false, skipmissing=false), names(d, Not(cols)) .=> f)`
    │   caller = top-level scope at In[4]:6
    └ @ Core In[4]:6





<table class="data-frame"><thead><tr><th></th><th>year</th><th>cy_mean</th><th>dr_mean</th><th>at_mean</th><th>ps_mean</th><th>air_mean</th><th>drv_mean</th><th>p_mean</th><th>wt_mean</th><th>dom_mean</th><th>disp_mean</th><th>hp_mean</th><th>lng_mean</th><th>wdt_mean</th><th>wb_mean</th><th>mpg_mean</th></tr><tr><th></th><th>Int64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>20 rows × 16 columns</p><tr><th>1</th><td>1986</td><td>4.94615</td><td>3.43846</td><td>0.330769</td><td>0.684615</td><td>0.323077</td><td>0.576923</td><td>11.7826</td><td>2733.55</td><td>0.569231</td><td>160.212</td><td>110.192</td><td>180.583</td><td>67.9892</td><td>101.829</td><td>23.5546</td></tr><tr><th>2</th><td>1987</td><td>5.0</td><td>3.43357</td><td>0.321678</td><td>0.72028</td><td>0.391608</td><td>0.615385</td><td>13.4363</td><td>2785.15</td><td>0.538462</td><td>163.357</td><td>117.028</td><td>180.416</td><td>67.9965</td><td>101.909</td><td>22.6503</td></tr><tr><th>3</th><td>1988</td><td>5.09333</td><td>3.44</td><td>0.4</td><td>0.766667</td><td>0.446667</td><td>0.613333</td><td>14.8857</td><td>2847.8</td><td>0.533333</td><td>163.836</td><td>125.053</td><td>181.756</td><td>68.2853</td><td>102.443</td><td>21.88</td></tr><tr><th>4</th><td>1989</td><td>4.9932</td><td>2.9932</td><td>0.37415</td><td>0.795918</td><td>0.503401</td><td>0.680272</td><td>16.6905</td><td>2895.97</td><td>0.496599</td><td>163.503</td><td>133.837</td><td>181.348</td><td>68.5537</td><td>103.037</td><td>21.7891</td></tr><tr><th>5</th><td>1990</td><td>5.0687</td><td>2.85496</td><td>0.40458</td><td>0.816794</td><td>0.458015</td><td>0.717557</td><td>14.0384</td><td>2923.47</td><td>0.51145</td><td>2.7084</td><td>133.588</td><td>182.156</td><td>68.655</td><td>101.843</td><td>21.8168</td></tr><tr><th>6</th><td>1971</td><td>6.04348</td><td>3.36957</td><td>0.152174</td><td>0.163043</td><td>0.0</td><td>0.0</td><td>8.85633</td><td>3176.08</td><td>0.684783</td><td>245.548</td><td>171.891</td><td>195.446</td><td>72.9891</td><td>110.787</td><td>17.1995</td></tr><tr><th>7</th><td>1975</td><td>5.86022</td><td>3.26882</td><td>0.311828</td><td>0.387097</td><td>0.107527</td><td>0.0752688</td><td>9.64184</td><td>3328.6</td><td>0.645161</td><td>238.5</td><td>119.473</td><td>195.312</td><td>71.7419</td><td>108.129</td><td>16.2688</td></tr><tr><th>8</th><td>1976</td><td>5.62626</td><td>3.28283</td><td>0.282828</td><td>0.343434</td><td>0.10101</td><td>0.0909091</td><td>9.49007</td><td>3159.45</td><td>0.616162</td><td>218.66</td><td>112.889</td><td>192.515</td><td>70.8889</td><td>106.833</td><td>18.5556</td></tr><tr><th>9</th><td>1977</td><td>5.61053</td><td>3.24211</td><td>0.294737</td><td>0.378947</td><td>0.0842105</td><td>0.136842</td><td>9.86429</td><td>3070.06</td><td>0.610526</td><td>204.085</td><td>109.463</td><td>190.747</td><td>70.4947</td><td>106.038</td><td>19.7789</td></tr><tr><th>10</th><td>1980</td><td>5.23301</td><td>3.35922</td><td>0.252427</td><td>0.359223</td><td>0.174757</td><td>0.252427</td><td>10.7269</td><td>2776.99</td><td>0.601942</td><td>179.446</td><td>99.699</td><td>185.398</td><td>69.4854</td><td>103.684</td><td>21.6699</td></tr><tr><th>11</th><td>1981</td><td>5.16379</td><td>3.21552</td><td>0.336207</td><td>0.491379</td><td>0.241379</td><td>0.232759</td><td>13.0352</td><td>2819.37</td><td>0.517241</td><td>171.325</td><td>102.017</td><td>186.56</td><td>69.3276</td><td>103.929</td><td>22.5</td></tr><tr><th>12</th><td>1982</td><td>4.98182</td><td>3.32727</td><td>0.336364</td><td>0.490909</td><td>0.236364</td><td>0.381818</td><td>11.5913</td><td>2747.25</td><td>0.6</td><td>158.536</td><td>98.3</td><td>185.273</td><td>68.8636</td><td>103.151</td><td>23.7727</td></tr><tr><th>13</th><td>1983</td><td>4.88696</td><td>3.30435</td><td>0.33913</td><td>0.530435</td><td>0.2</td><td>0.434783</td><td>11.1408</td><td>2736.82</td><td>0.617391</td><td>156.169</td><td>97.8522</td><td>184.087</td><td>68.7913</td><td>102.857</td><td>25.1478</td></tr><tr><th>14</th><td>1984</td><td>4.92035</td><td>3.35398</td><td>0.345133</td><td>0.663717</td><td>0.283186</td><td>0.477876</td><td>11.6477</td><td>2765.41</td><td>0.59292</td><td>166.069</td><td>107.265</td><td>183.822</td><td>68.4637</td><td>102.735</td><td>24.1504</td></tr><tr><th>15</th><td>1985</td><td>4.96324</td><td>3.31618</td><td>0.345588</td><td>0.669118</td><td>0.330882</td><td>0.522059</td><td>12.4764</td><td>2759.99</td><td>0.566176</td><td>163.106</td><td>108.081</td><td>182.096</td><td>68.1625</td><td>102.349</td><td>22.3868</td></tr><tr><th>16</th><td>1978</td><td>5.57895</td><td>3.33684</td><td>0.263158</td><td>0.347368</td><td>0.0947368</td><td>0.178947</td><td>10.6021</td><td>2964.8</td><td>0.642105</td><td>199.494</td><td>107.695</td><td>189.347</td><td>70.3579</td><td>105.337</td><td>19.9263</td></tr><tr><th>17</th><td>1979</td><td>5.33333</td><td>3.28431</td><td>0.235294</td><td>0.313725</td><td>0.0882353</td><td>0.215686</td><td>10.4513</td><td>2840.49</td><td>0.598039</td><td>184.635</td><td>104.108</td><td>186.559</td><td>69.5</td><td>103.849</td><td>20.1078</td></tr><tr><th>18</th><td>1972</td><td>6.20225</td><td>3.44944</td><td>0.337079</td><td>0.348315</td><td>0.0449438</td><td>0.0</td><td>9.04282</td><td>3253.78</td><td>0.696629</td><td>256.947</td><td>134.348</td><td>196.775</td><td>73.3146</td><td>111.563</td><td>16.3317</td></tr><tr><th>19</th><td>1973</td><td>6.37209</td><td>3.44186</td><td>0.395349</td><td>0.372093</td><td>0.0697674</td><td>0.0</td><td>9.0452</td><td>3337.42</td><td>0.709302</td><td>261.949</td><td>131.256</td><td>198.384</td><td>73.0233</td><td>111.215</td><td>16.2506</td></tr><tr><th>20</th><td>1974</td><td>6.0</td><td>3.30556</td><td>0.375</td><td>0.375</td><td>0.125</td><td>0.0</td><td>9.25473</td><td>3268.29</td><td>0.652778</td><td>239.179</td><td>122.556</td><td>196.194</td><td>71.9306</td><td>108.874</td><td>16.3329</td></tr></tbody></table>



### A2. Data cleaning

#### Step 1. Obtain market share for each good $j$: $s_{jt}$ = $\frac{q_{jt}}{nb\_hh_{t}}$

For notation, let denote total market size $nb\_hh_t = M_t$


```julia
dataset = @linq dataset |> 
groupby([:year]) |>
transform(Total_Q = sum(:q))

dataset.s_j = dataset.q./dataset.nb_hh;
```

#### Step 2. Obtain market share for outside good 0: $s_{0t}$ = $\frac{ \Big(nb\_hh_t - \sum_{k=1}^J(q_{kt}) \Big)}{nb\_hh_t}$


```julia
dataset.s_0 = (dataset.nb_hh-dataset.Total_Q)./dataset.nb_hh;
```

#### Step 3. Construct dependent variable: $ \text{ln}(s_{jt}) - \text{ln}(s_{0t}) $


```julia
dataset.log_s_j_0 = log.(dataset.s_j) - log.(dataset.s_0);
```

### A3. Run linear regression using eq(A)

#### Step 4. we use hp2wt, air, mpgd, size as product characteristics:
$$ \text{ln}(s_j) - \text{ln} (s_0) \ = \delta_j \equiv x_j' \beta - \alpha p_j + \xi_j $$ 


```julia
result = reg(dataset, @formula(log_s_j_0 ~ p+hp2wt+air+mpgd+size ),  save = true,);
print(result)
```

                                   Linear Model                               
    ===========================================================================
    Number of obs:                  2217   Degrees of freedom:                6
    R2:                            0.390   R2 Adjusted:                   0.388
    F Statistic:                  282.47   p-value:                       0.000
    ===========================================================================
                   Estimate  Std.Error   t value Pr(>|t|)  Lower 95%  Upper 95%
    ---------------------------------------------------------------------------
    p            -0.0886398 0.00401348  -22.0855    0.000 -0.0965104 -0.0807692
    hp2wt        -0.0731735   0.276701  -0.26445    0.791  -0.615794   0.469447
    air          -0.0380115  0.0726455 -0.523246    0.601  -0.180472   0.104449
    mpgd          0.0288388 0.00439518   6.56147    0.000  0.0202197  0.0374579
    size            2.40052   0.126801   18.9315    0.000    2.15186    2.64919
    (Intercept)    -10.2035    0.26002  -39.2412    0.000   -10.7134   -9.69356
    ===========================================================================


####  Step 5. Obtain Price elasticities:

Note that own price elasticities $(\eta_j$) is given by: 
$$
\begin{align}
\eta_j & = \frac{\partial Pr(j)}{\partial price_j} \underbrace{\frac{price_j}{Pr(j)}}_{\frac{price_j}{s_j \times M}} \\
& \text{Note that} \ \frac{\partial Pr(j)}{\partial price_j} = \frac{\partial s_j}{\partial price_j} \times M  \ \text{where} \ s_j = \frac{e^{\delta_j}}{\sum_k^J e^{\delta_j}}  \\
& \text{Appealing to chain rule}: \frac{\partial s_j}{\partial price_j} \ M = \Bigg[ \alpha \frac{e^{\delta_j}}{\sum_k^J e^{\delta_j}} - \alpha \Big(  \frac{e^{\delta_j}}{\sum_k^J e^{\delta_j}}\Big)^2   \Bigg] = M \alpha [s_j - s_j^2]  = M \alpha s_j[1- s_j]\\
& \text{Rearranging these terms gives us:} \\
& \eta_j = \underbrace{\frac{\partial Pr(j)}{\partial price_j}}_{M \alpha s_j[1- s_j]} \underbrace{\frac{price_j}{Pr(j)}}_{\frac{price_j}{s_j \times M}} = M \alpha s_j[1- s_j] \times \frac{price_j}{s_j} \frac{1}{M} = \underbrace{\alpha \times (1-s_j) \times price_j}_\text{price elasticities for good j} \\
& = \alpha \times (1-s_j) \times price_j
\end{align}
$$

```julia
# Following price elasticities, I can derive price elasticities for each good j, using price coefficients alpha.

price_coef = coef(result)[2];
dataset.e = price_coef * (ones(nrow(dataset))-dataset.s_j) .* dataset.p;

q1 = @from i in dataset begin;
            @where i.e >-1
            @select {elasticity=i.e}
            @collect DataFrame
    end;
nrow(q1)

```




    1502



#### Replication: BLP Table 3, IV Logit Demand Column (Row: No. Inelastic De) in page 873.
#### I derive the number of inelastic car model. My estimates are 1,502. BLP's estimates were 1,494, which is pretty close.

### A4. Run 2sls using instruments

Following BLP, I use the following instruments for price.

#### 1. the sum of size at market $t$. (Note that you need to drop product $j$'s own size.)
#### 2. the sum of size across rival firm products at market $t$.


```julia
# IV 1

dataset = @linq dataset |> 
groupby([:year]) |>
transform(Total_size = sum(:size));
dataset.iv_size1 = dataset.Total_size - dataset.size;
```


```julia
# IV 2

dataset = @linq dataset |> 
groupby([:year , :firmids]) |>
transform(sum_size = sum(:size));
dataset.iv_size2 = dataset.Total_size - dataset.sum_size;
```


```julia
# 2SLS Regression for demand estimation
# First stage: regress price on Z and X

first_stage_result = reg(dataset, @formula(p ~ iv_size1+ iv_size2+hp2wt + air+mpgd+size), save = true, );
print(first_stage_result)
```

                                  Linear Model                              
    =========================================================================
    Number of obs:                 2217   Degrees of freedom:               7
    R2:                           0.592   R2 Adjusted:                  0.591
    F Statistic:                534.872   p-value:                      0.000
    =========================================================================
                  Estimate  Std.Error  t value Pr(>|t|)  Lower 95%  Upper 95%
    -------------------------------------------------------------------------
    iv_size1     -0.030544 0.00999712 -3.05528    0.002 -0.0501487 -0.0109393
    iv_size2     0.0919841 0.00809711  11.3601    0.000  0.0761054   0.107863
    hp2wt          25.8362    1.31783  19.6051    0.000    23.2518    28.4205
    air            9.57023   0.327809  29.1945    0.000    8.92738    10.2131
    mpgd         -0.272477  0.0270151 -10.0861    0.000  -0.325455  -0.219499
    size           2.25533   0.727117  3.10174    0.002   0.829424    3.68123
    (Intercept)   -5.21123    1.50736  -3.4572    0.001   -8.16721   -2.25525
    =========================================================================



```julia
# Second Stage: regress log(s_j)-log(s_0) on xhat

xhat = predict(first_stage_result, dataset);
dataset.p_iv = xhat;
second_stage_result = reg(dataset, @formula(log_s_j_0 ~ p_iv+hp2wt + air+mpgd+size), save = true);
print(second_stage_result)
```

                                    Linear Model                                
    ============================================================================
    Number of obs:                   2217  Degrees of freedom:                 6
    R2:                             0.355  R2 Adjusted:                    0.354
    F Statistic:                  243.848  p-value:                        0.000
    ============================================================================
                    Estimate  Std.Error  t value Pr(>|t|)  Lower 95%   Upper 95%
    ----------------------------------------------------------------------------
    p_iv           -0.289427  0.0156063 -18.5455    0.000  -0.320032   -0.258823
    hp2wt            5.63223   0.513603  10.9661    0.000    4.62503     6.63942
    air              2.18099   0.182327  11.9619    0.000    1.82344     2.53854
    mpgd         -0.00984143 0.00536771 -1.83345    0.067 -0.0203677 0.000684853
    size             2.19676   0.131213  16.7419    0.000    1.93944     2.45407
    (Intercept)      -9.5444   0.271767 -35.1198    0.000   -10.0773    -9.01145
    ============================================================================



```julia
price_coef = coef(second_stage_result)[2];
dataset.e_iv = price_coef * (ones(nrow(dataset))-dataset.s_j) .* dataset.p;
```


```julia
q1 = @from i in dataset begin;
            @where i.e_iv >-1
            @select {number_of_children=i.e_iv}
            @collect DataFrame
    end;
```

### Comparision with BLP Table 3, IV Logit Demand Column (Row: No. Inelastic De) in page 873.

#### Note that I have slightly different price coefficients, I observe number of inelastic demand good is 2. BLP estimates were 22.


```julia
nrow(q1)
```




    2



### Step 6. Discussion : IV regressions

#### Reported price coefficient ($\alpha$) is -0.0886 in OLS.
#### Now we have -0.2894  ($\alpha$) in IV regression. Prices are upward biased in OLS.


```julia
# OLS Results
print(result)
```

                                   Linear Model                               
    ===========================================================================
    Number of obs:                  2217   Degrees of freedom:                6
    R2:                            0.390   R2 Adjusted:                   0.388
    F Statistic:                  282.47   p-value:                       0.000
    ===========================================================================
                   Estimate  Std.Error   t value Pr(>|t|)  Lower 95%  Upper 95%
    ---------------------------------------------------------------------------
    p            -0.0886398 0.00401348  -22.0855    0.000 -0.0965104 -0.0807692
    hp2wt        -0.0731735   0.276701  -0.26445    0.791  -0.615794   0.469447
    air          -0.0380115  0.0726455 -0.523246    0.601  -0.180472   0.104449
    mpgd          0.0288388 0.00439518   6.56147    0.000  0.0202197  0.0374579
    size            2.40052   0.126801   18.9315    0.000    2.15186    2.64919
    (Intercept)    -10.2035    0.26002  -39.2412    0.000   -10.7134   -9.69356
    ===========================================================================



```julia
# 2SLS Results
print(second_stage_result)
```

                                    Linear Model                                
    ============================================================================
    Number of obs:                   2217  Degrees of freedom:                 6
    R2:                             0.355  R2 Adjusted:                    0.354
    F Statistic:                  243.848  p-value:                        0.000
    ============================================================================
                    Estimate  Std.Error  t value Pr(>|t|)  Lower 95%   Upper 95%
    ----------------------------------------------------------------------------
    p_iv           -0.289427  0.0156063 -18.5455    0.000  -0.320032   -0.258823
    hp2wt            5.63223   0.513603  10.9661    0.000    4.62503     6.63942
    air              2.18099   0.182327  11.9619    0.000    1.82344     2.53854
    mpgd         -0.00984143 0.00536771 -1.83345    0.067 -0.0203677 0.000684853
    size             2.19676   0.131213  16.7419    0.000    1.93944     2.45407
    (Intercept)      -9.5444   0.271767 -35.1198    0.000   -10.0773    -9.01145
    ============================================================================


## B. Monte Carlo Example: estimate logit-demand after solving Nash-Bertrand game
- B1. Data Generating Process
- B2. Obtain (numerically) equilibrium price and market shares
- B3. Regress using OLS / IV

### B1. Data Generating Process


Market is characterized by dupoly firms which procuce single good, with aggregate market shares, and price for each good. We assume that duopoly firms compete in 500 "independent" (isolated) markets.

In D.G.P., we solve Nash-Bertrand game so that we derive dupoly firm's price, and market shares. We use cost shifters and product characteritics to numerically solve this game. Since it is D.G.P. we use true parameters to obtain price, and market shares.


As an econometrician, we observe dupoly firm's market share, price, costs, and product characteristics. 

The utility of each consumer $i$ in each market is given by:

\begin{equation}
    u_{ij} = \beta_0 + \beta_1 x_j + \sigma_d \xi_j - \alpha p_j + \epsilon_{ij}
\end{equation}
Marginal cost is constrained to be postitive and given by:

\begin{equation}
    c_j = e^{\gamma_0 + \gamma_x x_j + \sigma_c \xi_j + \gamma_w w_j + \sigma_\omega \omega_j}
\end{equation}

The exogenous data $x_j, \xi_j, w_j, $ and $\omega_j$ are all created standard normal random variables.

True parameter is given by: 

| Parameter        | True Value | Description |
|------------------|------------|-------------|
| $ \beta_0$       | 5          |  Intercept (demand)          |
| $ \beta_x$       | 2          |  Utility from good $x$       |
| $ \sigma_d$      | 1 / 3 (second monte carlo)      |  Covariance                  |
| $\alpha $        | 1          |  Price coefficients          |
| $ \gamma_0$      | 1          |  Intercept (Supply)          |
| $\gamma_x $      | 5          |  Cost from good $x$          |
| $ \sigma_c$      | 0.25       |  Covariance  $(\xi_j)$                |
| $ \gamma_w$      | 0.25       |   Parameters for input costs |
| $ \sigma_\omega$ | 0.25       |   Covariance $(\omega_j)$                |




### B2. Obtain (numerically) equilibrium price and market shares (still D.G.P)
I solve nonlinear equation where under j=1,2. (Argument: $p_1, p_2, s_1(p_1, p_2), s_2(p_1,p_2)$)  $X$ is a vector dupoly firm's product charactersitics $ X = \{ x_1, x_2 \}$
Note that $s_0 = 1-s_1-s_2$
$$
\begin{align}
p_1 & = c_1 - \frac{1}{\alpha (1-s_1)} \\
p_2 & = c_2 - \frac{1}{\alpha (1-s_2)} \\
& \text{Note that $s_1$, and $s_2$ is given by} \\
s_1(X,p_1, p_2) & = \frac{exp^{\beta_0 + \beta_1 x_1 + \sigma_d \xi_1 - \alpha p_1}}{1+exp^{\beta_0 + \beta_1 x_1 + \sigma_d \xi_1 - \alpha p_1} +exp^{\beta_0 + \beta_1 x_2 + \sigma_d \xi_2 - \alpha p_2} } \\
s_2(X,p_1, p_2) & = \frac{exp^{\beta_0 + \beta_1 x_2 + \sigma_d \xi_2 - \alpha p_2}}{1+exp^{\beta_0 + \beta_1 x_1 + \sigma_d \xi_1 - \alpha p_1} + exp^{\beta_0 + \beta_1 x_2 + \sigma_d \xi_2 - \alpha p_2}}
\end{align}
$$

Using Nonlinear solver, we can obtain equilibrium outcome: $p_1, p_2, s_1, s_2 (s_0= 1-s_1-s_2).$ One might concern about multiple equilibria for this game. Since we solve single product under duopoly (which is simple market), we observe unique solution for this Monte Carlo Study. Please see Nalebuff (1991) for multi-product firm problem, or uniquness of this game.

### B3. Regress using OLS / IV
Following Berry's inversion an econometrician run following OLS/IV regression. An econometrician observes price, product characteristics, and cost shifters for 500 independent dupoly markets.

\begin{align}
\text{ln} (s_j) - \text{ln} (s_0) \ & =  \delta_j \\
& = \beta_0 + \beta_1 x_j - \alpha p_j + \sigma_d \xi_j 
\end{align}

For IV, I use cost shifters, and competitors' product characteristics, as in Berry (1994)

Note that in OLS, since an econometrician cannot observe $\xi_j$ term, price coefficients $\alpha$ is upward biased. In the IV regression, the observed cost factors, $w_j$, and the product characteristic of the rival firm are used as instruments for price.

#### For each simulation, we independently draw these for 500 markets, as in Berry(1994).
#### Repeat 100 times, and report Monte Racrlo results


```julia
# Define parameters, as in Berry's 1994 Monte Carlo

beta_0 = 5.0
beta_x = 2.0
sigma_d = 1.0
alpha = -1.0

gamma_0 = 1.0
gamma_x = 0.5

sigma_c = 0.25
gamma_w = 0.25
sigma_omega = 0.25

T = 500
S = 100
d = Normal()
```




    Normal{Float64}(μ=0.0, σ=1.0)




```julia
# Define Non-linear solver

function f!(F, x)
    
    # D.G.P for true costs
    cost_1 = exp(gamma_0 + gamma_x * data_temp1[:x] + sigma_c * data_temp1[:xi] + gamma_w * data_temp1[:w] + sigma_omega * data_temp1[:omega])
    cost_2 = exp(gamma_0 + gamma_x * data_temp2[:x] + sigma_c * data_temp2[:xi] + gamma_w * data_temp2[:w] + sigma_omega * data_temp2[:omega])
    
    # Derive equillibrium price / quantity(markset shares)
    price_1 = cost_1 - 1/(alpha*(1-x[1]))
    price_2 = cost_2 - 1/(alpha*(1-x[2]))    
    w
    #x[1]: market share of good 1
    #x[2]: market share of good 2
    
    denom = 1 + exp(beta_0 + beta_x*data_temp1[:x] + sigma_d*data_temp1[:xi] + alpha*price_1)  + exp(beta_0 + beta_x*data_temp2[:x] + sigma_d*data_temp2[:xi] + alpha*price_2) 
    F[1] = x[1] - exp(beta_0 + beta_x*data_temp1[:x] + sigma_d*data_temp1[:xi] + alpha*price_1)/denom
    F[2] = x[2] - exp(beta_0 + beta_x*data_temp2[:x] + sigma_d*data_temp2[:xi] + alpha*price_2)/denom
end
```

### Replicate Table 1 in Berry(1994), column (1) and (2), where $\sigma_d=1$


```julia
### It takes 5 seconds, at the current workflow
sigma_d = 1

for s = 1:S
    ### Step B1. D.G.P.
    
    # If you want to have same results, you need to assign random seed
    Random.seed!(s*100+T+1)
    x_1 = rand(d, T);
    Random.seed!(s*100+T+2)
    xi_1 = rand(d, T);
    Random.seed!(s*100+T+3)
    w_1 = rand(d, T);
    Random.seed!(s*100+T+4)
    omega_1 = rand(d, T);
    
    Random.seed!(s*100+T+5)
    x_2 = rand(d, T);
    Random.seed!(s*100+T+6)
    xi_2 = rand(d, T);
    Random.seed!(s*100+T+7)
    w_2 = rand(d, T);
    Random.seed!(s*100+T+8)
    omega_2 = rand(d, T);

    data_1 = DataFrame(x = x_1[1:T], xi = xi_1[1:T], w = w_1[1:T], omega = omega_1[1:T], iv=x_2[1:T]);
    data_2 = DataFrame(x = x_2[1:T], xi = xi_2[1:T], w = w_2[1:T], omega = omega_2[1:T], iv=x_1[1:T]);

    # For the first periods

    data_temp1 = data_1[1,:]
    data_temp2 = data_2[1,:]
    global data_temp1, data_temp2
    
    ### Step B2. Solve Equilibrium price and market shares using nonlinear-solver
    
    a= nlsolve(f!, [0.1; 0.1])

    vector_s1 = [a.zero[1]]
    vector_s2 = [a.zero[2]]
    vector_s0 = [1-a.zero[1]-a.zero[2]]

    cost_1 = exp(gamma_0 + gamma_x * data_temp1[:x] + sigma_c * data_temp1[:xi] + gamma_w * data_temp1[:w] + sigma_omega * data_temp1[:omega])
    cost_2 = exp(gamma_0 + gamma_x * data_temp2[:x] + sigma_c * data_temp2[:xi] + gamma_w * data_temp2[:w] + sigma_omega * data_temp2[:omega])

    vector_p1 = [cost_1 - 1/(alpha*(1-a.zero[1]))]
    vector_p2 = [cost_2 - 1/(alpha*(1-a.zero[2]))]

    vector_delta_1 = [beta_0 + beta_x * data_temp1[:x] + alpha*(cost_1 - 1/(alpha*(1-a.zero[1])) )]
    vector_delta_2 = [beta_0 + beta_x * data_temp2[:x] + alpha*(cost_2 - 1/(alpha*(1-a.zero[2])) )]

    # From the second market to T markets.
    t=2
    for t = 2:T

        data_temp1 = data_1[t,:]
        data_temp2 = data_2[t,:]
        
        # Step 1. Solve Equilibrium price / market shares
        
        a= nlsolve(f!, [0.0; 0.0])

        append!(vector_s1, [a.zero[1]]);
        append!(vector_s2, [a.zero[2]]);
        append!(vector_s0, [1-a.zero[1]-a.zero[2]]);
        cost_1 = exp(gamma_0 + gamma_x * data_temp1[:x] + sigma_c * data_temp1[:xi] + gamma_w * data_temp1[:w] + sigma_omega * data_temp1[:omega])
        cost_2 = exp(gamma_0 + gamma_x * data_temp2[:x] + sigma_c * data_temp2[:xi] + gamma_w * data_temp2[:w] + sigma_omega * data_temp2[:omega])

        append!(vector_p1, [cost_1 - 1/(alpha*(1-a.zero[1]))]);
        append!(vector_p2, [cost_2 - 1/(alpha*(1-a.zero[2]))]);

        append!(vector_delta_1, [beta_0 + beta_x * data_temp1[:x] + alpha*(cost_1 - 1/(alpha*(1-a.zero[1])) )]);
        append!(vector_delta_2, [beta_0 + beta_x * data_temp2[:x] + alpha*(cost_2 - 1/(alpha*(1-a.zero[2])) )]);

    end

    data_1.price = vector_p1;
    data_2.price = vector_p2;

    data_1.s = vector_s1;
    data_2.s = vector_s2;

    data_1.delta = vector_delta_1;
    data_2.delta = vector_delta_2;

    data_1.log_sj_s0 = log.(vector_s1) - log.(vector_s0);
    data_2.log_sj_s0 = log.(vector_s2) - log.(vector_s0);
    
    # Merge into dataset
    data_merged =  append!(data_1, data_2);
    
    ### B3. Regress using OLS / IV
    
    ## OLS Regression
    
    ols_result = reg(data_merged, @formula(log_sj_s0 ~ x + price), save = true, Vcov.robust());
    ols_cons = coef(ols_result)[1];
    ols_x = coef(ols_result)[2];
    ols_p = coef(ols_result)[3];

    ## IV Regression
    first_stage_result = reg(data_merged, @formula(price ~  iv + w +x), save = true, Vcov.robust());

    xhat = predict(first_stage_result, data_merged);
    data_merged.xhat = xhat;

    iv_result = reg(data_merged, @formula(log_sj_s0 ~  x + xhat), save = true, Vcov.robust());
    iv_cons = coef(iv_result)[1];
    iv_x = coef(iv_result)[2];
    iv_p = coef(iv_result)[3];
    
    if s == 1
        vector_ols_cons = [ols_cons]
        vector_ols_x = [ols_x]
        vector_ols_p = [ols_p]
    
        vector_iv_cons = [iv_cons]
        vector_iv_x = [iv_x]
        vector_iv_p = [iv_p]
        
        # Store Monte Carlo Results
        global vector_ols_cons, vector_ols_x, vector_ols_p, vector_iv_cons, vector_iv_x, vector_iv_p
    else

        append!(vector_ols_cons, [ols_cons])
        append!(vector_ols_x, [ols_x])
        append!(vector_ols_p, [ols_p])
        
        append!(vector_iv_cons, [iv_cons])
        append!(vector_iv_x, [iv_x])
        append!(vector_iv_p, [iv_p])

    end
end

print("Monte Carlo Parameter Estimates 100 Random Samples of 500 Duopoly Markets  Logit Utility (sigma_d  = 1)")

result_summary =DataFrame( True_parameter = [beta_0, beta_x, alpha], OLS_mean = [mean(vector_ols_cons),mean(vector_ols_x),mean(vector_ols_p)], OLS_se = [std(vector_ols_cons),std(vector_ols_x),std(vector_ols_p)],
    IV_mean = [mean(vector_iv_cons),mean(vector_iv_x),mean(vector_iv_p)], IV_se =[std(vector_iv_cons),std(vector_iv_x),std(vector_iv_p)]);
print("Result Summary")
print(result_summary)
```

    Monte Carlo Parameter Estimates 100 Random Samples of 500 Duopoly Markets  Logit Utility (sigma_d  = 1)Result Summary3×5 DataFrame
    │ Row │ True_parameter │ OLS_mean │ OLS_se    │ IV_mean  │ IV_se     │
    │     │ [90mFloat64[39m        │ [90mFloat64[39m  │ [90mFloat64[39m   │ [90mFloat64[39m  │ [90mFloat64[39m   │
    ├─────┼────────────────┼──────────┼───────────┼──────────┼───────────┤
    │ 1   │ 5.0            │ 3.1872   │ 0.235697  │ 5.01814  │ 0.266852  │
    │ 2   │ 2.0            │ 1.33611  │ 0.0742151 │ 2.01013  │ 0.0994837 │
    │ 3   │ -1.0           │ -0.63979 │ 0.0482097 │ -1.00436 │ 0.0513297 │

### Replicate Table 1 in Berry(1994), column (1) and (2), where $\sigma_d=3$


```julia
sigma_d = 3.0

for s = 1:S
    ### Step B1. D.G.P.
    
    # If you want to have same results, you need to assign random seed
    Random.seed!(s*100+T+1)
    x_1 = rand(d, T);
    Random.seed!(s*100+T+2)
    xi_1 = rand(d, T);
    Random.seed!(s*100+T+3)
    w_1 = rand(d, T);
    Random.seed!(s*100+T+4)
    omega_1 = rand(d, T);
    
    Random.seed!(s*100+T+5)
    x_2 = rand(d, T);
    Random.seed!(s*100+T+6)
    xi_2 = rand(d, T);
    Random.seed!(s*100+T+7)
    w_2 = rand(d, T);
    Random.seed!(s*100+T+8)
    omega_2 = rand(d, T);

    data_1 = DataFrame(x = x_1[1:T], xi = xi_1[1:T], w = w_1[1:T], omega = omega_1[1:T], iv=x_2[1:T]);
    data_2 = DataFrame(x = x_2[1:T], xi = xi_2[1:T], w = w_2[1:T], omega = omega_2[1:T], iv=x_1[1:T]);

    # For the first periods

    data_temp1 = data_1[1,:]
    data_temp2 = data_2[1,:]
    global data_temp1, data_temp2
    
    ### Step B2. Solve Equilibrium price and market shares using nonlinear-solver
    
    a= nlsolve(f!, [0.1; 0.1])

    vector_s1 = [a.zero[1]]
    vector_s2 = [a.zero[2]]
    vector_s0 = [1-a.zero[1]-a.zero[2]]

    cost_1 = exp(gamma_0 + gamma_x * data_temp1[:x] + sigma_c * data_temp1[:xi] + gamma_w * data_temp1[:w] + sigma_omega * data_temp1[:omega])
    cost_2 = exp(gamma_0 + gamma_x * data_temp2[:x] + sigma_c * data_temp2[:xi] + gamma_w * data_temp2[:w] + sigma_omega * data_temp2[:omega])

    vector_p1 = [cost_1 - 1/(alpha*(1-a.zero[1]))]
    vector_p2 = [cost_2 - 1/(alpha*(1-a.zero[2]))]

    vector_delta_1 = [beta_0 + beta_x * data_temp1[:x] + alpha*(cost_1 - 1/(alpha*(1-a.zero[1])) )]
    vector_delta_2 = [beta_0 + beta_x * data_temp2[:x] + alpha*(cost_2 - 1/(alpha*(1-a.zero[2])) )]

    # From the second market to T markets.
    t=2
    for t = 2:T

        data_temp1 = data_1[t,:]
        data_temp2 = data_2[t,:]
        
        # Step 1. Solve Equilibrium price / market shares
        
        a= nlsolve(f!, [0.0; 0.0])

        append!(vector_s1, [a.zero[1]]);
        append!(vector_s2, [a.zero[2]]);
        append!(vector_s0, [1-a.zero[1]-a.zero[2]]);
        cost_1 = exp(gamma_0 + gamma_x * data_temp1[:x] + sigma_c * data_temp1[:xi] + gamma_w * data_temp1[:w] + sigma_omega * data_temp1[:omega])
        cost_2 = exp(gamma_0 + gamma_x * data_temp2[:x] + sigma_c * data_temp2[:xi] + gamma_w * data_temp2[:w] + sigma_omega * data_temp2[:omega])

        append!(vector_p1, [cost_1 - 1/(alpha*(1-a.zero[1]))]);
        append!(vector_p2, [cost_2 - 1/(alpha*(1-a.zero[2]))]);

        append!(vector_delta_1, [beta_0 + beta_x * data_temp1[:x] + alpha*(cost_1 - 1/(alpha*(1-a.zero[1])) )]);
        append!(vector_delta_2, [beta_0 + beta_x * data_temp2[:x] + alpha*(cost_2 - 1/(alpha*(1-a.zero[2])) )]);

    end

    data_1.price = vector_p1;
    data_2.price = vector_p2;

    data_1.s = vector_s1;
    data_2.s = vector_s2;

    data_1.delta = vector_delta_1;
    data_2.delta = vector_delta_2;

    data_1.log_sj_s0 = log.(vector_s1) - log.(vector_s0);
    data_2.log_sj_s0 = log.(vector_s2) - log.(vector_s0);
    
    # Merge into dataset
    data_merged =  append!(data_1, data_2);
    
    ### B3. Regress using OLS / IV
    
    ## OLS Regression
    
    ols_result = reg(data_merged, @formula(log_sj_s0 ~ x + price), save = true, Vcov.robust());
    ols_cons = coef(ols_result)[1];
    ols_x = coef(ols_result)[2];
    ols_p = coef(ols_result)[3];

    ## IV Regression
    first_stage_result = reg(data_merged, @formula(price ~  iv + w +x), save = true, Vcov.robust());

    xhat = predict(first_stage_result, data_merged);
    data_merged.xhat = xhat;

    iv_result = reg(data_merged, @formula(log_sj_s0 ~  x + xhat), save = true, Vcov.robust());
    iv_cons = coef(iv_result)[1];
    iv_x = coef(iv_result)[2];
    iv_p = coef(iv_result)[3];
    
    if s == 1
        vector_ols_cons = [ols_cons]
        vector_ols_x = [ols_x]
        vector_ols_p = [ols_p]
    
        vector_iv_cons = [iv_cons]
        vector_iv_x = [iv_x]
        vector_iv_p = [iv_p]
        
        # Store Monte Carlo Results
        global vector_ols_cons, vector_ols_x, vector_ols_p, vector_iv_cons, vector_iv_x, vector_iv_p
    else

        append!(vector_ols_cons, [ols_cons])
        append!(vector_ols_x, [ols_x])
        append!(vector_ols_p, [ols_p])
        
        append!(vector_iv_cons, [iv_cons])
        append!(vector_iv_x, [iv_x])
        append!(vector_iv_p, [iv_p])

    end
end

print("Monte Carlo Parameter Estimates 100 Random Samples of 500 Duopoly Markets  Logit Utility (sigma_d  = 1)")

result_summary =DataFrame( True_parameter = [beta_0, beta_x, alpha], OLS_mean = [mean(vector_ols_cons),mean(vector_ols_x),mean(vector_ols_p)], OLS_se = [std(vector_ols_cons),std(vector_ols_x),std(vector_ols_p)],
    IV_mean = [mean(vector_iv_cons),mean(vector_iv_x),mean(vector_iv_p)], IV_se =[std(vector_iv_cons),std(vector_iv_x),std(vector_iv_p)]);
print("Result Summary")
print(result_summary)
```

    Monte Carlo Parameter Estimates 100 Random Samples of 500 Duopoly Markets  Logit Utility (sigma_d  = 3)3×5 DataFrame
    │ Row │ True_parameter │ OLS_mean  │ OLS_se    │ IV_mean  │ IV_se    │
    │     │ [90mFloat64[39m        │ [90mFloat64[39m   │ [90mFloat64[39m   │ [90mFloat64[39m  │ [90mFloat64[39m  │
    ├─────┼────────────────┼───────────┼───────────┼──────────┼──────────┤
    │ 1   │ 5.0            │ -0.762803 │ 0.418897  │ 5.03208  │ 0.847055 │
    │ 2   │ 2.0            │ 0.0195958 │ 0.115257  │ 1.99649  │ 0.301829 │
    │ 3   │ -1.0           │ 0.105563  │ 0.0831606 │ -1.00709 │ 0.166926 │


```julia

```

# Reference

Berry, Steven T. "Estimating discrete-choice models of product differentiation." <em>The RAND Journal of Economics<em> (1994): 242-262.

Berry, Steven, James Levinsohn, and Ariel Pakes. "Automobile prices in market equilibrium." <em>Econometrica: Journal of the Econometric Society<em> (1995): 841-890.
