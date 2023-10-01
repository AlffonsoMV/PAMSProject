 # PAMS's Project

## Author
- **Name:** Alfonso Mateos Vicente
- **Email:** alfonso.mateos-vicente@eleves.enpc.fr
- **Institution:** École des Ponts ParisTech
- **Tutor:** PhD. Noé Blassel

## Introduction
This project explores the simulation of random variables from given probability distributions. It primarily focuses on two methods for simulating these variables: the Inverse CDF method and the Rejection Sampling method. 

## Sections
### Simulating Random Variables
#### Overview
The aim of this section is to introduce the simulation of random variables from a probability distribution already given, focusing on the Inverse CDF method and the Rejection Sampling method.

#### Inverse CDF Method
This method is a basic technique to generate pseudo-random numbers from any probability distribution, given its cumulative distribution function (CDF). It is based on the assumption that if \(X\) is a random variable with cumulative distribution function \(F\) and probability density function \(f\), then the cumulative distribution function of \(Y = F^{-1}(U)\) behaves like \(F\), and the probability density function of \(Y\) is \(f\).

#### Rejection Sampling Method
This method is used to generate random numbers from a distribution by using an easy-to-sample distribution. It envelopes the desired distribution with a known distribution scaled by a factor \(M\). The histogram of generated random numbers using this method aligns remarkably well with the target distribution. The error to the theoretical mean as the number of trials increases is also discussed.

### Variance Reduction Techniques
#### Overview
This section explores various variance reduction techniques, such as Antithetic Variates, Control Variates, Importance Sampling, and Stratified Sampling. These techniques are applied to estimate integrals of different functions, such as Parabola, Gaussian, Sine, Polynomial, and Exponential functions, to compare the error in each method.

#### Method-wise Error Comparison
The method-wise error for each function is illustrated, and it is observed that stratified sampling stands out for its rapid convergence. However, the importance sampling lags due to applying a general normal distribution for all functions. A thorough exploration involving varied problems is essential to cement the initial conclusions while maintaining professional and scientific rigor.

## Conclusion
The project provides a comprehensive exploration of simulating random variables and applying variance reduction techniques to estimate integrals of different functions. It offers insights into the performance and application of each method in various scenarios.

## Contact
For any queries or further clarification, please contact:
- **Alfonso Mateos Vicente**
- **Email:** alfonso.mateos-vicente@eleves.enpc.fr
