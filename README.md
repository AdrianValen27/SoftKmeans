# Stan Soft K-means Clustering

### 1. Installation

a. In your terminal, you will need to install Numpy, Pandas, and Stan into your Python virtual environment using
```
pip install numpy pandas cmdstanpy
```

b. Once these libraries are installed, you can create a new Jupyter notebook and code from there. The Jupyter notebook I will be using is called `demo.ipynb`.

### 2. Using Stan's Soft K-means Clustering algorithm

a. Create a `.stan` file, this demo will call it `soft_kmeans.stan` to store our Stan code that will be used by our Jupyter notebook.

b. Next, in that Stan file, copy and paste this reference code below.
```
data {
  int<lower=0> N;        // number of data points
  int<lower=1> D;        // number of dimensions
  int<lower=1> K;        // number of clusters
  array[N] vector[D] y;  // observations
}
transformed data {
  real<upper=0> neg_log_K;
  neg_log_K = -log(K);
}
parameters {
  array[K] vector[D] mu; // cluster means
}
transformed parameters {
  array[N, K] real<upper=0> soft_z; // log unnormalized clusters
  for (n in 1:N) {
    for (k in 1:K) {
      soft_z[n, k] = neg_log_K
                     - 0.5 * dot_self(mu[k] - y[n]);
    }
  }
}
model {
  // prior
  for (k in 1:K) {
    mu[k] ~ std_normal();
  }

  // likelihood
  for (n in 1:N) {
    target += log_sum_exp(soft_z[n]);
  }
}
```

c. Now you're set! We can now begin working in our Jupyter notebook. If you would like to learn more about the clustering models Stan offers, you can visit their user guide [here](https://mc-stan.org/docs/stan-users-guide/clustering.html) to learn more.