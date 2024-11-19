library(testthat)

################## Test Function initialize_bw #################################
for (i in 1:5) {
  p = sample(1:15, size = 1)
  hidden_p = sample(6:20, size = 1)
  K = sample(1:10, size = 1)
  
  init_params = initialize_bw(p, hidden_p, K)
  
  test_that("Check dimensions of initialized parameters",{
    expect_true(length(init_params$b1) == hidden_p) # first bias is a vector of dimension 'hidden p'
    expect_true(length(init_params$b2) == K) # second bias is a vector of dimension 'K'
    expect_true(all(dim(init_params$W1) == c(p, hidden_p))) # first weight matrix has dimensions 'p by hidden p'
    expect_true(all(dim(init_params$W2) == c(hidden_p, K))) # second weight matrix has dimensions 'hidden p by K'
  })
}





############# Test Function loss_grad_scores ###################################

p = 7
hidden_p = 15
K = 10
n = 500
sd_val = 4
drop_out_rate = .25

# Calculate the probabilities of each class by assigning "true" parameters to pass 
# through a fully connected NN. 
# This allows for the target function to be explicitly in the model space 
X <- matrix(rnorm(n*p), nrow = n)

# For W1, randomly generate values and drop some of them
W1_vals = rnorm(p*hidden_p, sd = sd_val)
W1_vals[sample(1:(hidden_p*p), size = floor(drop_out_rate*hidden_p*p), replace = F)] <- 0 # drop some of the values to 0
W1_true = matrix(W1_vals, nrow = p, ncol = hidden_p)

# for b1, randomly generate values and drop some of them 
b1_true = rnorm(hidden_p, sd = sd_val)
b1_true[sample(hidden_p, size = floor(drop_out_rate*hidden_p), replace = F)] <- 0 # drop some of the values to 0

# For W2, randomly generate values and drop some of them
W2_vals = rnorm(K*hidden_p, sd = sd_val)
W2_vals[sample(1:(hidden_p*K), size = floor(drop_out_rate*hidden_p*K), replace = F)] <- 0 # drop some of the values to 0
W2_true = matrix(W2_vals, nrow = hidden_p, ncol = K)


# for b2, randomly generate values and drop some of them 
b2_true = rnorm(K, sd = sd_val)
b2_true[sample(K, size = floor(drop_out_rate*K), replace = F)] <- 0 # drop some of the values to 0

# Pass X through fully connected network with true parameters
# First layer and bias
scores_true = X %*% W1_true + matrix(b1_true, nrow = n, ncol = hidden_p, byrow = T)
# RELU
scores_true = (abs(scores_true) + scores_true) / 2
# 2nd layer and bias
scores_true = scores_true %*% W2_true + matrix(b2_true, nrow = n, ncol = K, byrow = T)
# Calculate probabilities based on these scores
true_probs = exp(scores_true)
true_probs = true_probs/rowSums(true_probs)

# Sometimes values in scores_true are too large and result in NaN in true_probs
# If this occurs, find NaN values and replace with 1
if (anyNA(true_probs)) {
  true_probs[is.na(true_probs)] <- 1
}

# assign classifications of y based on true probs
y <- vector(mode = 'double', length = n)
for (i in 1:n) {
  y[i] = sample.int(K, size = 1, prob = true_probs[i, ]) - 1
}

out = loss_grad_scores(y, scores = scores_true, K = K)


######################## Test Function one pass ################################

pass = one_pass(X = X, y = y, K =  K, W1 = W1_true, b1 = b1_true, W2 = W2_true, b2 = b2_true, lambda = .01)
pass$grads$dW2




######################### Test Function NN_train ###############################
p = 15
hidden_p = 20
K = 2
n = 1200
nval = 200
sd_val = 5
drop_out_rate = .5

# Calculate the probabilities of each class by assigning "true" parameters to pass 
# through a fully connected NN. 
# This allows for the target function to be explicitly in the model space 
Xdata <- matrix(rnorm(n*p), nrow = n)

# For W1, randomly generate values and drop some of them
W1_vals = rnorm(p*hidden_p, sd = sd_val)
W1_vals[sample(1:(hidden_p*p), size = floor(drop_out_rate*hidden_p*p), replace = F)] <- 0 # drop some of the values to 0
W1_true = matrix(W1_vals, nrow = p, ncol = hidden_p)

# for b1, randomly generate values and drop some of them 
b1_true = rnorm(hidden_p, sd = sd_val)
b1_true[sample(hidden_p, size = floor(drop_out_rate*hidden_p), replace = F)] <- 0 # drop some of the values to 0

# For W2, randomly generate values and drop some of them
W2_vals = rnorm(K*hidden_p, sd = sd_val)
W2_vals[sample(1:(hidden_p*K), size = floor(drop_out_rate*hidden_p*K), replace = F)] <- 0 # drop some of the values to 0
W2_true = matrix(W2_vals, nrow = hidden_p, ncol = K)


# for b2, randomly generate values and drop some of them 
b2_true = rnorm(K, sd = sd_val)
b2_true[sample(K, size = floor(drop_out_rate*K), replace = F)] <- 0 # drop some of the values to 0

# Pass X through fully connected network with true parameters
# First layer and bias
scores_true = Xdata %*% W1_true + matrix(b1_true, nrow = n, ncol = hidden_p, byrow = T)
# RELU
scores_true = (abs(scores_true) + scores_true) / 2
# 2nd layer and bias
scores_true = scores_true %*% W2_true + matrix(b2_true, nrow = n, ncol = K, byrow = T)
# Calculate probabilities based on these scores
true_probs = exp(scores_true)
true_probs = true_probs/rowSums(true_probs)

# Sometimes values in scores_true are too large and result in NaN in true_probs
# If this occurs, find NaN values and replace with 1
if (anyNA(true_probs)) {
  true_probs[is.na(true_probs)] <- 1
}

# assign classifications of y based on true probs
ydata <- vector(mode = 'double', length = n)
for (i in 1:n) {
  ydata[i] = sample.int(K, size = 1, prob = true_probs[i, ]) - 1
}

Xval = Xdata[(n-nval+1):n, , drop = FALSE]
yval = ydata[(n-nval+1):n]
Xtrain = Xdata[1:(n-nval), , drop = FALSE]
ytrain = ydata[1:(n-nval)]

out = NN_train(X = Xtrain, y = ytrain, Xval = Xval, yval = yval, lambda = .05, rate = .1, 
         mbatch = 20, nEpoch = 500, hidden_p = hidden_p, scale = 1e-3, seed = 12345)


