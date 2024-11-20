library(testthat)
library(microbenchmark)
library(profvis)

source("FunctionsNN.R")
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


################### Helper function for generating data ########################

gen_data <- function(p, hidden_p, K, n, sd_val, drop_out_rate, nval = 0, seed = 0){
  set.seed(seed)
  
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
  # if (anyNA(true_probs)) {
  #   true_probs[is.na(true_probs)] <- 1
  # }
  
  # assign classifications of y based on true probs
  ydata <- vector(mode = 'double', length = n)
  for (i in 1:n) {
    ydata[i] = sample.int(K, size = 1, prob = true_probs[i, ]) - 1
  }
  
  Xval = NULL; yval = NULL
  
  if(nval > 0){
    Xval = Xdata[(n-nval+1):n, , drop = FALSE]
    yval = ydata[(n-nval+1):n]
  }

  Xtrain = Xdata[1:(n-nval), , drop = FALSE]
  ytrain = ydata[1:(n-nval)]
  
  return(list(X = Xtrain, y = ytrain, Xval = Xval, yval = yval,
              W1_true = W1_true, b1_true = b1_true, W2_true = W2_true, b2_true = b2_true,
              scores_true = scores_true, probs_true = true_probs))
  
}


############# Test Function loss_grad_scores ###################################
for (i in 1:5) {
  test_that("Check output of loss_grad_scores", {
    p = round(runif(1, 5, 12))
    hidden_p = round(runif(1, 5, 20))
    K = round(runif(1, 5, 10))
    n = round(runif(1, 100, 500))
    sd_val = runif(1, 1, 3)
    drop_out_rate = runif(1, .5, .9)
    
    params <- gen_data(p = p, hidden_p = hidden_p, K = K, n = n,
                       sd_val = sd_val, drop_out_rate = drop_out_rate)
    
    out = loss_grad_scores(params$y, scores = params$scores_true, K = K)
    
    
    expect_type(out$loss, "double") # expect double output for loss
    expect_type(out$error, "double") # expect double output for error
    expect_type(out$grad, "double") # expect double output for gradient
    expect_true(all(dim(out$grad) == c(n, K))) # expect dimension of gradient output to be n x K
    
    # Expect that when we take scores of the opposite sign, the loss and error increases
    opp_out = loss_grad_scores(params$y, scores = -params$scores_true, K)
    expect_lte(out$loss , opp_out$loss)
    expect_lte(out$error , opp_out$error)
    
    # Expect that the gradient is larger (in absolute value) on average when comparing scores_true to -1*scores_true
    # (really it's because gradient = (scores - one_hot(y))/n
    # this means that when the scores are closer to a one-hot-encoding matrix of y, the gradient is closer to 0) 
    expect_lte(mean(abs(out$grad)) , mean(abs(opp_out$grad)))
  })
}


#### Speed test different versions of loss_grad_scores
p = 10
hidden_p = 15
K = 10
n = 50
sd_val = 2
drop_out_rate = .5

params <- gen_data(p = p, hidden_p = hidden_p, K = K, n = n,
                   sd_val = sd_val, drop_out_rate = drop_out_rate)

expect_true(max(abs(
  loss_grad_scores(params$y, scores = params$scores_true, K = K)$grad -
    loss_grad_scores_2(params$y, scores = params$scores_true, K = K)$grad)) < 1e-10)

microbenchmark(
  ver_1 = loss_grad_scores(params$y, scores = params$scores_true, K = K),
  ver_2 = loss_grad_scores_2(params$y, scores = params$scores_true, K = K)
)


######################## Test Function one pass ################################
for(i in 1:5){
  test_that("Check output of one_pass ", {
    p = round(runif(1, 5, 12))
    hidden_p = round(runif(1, 5, 20))
    K = round(runif(1, 5, 10))
    n = round(runif(1, 100, 500))
    sd_val = runif(1, 1, 3)
    drop_out_rate = runif(1, .5, .9)
    lambda = runif(1, 1e-4, 9e-2)
    
    params <- gen_data(p = p, hidden_p = hidden_p, K = K, n = n,
                       sd_val = sd_val, drop_out_rate = drop_out_rate)
    
    pass = one_pass(X = params$X, y = params$y, K =  K, W1 = params$W1_true,
                    b1 = params$b1_true, W2 = params$W2_true, b2 = params$b2_true, lambda = lambda)
    # compare pass at X and -X
    opp_pass = one_pass(X = -params$X, y = params$y, K =  K, W1 = params$W1_true,
                        b1 = params$b1_true, W2 = params$W2_true, b2 = params$b2_true, lambda = lambda)
    
    # Check dimensionality of gradient outputs
    expect_true(all(dim(pass$grads$dW1) == dim(params$W1_true)))
    expect_true(all(dim(pass$grads$dW2) == dim(params$W2_true)))
    expect_true(length(pass$grads$db1) == length(params$b1_true))
    expect_true(length(pass$grads$db2) == length(params$b2_true))
    
    # Check that loss and error are better at X than -X. This makes sure our output makes sense
    # since the data was generated at X 
    expect_lte(pass$loss, opp_pass$loss)
    expect_lte(pass$error, opp_pass$error)
    
    # Should expect larger gradients (in absolute value) on average at X than at -X since we are "closer to the truth"
    # (really it's because gradient = (scores - one_hot(y))/n
    # this means that when the scores are closer to a one-hot-encoding matrix of y, the gradient is closer to 0) 
    expect_lte(mean(abs(pass$grads$dW1)), mean(abs(opp_pass$grads$dW1)))
    expect_lte(mean(abs(pass$grads$dW2)), mean(abs(opp_pass$grads$dW2)))
    expect_lte(mean(abs(pass$grads$db1)), mean(abs(opp_pass$grads$db1)))
    expect_lte(mean(abs(pass$grads$db2)), mean(abs(opp_pass$grads$db2)))
  })
}

#### Speed test different versions of one_pass

p = 10
hidden_p = 15
K = 10
n = 50
sd_val = 2
drop_out_rate = .5
lambda = .01

params <- gen_data(p = p, hidden_p = hidden_p, K = K, n = n,
                   sd_val = sd_val, drop_out_rate = drop_out_rate)
pass1 = one_pass(X = params$X, y = params$y, K =  K, W1 = params$W1_true,
                 b1 = params$b1_true, W2 = params$W2_true, b2 = params$b2_true, lambda = lambda)

pass2 = one_pass_2(X = params$X, y = params$y, K =  K, W1 = params$W1_true,
                 b1 = params$b1_true, W2 = params$W2_true, b2 = params$b2_true, lambda = lambda)

expect_true(all(
  max(abs(pass1$grads$dW1 - pass2$grads$dW1)) < 1e-10,
  max(abs(pass1$grads$dW2 - pass2$grads$dW2)) < 1e-10,
  max(abs(pass1$grads$db1 - pass2$grads$db1)) < 1e-10,
  max(abs(pass1$grads$db2 - pass2$grads$db2)) < 1e-10
))


microbenchmark(
  ver_1 = one_pass(X = params$X, y = params$y, K =  K, W1 = params$W1_true,
                   b1 = params$b1_true, W2 = params$W2_true, b2 = params$b2_true, lambda = lambda),
  ver_2 = one_pass_2(X = params$X, y = params$y, K =  K, W1 = params$W1_true,
                     b1 = params$b1_true, W2 = params$W2_true, b2 = params$b2_true, lambda = lambda)
)


######################### Test Function NN_train ###############################
p = 15
hidden_p = 30
K = 10
n = round(runif(1, 500, 1000))
nval = round(n/6)
sd_val = 2
drop_out_rate = .5
lambda = runif(1, 1e-4, 9e-2)

params <- gen_data(p = p, hidden_p = hidden_p, K = K, n = n,
                   sd_val = sd_val, drop_out_rate = drop_out_rate, nval = nval)

out = NN_train(X = params$X, y = params$y, Xval = params$Xval, yval = params$yval,
               lambda = lambda, rate = .1, mbatch = 20, nEpoch = 100,
               hidden_p = hidden_p, scale = 1e-3, seed = 12345)



# Regress log(error) on epochs and look at slope coefficient. 
# If slope coefficient is negative, then error goes down over epochs
model_mat <- cbind(1, 1:length(out$error))
expect_true(solve(crossprod(model_mat), crossprod(model_mat, log(out$error)))[2] < 0)


profvis::profvis(NN_train(X = params$X, y = params$y, Xval = params$Xval, yval = params$yval,
                         lambda = lambda, rate = .1, mbatch = 20, nEpoch = 100,
                         hidden_p = hidden_p, scale = 1e-3, seed = 12345))
