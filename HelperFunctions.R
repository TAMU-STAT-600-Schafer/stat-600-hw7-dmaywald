#################### Helper functions for Testing ##############################

test_lambda_vals <- function(lambda_vals, r_val, hid_p, Xtrain, Ytrain, Xval, Yval, Xt, Yt){
  results <- setNames(data.frame(matrix(ncol = 4, nrow = length(lambda_vals))), c("lambda","train", "val", "test"))
  
  for (i in 1:length(lambda_vals)) {
    temp = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = lambda_vals[i],
                    rate = r_val, mbatch = 50, nEpoch = 150,
                    hidden_p = hid_p, scale = 1e-3, seed = 12345)
    results[i, ] = c(lambda_vals[i],
                     mean(tail(temp$error)),
                     mean(tail(temp$error_val)),
                     evaluate_error(Xt, Yt, temp$params$W1, temp$params$b1, temp$params$W2, temp$params$b2))
  }
  
  return(results)
}

test_rate_vals <- function(rate_vals, lam, hid_p, Xtrain, Ytrain, Xval, Yval, Xt, Yt){
  results <- setNames(data.frame(matrix(ncol = 4, nrow = length(rate_vals))), c("rate","train", "val", "test"))
  
  for (i in 1:length(rate_vals)) {
    temp = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = lam,
                    rate = rate_vals[i], mbatch = 50, nEpoch = 150,
                    hidden_p = hid_p, scale = 1e-3, seed = 12345)
    results[i, ] = c(rate_vals[i],
                     mean(tail(temp$error)),
                     mean(tail(temp$error_val)),
                     evaluate_error(Xt, Yt, temp$params$W1, temp$params$b1, temp$params$W2, temp$params$b2))
  }
  
  return(results)
}


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