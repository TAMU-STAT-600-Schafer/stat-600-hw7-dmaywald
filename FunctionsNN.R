# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  # [ToDo] Initialize intercepts as zeros
  # Intercepts are zero
  b1 = rep(0, hidden_p)
  b2 = rep(0, K)
  
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  
  set.seed(seed)
  W1 = scale * matrix(rnorm(p * hidden_p), p, hidden_p)
  W2 = scale * matrix(rnorm(hidden_p * K), hidden_p, K)
  
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  
  n = length(y)
  # [ToDo] Calculate loss when lambda = 0
  expScores = exp(scores)
  probs = expScores / rowSums(expScores)
  
  # # Sometimes values in scores_true are too large and result in NaN in true_probs
  # # If this occurs, find NaN values and replace with 1
  # if (anyNA(probs)) {
  #   probs[is.na(probs)] <- 1
  # }
  
  loss = 0
  
  for (i in 1:K) {
    loss = loss + sum(log(probs[y == (i-1), i]))
  }
  
  loss = -1 * (loss/n)
               
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  # use rowwise maximum of SCORES rather than probs. 
  # scores are less likely to have "tie breakers" needed
  # not a perfect solution, sometimes tie breakers are still needed
  # "tie breakers" are done by selecting the first column containing the minimal value
  error = 100 * (1 - mean(.Internal(max.col(scores, 1)) == (y+1)))
  
  grad = probs
  
  for (i in 1:K) {
    idx = y == (i-1)
    grad[idx,i] <- grad[idx,i] - 1
  }
  
  grad = grad/n
  
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}


loss_grad_scores_2 <- function(y, scores, K){
  
  n = length(y)
  # [ToDo] Calculate loss when lambda = 0
  expScores = exp(scores)
  probs = expScores / rowSums(expScores)
  
  # # Sometimes values in scores_true are too large and result in NaN in true_probs
  # # If this occurs, find NaN values and replace with 1
  # if (anyNA(probs)) {
  #   probs[is.na(probs)] <- 1
  # }
  
  loss = 0
  
  for (i in 1:K) {
    loss = loss + sum(log(probs[y == (i-1), i]))
  }
  
  loss = -1 * (loss/n)
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  # use rowwise maximum of SCORES rather than probs. 
  # scores are less likely to have "tie breakers" needed
  # not a perfect solution, sometimes "tie breakers" are still needed
  # "tie breakers" are done by selecting the first column containing the minimal value
  # error = 100 * (1 - mean(max.col(scores) == (y+1)))
  
  error = 100 * (1 - mean(.Internal(max.col(scores, 1)) == (y+1)))
  
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  
  # This for loop operates similarly to "table(y)/n",
  # but "table(y)/n" will not make a save counts of classifications that does
  # not appear
  # Ex: if y = c(0,0,2,3,4), then table(y)/n gives ".4 .2 .2 .2" instead of ".4 0 .2 .2 .2"
  
  grad = probs/n
  
  n.inv = 1/n
  
  for (i in 1:K) {
    idx = y == (i-1)
    grad[idx,i] <- grad[idx,i] - n.inv
  }
  
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){
  # [To Do] Forward pass
  # From input to hidden and add bias
  H <- X %*% W1 + matrix(b1, nrow = length(y), ncol = length(b1), byrow = T)
  
  # ReLU
  idx = H < 0
  H[idx] <- 0
  
  # From hidden to output scores
  # print(dim(H))
  # print(dim(W2))
  # print(dim(matrix(b2, nrow = length(y), ncol = length(b2), byrow = T)))
  scores = H %*% W2 + matrix(b2, nrow = nrow(H), ncol = ncol(W2), byrow = T)
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out = loss_grad_scores(y, scores, K)
  
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 = crossprod(H, out$grad) + lambda*W2
  db2 = colSums(out$grad)
  
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dH  = tcrossprod(out$grad, W2)
  dH[idx] = 0
  dW1 = crossprod(X, dH) + lambda*W1
  db1 = colSums(dH)
  
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = out$loss, error = out$error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

one_pass_2 <- function(X, y, K, W1, b1, W2, b2, lambda){
  
  # [To Do] Forward pass
  # From input to hidden and add bias
  H <- X %*% W1 + matrix(b1, nrow = length(y), ncol = length(b1), byrow = T)
  
  # ReLU
  idx = H < 0
  H[idx] <- 0
  
  # From hidden to output scores
  # print(dim(H))
  # print(dim(W2))
  # print(dim(matrix(b2, nrow = length(y), ncol = length(b2), byrow = T)))
  scores = H %*% W2 + matrix(b2, nrow = nrow(H), ncol = ncol(W2), byrow = T)
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out = loss_grad_scores(y, scores, K)
  
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 = crossprod(H, out$grad) + lambda*W2
  db2 = colSums(out$grad)
  
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dH  = tcrossprod(out$grad, W2)
  dH[idx] = 0
  dW1 = crossprod(X, dH) + lambda*W1
  db1 = colSums(dH)
  
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = out$loss, error = out$error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  # [ToDo] Forward pass to get scores on validation data
  
  # First layer and bias
  H = Xval %*% W1 + matrix(b1, nrow = length(yval), ncol = length(b1), byrow = T)
  
  # RELU
  H = (abs(H) + H) / 2
  
  # 2nd layer and bias
  scores = H %*% W2 + matrix(b2, nrow = nrow(H), ncol = ncol(W2), byrow = T)
  
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  error = 100 * (1 - mean(.Internal(max.col(scores, 1)) == (yval+1)))
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n/mbatch)
  K = max(c(y, yval)) + 1

  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  p = ncol(X)
  
  init_params = initialize_bw(
    p = p,
    hidden_p = hidden_p,
    K = K,
    scale = scale,
    seed = seed
  )
  
  W1 = init_params$W1
  W2 = init_params$W2
  b1 = init_params$b1
  b2 = init_params$b2
  
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    # Accumulate error over batches, and compute average
    cur_err = 0
    # [ToDo] For each batch
    for (j in 1:nBatch) {
      #  - do one_pass to determine current error and gradients
      pass = one_pass(X = X[batchids == j, , drop = FALSE], y = y[batchids == j], K = K,
                      W1 =  W1, b1 = b1, W2 = W2, b2 = b2, lambda = lambda)
      
      # print(pass$error)
      cur_err = cur_err + pass$error
      
      #  - perform SGD step to update the weights and intercepts

      W1 <- W1 - rate * pass$grads$dW1
      b1 <- b1 - rate * pass$grads$db1
      W2 <- W2 - rate * pass$grads$dW2
      b2 <- b2 - rate * pass$grads$db2
      
    }
    # print(pass$grads$db2)
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    error[i] = cur_err/nBatch
    # - validation error using evaluate_error function
    error_val[i] = evaluate_error(Xval, yval, W1, b1, W2, b2)
  }
  
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}