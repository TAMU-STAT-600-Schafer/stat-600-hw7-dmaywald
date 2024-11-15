# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL) {
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  
  if (!(all(X[, 1] == rep(1, nrow(X))))) {
    stop("First column of X is not a column of all 1s. Stopping Execution")
  }
  
  if (!(all(Xt[, 1] == rep(1, nrow(Xt))))) {
    stop("First column of Xt is not a column of all 1s. Stopping Execution")
  }
  
  # Check for compatibility of dimensions between X and y.
  # I check if y can be made as a matrix as an adversarial user check
  if (!(nrow(X) == nrow(matrix(y)))) {
    stop("X and Y do not have the same number of rows. Check dimension compatability")
  }
  
  # Check for compatibility of dimensions between Xt and yt
  # I check if yt can be made as a matrix as an adversarial user check
  if (!(nrow(Xt) == nrow(matrix(yt)))) {
    stop("Xt and Yt do not have the same number of rows. Check dimension compatability")
  }
  
  # Check for compatibility of dimensions between X and Xt
  if (!(ncol(X) == ncol(Xt))) {
    stop("X and Xt do not have the same number of columns. Check dimension compatability")
  }
  
  # Check eta is positive
  if (eta <= 0) {
    stop("Eta needs to be strictly positive")
  }
  
  # Check lambda is non-negative
  if (lambda < 0) {
    stop("lambda needs to be non-negative")
  }
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes.
  # If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (!is.null(beta_init)) {
    if (!is.matrix(beta_init)) {
      stop("Supplied argument beta_init is not a matrix!")
    }
    
    K <- ncol(beta_init)
    if (length(unique(y)) != K) {
      K <- min(K, length(unique(y)))
      warning("Number of unique classifications found in y does not match the column dimension of supplied beta_init")
    }
  }
  
  if (is.null(beta_init)) {
    # Change to matrix of 0's
    K <- length(unique(y)) # Calculate the number of classes for Y
    beta_init <- matrix(0, nrow = ncol(X), ncol = K) 
  }
  
  # Check if numIter > 0. If not, return output without any iteration
  if(!(numIter>0)){
    warning("Supplied numIter is not greater than 0. Function LRMultiClass() is returning output with no iterative solution. If no beta_init was supplied by user, then a matrix of 0 will be returned")
    expXB <- exp(crossprod(t(X), beta_init))
    Pk <- expXB / rowSums(expXB)
    # logPk <- log(Pk)
    
    # Calculate double sum found in the objective function
    temp <- 0
    for (i in 1:K) {
      temp <- temp + sum(log(Pk[y == (i-1), i]))
    }
    
    
    # initialize objective vector with repeated initial objective function evaluation
    # note that the frobenius norm of beta is equivalent to the double sum
    objective <- c(-1 * temp + (lambda / 2) * norm(beta_init, "F") ^ 2)
    
    
    
    # Needed for quickly calculating train accuracy and test accuracy
    ntrain = nrow(X)
    ntest = nrow(Xt)
    
    
    # Get training accuracy by calculating mean of indicator function I(pred == y)
    # where pred is row wise maximum of expXB
    # would it be faster to apply which.max to rows of expXB
    # or would it be faster to for loop?
    trainAcc <- 0
    for (n in 1:ntrain) {
      if (which.max(Pk[n, ])-1 == y[n]) {
        trainAcc <- trainAcc + 1
      }
    }
    error_train <- c(1 - trainAcc / ntrain)
    
    
    # Get testing accuracy by calculating mean of indicator function I(pred == yt)
    # where pred is row wise maximum of Xt %*% beta_init
    # would it be faster to apply which.max to rows of expXB
    # or would it be faster to for loop?
    XtestBeta <- crossprod(t(Xt), beta_init)
    testAcc <- 0
    for (n in 1:ntest) {
      if (which.max(XtestBeta[n, ])-1 == yt[n]) {
        testAcc <- testAcc + 1
      }
    }
    
    error_test <- c(1 - testAcc / ntest)
    return(list(beta = beta_init, error_train = 100*error_train, error_test = 100*error_test, objective =  objective))
  }
  
  
  
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  p <- ncol(X) # dimension of training data
  
  # Need to calculate all of Pk in each iteration. This is for the initial step
  expXB <- exp(crossprod(t(X), beta_init))
  Pk <- expXB / rowSums(expXB)
  # print(Pk)
  # logPk <- log(Pk)
  
  # Calculate double sum found in the objective function
  temp <- 0
  for (i in 1:K) {
    temp <- temp + sum(log(Pk[y == (i-1), i]))
  }
  
  
  # initialize objective vector with repeated initial objective function evaluation
  # note that the frobenius norm of beta is equivalent to the double sum
  objective <- rep(-1 * temp + (lambda / 2) * norm(beta_init, "F") ^ 2, numIter + 1)
  
  
  
  # Needed for quickly calculating train accuracy and test accuracy
  ntrain = nrow(X)
  ntest = nrow(Xt)
  
  
  # Get training accuracy by calculating mean of indicator function I(pred == y)
  # where pred is row wise maximum of expXB
  # would it be faster to apply which.max to rows of expXB
  # or would it be faster to for loop?
  # initialize error_train as repeated initial error
  trainAcc <- 0
  for (n in 1:ntrain) {
    if (which.max(Pk[n, ])-1 == y[n]) {
      trainAcc <- trainAcc + 1
    }
  }
  error_train <- rep(1 - trainAcc / ntrain, numIter + 1)
  
  # Get testing accuracy by calculating mean of indicator function I(pred == yt)
  # where pred is row wise maximum of Xt %*% beta_init
  # would it be faster to apply which.max to rows of expXB
  # or would it be faster to for loop?
  # initialize error_test as repeated initial error
  XtestBeta <- Xt %*% beta_init
  testAcc <- 0
  for (n in 1:ntest) {
    if (which.max(XtestBeta[n, ])-1 == yt[n]) {
      testAcc <- testAcc + 1
    }
  }
  
  error_test <- rep(1 - testAcc / ntest, numIter + 1)
  
  # rename beta_init to beta_mat (the output beta)
  # I didn't like using the name "beta" since there's a function in base R called beta
  beta_mat <- beta_init
  
  
  for (j in 2:(numIter + 1)) {
    ## Newton's method cycle - implement the update EXACTLY numIter iterations
    ##########################################################################
    
    # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
    
    for (k in 1:K) {
      
      beta_mat[, k] = beta_mat[, k] -
        # this follows formula at bottom of page 1 in README.pdf
        eta * solve(crossprod(X, (Pk[, k] * (1 - Pk[, k]) * X)) + diag(lambda, p),
                    crossprod(X, Pk[, k] - (y == (k-1))) + lambda * beta_mat[, k])
    }
    
    # recalculate probabilities and objective function,
    # as well as test error and training error
    
    expXB <- exp(X %*% beta_mat)
    Pk <- expXB / rowSums(expXB)
    # logPk <- log(Pk)
    
    # Calculate double sum found in the objective function
    temp <- 0
    for (i in 1:K) {
      temp <- temp + sum(log(Pk[y == (i-1), i]))
    }
    
    objective[j] <- -1 * temp + (lambda / 2) * norm(beta_mat, "F") ^ 2
    
    # Get training accuracy by calculating mean of indicator function I(pred == y)
    # where pred is row wise maximum of expXB
    # would it be faster to apply which.max to rows of expXB
    # or would it be faster to for loop?
    trainAcc <- 0
    for (n in 1:ntrain) {
      if (which.max(Pk[n, ])-1 == y[n]) {
        trainAcc <- trainAcc + 1
      }
    }
    error_train[j] <- 1 - trainAcc / ntrain
    
    # Get testing accuracy by calculating mean of indicator function I(pred == yt)
    # where pred is row wise maximum of Xt %*% beta_init
    # would it be faster to apply which.max to rows of expXB
    # or would it be faster to for loop?
    XtestBeta <- Xt %*% beta_mat
    testAcc <- 0
    for (n in 1:ntest) {
      if (which.max(XtestBeta[n, ])-1 == yt[n]) {
        testAcc <- testAcc + 1
      }
    }
    error_test[j] <- 1 - testAcc / ntest
  }
  
  ## Return output
  ##########################################################################
  # beta_mat - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta_mat, error_train = 100*error_train, error_test = 100*error_test, objective =  objective))
}