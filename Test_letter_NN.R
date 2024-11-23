# Load the data

# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

# Recall the results of linear classifier from HW3
# Add intercept column
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

#  Apply LR (note that here lambda is not on the same scale as in NN due to scaling by training size)
out <- LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1)
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o') # around 19.5 if keep training
plot(out$error_test, type = 'o') # around 25 if keep training


# Apply neural network training with default given parameters
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)

plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 16.1


# microbenchmark(NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
#                         rate = 0.1, mbatch = 50, nEpoch = 150,
#                         hidden_p = 100, scale = 1e-3, seed = 12345), 
#                times = 10)

# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials


############ Implement a hyperparameter grid search in parallel ################

#### 300 total hyperparameter configurations tested
# library(doParallel)
# library(foreach)
# source("HelperFunctions.R")
# source("FunctionsNN.R")
# 
# # nworkers <- detectCores(logical = FALSE)
# nworkers <- detectCores(logical = FALSE) - 1

# rate_vals = seq(from = .09, to = .11, length.out = 5)
# lambda_vals = exp(seq(from = log(.000005), to = log(.001), length.out = nworkers))
# hidden_p_vals = c(200, 250, 300, 400)
# 
# 
# cl <- makeCluster(nworkers)
# registerDoParallel(cl)
# 
# grid_results <- foreach(hid_p = hidden_p_vals, .combine = rbind) %do% {
#   
#       foreach(lam = lambda_vals, .combine = rbind) %dopar% {
#         # source("HelperFunctions.R")
#         # source("FunctionsNN.R")
#         # load("parallel_data.rda")
#         data.frame(hidden_p = hid_p,
#                    lambda = lam,
#                    test_rate_vals(rate_vals, lam, hid_p, Xtrain, Ytrain, Xval, Yval, Xt, Yt))
#         # data.frame(hidden_p = hid_p,
#         #            lambda = lam,
#         #            data.frame(rate = rate_vals,
#         #                       train = rnorm(length(rate_vals)),
#         #                       val = rnorm(length(rate_vals)),
#         #                       test = rnorm(length(rate_vals))))
#       }
#   
# }
# 
# stopCluster(cl)
# 
# save(grid_results, file = "grid_results.rda")



load("grid_results.rda")

# Inspection of the Validation and training error gives this entry as the best 
# set of hyperparameters

best_param_row = 260
hidden_p = grid_results$hidden_p[best_param_row]
lambda = grid_results$lambda[best_param_row]
rate = grid_results$rate[best_param_row]
 
# out3 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = lambda,
#                 rate = rate, mbatch = 50, nEpoch = 150,
#                 hidden_p = hidden_p, scale = 1e-3, seed = 12345)
# 
# 
# plot(1:length(out3$error), out3$error, ylim = c(0, 70))
# lines(1:length(out3$error_val), out3$error_val, col = "red")
# 
# # Train/Validation Error
# mean(tail(out3$error)) # .629
# mean(tail(out3$error_val)) # 13.333
# 
# test_error = evaluate_error(Xt, Yt, out3$params$W1, out3$params$b1, out3$params$W2, out3$params$b2)
# test_error # 12.855
# 
# ######## Test if increasing mbatch & nEpoch has significant effects
# 
# out4 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = lambda,
#                 rate = rate, mbatch = 75, nEpoch = 200,
#                 hidden_p = hidden_p, scale = 1e-3, seed = 12345)
# 
# 
# plot(1:length(out4$error), out4$error, ylim = c(0, 70))
# lines(1:length(out4$error_val), out4$error_val, col = "red")
# 
# # Train/Validation Error
# mean(tail(out4$error)) # .888
# mean(tail(out4$error_val)) # 13.667
# 
# test_error = evaluate_error(Xt, Yt, out4$params$W1, out4$params$b1, out4$params$W2, out4$params$b2)
# test_error # 13.867
# #### Results are slightly worse
# 
# 
# ##### Test if decreasing mbatch and increasing nEpoch has a significant effect
# out5 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = lambda,
#                 rate = rate, mbatch = 35, nEpoch = 200,
#                 hidden_p = hidden_p, scale = 1e-3, seed = 12345)
# 
# 
# plot(1:length(out5$error), out5$error, ylim = c(0, 70))
# lines(1:length(out5$error_val), out5$error_val, col = "red")
# 
# # Train/Validation Error
# mean(tail(out5$error)) # 0.0
# mean(tail(out5$error_val)) # 11.5
# 
# test_error = evaluate_error(Xt, Yt, out5$params$W1, out5$params$b1, out5$params$W2, out5$params$b2)
# test_error # 12.12222
# #### Results are slightly better
# 
# 
# out6 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = lambda,
#                 rate = rate, mbatch = 25, nEpoch = 200,
#                 hidden_p = hidden_p, scale = 1e-3, seed = 12345)
# 
# 
# plot(1:length(out6$error), out6$error, ylim = c(0, 70))
# lines(1:length(out6$error_val), out6$error_val, col = "red")
# 
# # Train/Validation Error
# mean(tail(out6$error)) # 0.0
# mean(tail(out6$error_val)) # 14.1667
# 
# test_error = evaluate_error(Xt, Yt, out6$params$W1, out6$params$b1, out6$params$W2, out6$params$b2)
# test_error # 12.505
# #### Results are slightly worse



###################### Results of best model ###################################

out_best = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = lambda,
                    rate = rate, mbatch = 35, nEpoch = 200,
                    hidden_p = hidden_p, scale = 1e-3, seed = 12345)


plot(1:length(out_best$error), out5$error, ylim = c(0, 70))
lines(1:length(out_best$error_val), out5$error_val, col = "red")

# Train/Validation Error
mean(tail(out_best$error)) # 0.0
mean(tail(out_best$error_val)) # 11.5

test_error = evaluate_error(Xt, Yt, out_best$params$W1, out_best$params$b1, out_best$params$W2, out_best$params$b2)
test_error # 12.12222
