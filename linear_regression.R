# Implementation ----------------------------------------------------------

#CONCEPT OF ALGORITHM
#set the cost functions as mean squared error (the sum of squared error does not work - WHY??)
#initialize random coefficients
#use gradient descent
#DONE!


own_lm <- function(y, x){
  #convert data to matrix form
  x <- as.matrix(x)
  y <- as.matrix(y)
  
  #set up cost function
  cost_function <- function(...){ sum((y - (cbind(rep(1, nrow(x)), x) %*% unlist(...)))^2)/nrow(x) }
  
  #initialize coefficients
  b <- as.list(rep(0, ncol(x) + 1))
  
  #parameters for gradient descent
  epsilon <- 1e-9
  niter <- 100000
  lr <- 0.01
  h <- 1e-9
  iter <- 1
  diff <- epsilon + 1
  
  #function to calculate partial derivative
  calc_partial_deriv_approx <- function(fun, x, n_arg, h){
    x_new <- x
    x_new[[n_arg]] <- x_new[[n_arg]] + h
    
    partial_deriv <- (fun(x_new) - fun(x))/h
    
    return(partial_deriv)
  }
  
  #gradient descent
  while((iter < niter) & (abs(diff) > epsilon)){
    fun_val <- cost_function(b)
    
    deriv <- sapply(1:length(b), function(arg) calc_partial_deriv_approx(cost_function, b, arg, h))
    
    b <- lapply(1:length(b), function(elem) b[[elem]] - lr*deriv[elem])
    
    diff <- fun_val - cost_function(b)
    
    iter <- iter + 1
  }
  
  b <- unlist(b)
  names(b) <- c("(Intercept)", paste0("x", 1:ncol(x)))
  
  return(b)
}

# Testing algorithm -------------------------------------------------------

df <- data.frame(y = runif(1000, 1, 100), x1 = rnorm(1000), x2 = rbeta(1000, 1, 2))

lm_model <- lm(y ~ x1 + x2, data = df)
lm_model$coefficients
own_lm(df$y, df[, 2:3])


df <- data.frame(y = runif(1000, 1, 100), x1 = rnorm(1000), x2 = rbeta(1000, 1, 2),
                 x3 = rnorm(1000, 1, 1), x4 = rnorm(1000, 3, 0.2))

lm_model <- lm(y ~ x1 + x2 + x3 + x4, data = df)
lm_model$coefficients
own_lm(df$y, df[, 2:5])

