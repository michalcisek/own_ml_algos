# Implementation ----------------------------------------------------------

#CONCEPT OF ALGORITHM
#set starting point - choose randomly or set to zeros
#set maximum number of iterations, threshold for minimum change between iterations 
#and learning rate
#while below maximum number of iterations and value between iterations greater
#than threshold do:
#- calculate derivative of function
#- change point in the opposite direction to derivative times learning rate
#- calculate difference in function values for old and new point
#DONE!

#function to compute partial derivative of function using approximation
#n_arg - partial derivative with respect to which argument
calc_partial_deriv_approx <- function(fun, x, n_arg, h){
  x_new <- x
  x_new[[n_arg]] <- x_new[[n_arg]] + h
  
  partial_deriv <- (do.call(fun, x_new) - do.call(fun, x))/h
  
  return(partial_deriv)
}

own_gradient_descent <- function(fun, niter, epsilon, lr){
  n_args <- length(as.list(args(fun))) - 1
  
  x <- as.list(rep(0, n_args))

  h <- 1e-9
  iter <- 1
  diff <- epsilon + 1

  while((iter < niter) & (diff > epsilon)){
    fun_val <- do.call(fun, x)
    
    deriv <- sapply(1:n_args, function(arg) calc_partial_deriv_approx(fun, x, arg, h))
    
    x <- lapply(1:length(x), function(elem) x[[elem]] - lr*deriv[elem])
    
    diff <- fun_val - do.call(fun, x)
    
    iter <- iter + 1
  }
  
  if(niter == iter){ 
    warning("Convergence not achieved!")
  } else {
    message(paste0("Convergence achieved! Number of iterations: ", iter))
  }
  return(x)
}

# Testing algorithm -------------------------------------------------------

#1st case - univariate function - minimum at 3
fun <- function(x) (x-3)^2

own_gradient_descent(fun, 100000, 1e-7, 0.01)

#2nd case - bivariate function - minimum at (-1, 2)
fun <- function(x1, x2) (x1-1)^2 + (x2+2)^2

#visualization of function
x1 <- seq(-4, 4, length.out = 1000)
x2 <- seq(-4, 4, length.out = 1000)
gr <- expand.grid(x1, x2)
colnames(gr) <- c("x1", "x2")
y <- fun(gr$x1, gr$x2)
df <- cbind(gr, y)

minimum <- df[which(df$y == min(df$y)), ]

library(ggplot2)

ggplot(df, aes(x = x1, y = x2, z = y))+
  geom_contour()+
  geom_point(data = minimum, aes(x = x1, y = x2))

own_gradient_descent(fun, 100000, 1e-9, 0.1)
