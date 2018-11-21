# Implementation ----------------------------------------------------------

#CONCEPT OF ALGORITHM
#standarize features
#for i in points_test
#for j in points_train
#calculate distance between i and j
#get the k nearest j points
#classification: get the most frequent target and assign as prediction (if tie, choose randomly
#regression: calculate mean of target value
#DONE! 

own_knn <- function(train, test, y, k){
  #standarize features
  train <- sapply(train, function(x) (x - mean(x))/sd(x))
  test <- sapply(test, function(x) (x - mean(x))/sd(x))
  
  y_preds <- rep(NA, nrow(test))
  for(i in 1:nrow(test)){
    dist <- rep(NA, nrow(train))
    #calculate distance between test point and all training points
    for(j in 1:nrow(train)){
      dist[j] <- sqrt(sum((train[j, ] - test[i, ])^2))
    }
    
    which_smlst_dist <- which(rank(dist) %in% 1:k)
    y_pred <- y[which_smlst_dist]
    
    #calculate prediction
    if(is.factor(y) | is.character(y)){
      most_frq_val <- max(table(y_pred))
      most_frq_val <- table(y_pred)[table(y_pred) == most_frq_val]
      if(length(most_frq_val) > 1){
        y_preds[i] <- sample(names(most_frq_val), 1)
      } else{
        y_preds[i] <- names(most_frq_val)
      }
    } else if(is.numeric(y)){
      y_preds[i] <- mean(y_pred)
    } else {
      stop("Y is instance of unsupported class!")
    }
  }
  return(y_preds)
}

# Testing algorithm -------------------------------------------------------
library(MASS)
library(class)

#data generation
set.seed(56283)
df <- rbind(mvrnorm(500, c(0, 0), matrix(c(2, 1, 1, 2), 2, 2)),
            mvrnorm(500, c(2, -2), matrix(c(2, 1, 1, 2), 2, 2)))
df <- cbind(df, c(rep(0, 500), rep(1, 500)))
df <- as.data.frame(df)
colnames(df) <- c("x1", "x2", "y")
df$y <- as.factor(df$y)

#visualize data
plot(df$x1, df$x2, col = df$y)

#comparison of own implementation  vs implementation from class library
y_pred_orig <- knn(df[, 1:2], df[, 1:2], df$y, 5)
y_pred_own <- own_knn(df[, 1:2], df[, 1:2], df$y, 5)

table(df$y, y_pred_own)
table(df$y, y_pred_orig)

all(y_pred_own == y_pred_orig)

# Benchmarking ------------------------------------------------------------
library(microbenchmark)

# 1st iteration
microbenchmark(
  knn(df[, 1:2], df[, 1:2], df$y, 5),
  own_knn(df[, 1:2], df[, 1:2], df$y, 5),
  times = 10
)
# Unit: milliseconds
# expr         min          lq        mean      median         uq        max neval
# knn(df[, 1:2], df[, 1:2], df$y, 5)    4.567162    4.709182    7.574931    7.595219   10.80597   10.89856    10
# own_knn(df[, 1:2], df[, 1:2], df$y, 5) 2615.752707 2719.867843 3285.322351 2896.186728 3496.49582 5074.03923    10

