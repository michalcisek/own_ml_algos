rm(list = ls())

data(iris)
colnames(iris)
table(iris$Species)

y <- iris$Species
x <- iris[, -which(colnames(iris) == "Species")]

#impurity measures: Gini or Entropy

#define number of thresholds verified for each variable
n_thresholds <- 100

#select feature and threshold that minimise cost function: 
#(n_left/n_node)*impurity_left + (n_right/n_node)*impurity_right


calc_gini <- function(y){
  gini <- 1-sum((table(y)/length(y))^2)
  return(gini)
}

select_best_split <- function(x, y, n_thresholds){
  cost <- Inf
  best_feat <- NA
  best_threshold <- NA
  
  for(feat in colnames(x)){
    thresholds <- seq(min(x[, feat]), max(x[, feat]), length.out = n_thresholds + 2)
    thresholds <- thresholds[-c(1, length(thresholds))]
    
    for(thr in thresholds){
      ind_lnode <- which(x[, feat] <= thr)
      ind_rnode <- which(x[, feat] > thr)
      
      if(length(ind_lnode) == 0 | length(ind_rnode) == 0){
        next()
      }
      
      gini_l <- calc_gini(y[ind_lnode])
      gini_r <- calc_gini(y[ind_rnode])
      
      cost_fun <- length(ind_lnode)/nrow(x)*gini_l +length(ind_rnode)/nrow(x)*gini_r
      
      if(cost_fun < cost){
        cost <- cost_fun
        best_feat <- feat
        best_threshold <- thr
      }
    }
  }
  return(list(best_feature = best_feat, best_threshold = best_threshold))
}

add_node_info <- function(x, y, parent_node, child_type){
  node_info <- list()
  node_info$height <- height
  node_info$samples <- nrow(x)
  node_info$gini <- calc_gini(y)
  node_info$id <- max_id + 1
  node_info$is_leaf <- ifelse(node_info$gini == 0, TRUE, FALSE)
  node_info$parent_id <- ifelse(missing(parent_node), NA, parent_node$id)
  node_info$node_type <- ifelse(missing(child_type), NA, child_type)
  if(node_info$is_leaf == FALSE){
    split <- select_best_split(x, y, 100)
    node_info <- c(node_info, split)
  }
  max_id <<- max_id + 1   
  return(node_info)
}

find_all_parents <- function(node_id, nodes){
  parents <- list()
  
  curr_node <- which(sapply(nodes, function(x) x$id == node_id))
  curr_node <- nodes[[curr_node]]
  parent_id <- curr_node$parent_id
  
  while(!is.na(parent_id)){
    curr_node <- which(sapply(nodes, function(x) x$id == parent_id))
    curr_node <- nodes[[curr_node]]
    
    parent_id <- curr_node$parent_id
    parents[[length(parents) + 1]] <- curr_node  
  }
  return(parents)
}

max_id <- 0
nodes <- list()
height <- 1

nodes[[length(nodes) + 1]] <- add_node_info(x, y)

height <- 2
current_nodes <- nodes

while(any(sapply(current_nodes, function(x) x$is_leaf) == FALSE)){
  for(node in current_nodes){
    if(node$is_leaf){
      next()
    } else {
      parents <- find_all_parents(node$id, nodes)
  
      if(length(parents) > 0){
        sign <- c()
        var <- c()
        thresh <- c()
        
        prev_parent <- node    
        for(i in parents){
          sign <- c(sign, ifelse(prev_parent$node_type == "left", "<=", ">"))
          parent <- nodes[[which(sapply(nodes, function(x) x$id == prev_parent$parent_id))]]
          var <- c(var, i$best_feature)
          thresh <- c(thresh, i$best_threshold)
          
          prev_parent <- i
        }
        
        subset_conds <- paste0("which(", paste0("x$", var, " ", sign, " ", thresh, collapse = " & "), ")")
        subset <- eval(parse(text = subset_conds))
        subset_x <- x[subset, ]
        subset_y <- y[subset]
      } else {
        subset_x <- x
        subset_y <- y
      }
      
      left_child <- which(subset_x[node$best_feature] <= node$best_threshold)
      left_x <- subset_x[left_child, ]
      left_y <- subset_y[left_child]

      nodes[[length(nodes) + 1]] <- add_node_info(left_x, left_y, node, "left")
      
      right_child <- which(subset_x[node$best_feature] > node$best_threshold)
      right_x <- subset_x[right_child, ]
      right_y <- subset_y[right_child]

      nodes[[length(nodes) + 1]] <- add_node_info(right_x, right_y, node, "right")
    }
  }
  current_nodes <- nodes[which(sapply(nodes, function(x) x$height) == height)]
  height <- height + 1
}


