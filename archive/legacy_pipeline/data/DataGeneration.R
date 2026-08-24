setwd("/Users/tongxu/Downloads/projects/micodag-intervention/Data")

library(pcalg)
library(igraph)
library(gRbase)
library(MASS)


dag_generation <- function(m, sav=FALSE){  # generate random DAGs
    set.seed(m)
    g = pcalg::randomDAG(m, 0.2/(m/10)) # generated DAG
    moral_g = moralize(graph_from_graphnel(g)) # moral graph
    edge.list = apply(get.edgelist(igraph.from.graphNEL(g)), c(1, 2), as.numeric)
    moral.edge.list = apply(get.edgelist(moral_g), c(1, 2), as.numeric)

  if (sav){
    file.name = paste("./SyntheticData/DAG_", m, ".txt", sep="")
    moral.file.name = paste("./SyntheticData/Moral_", m, ".txt", sep="")
    write.table(edge.list, file.name, row.names=FALSE, col.names=FALSE)
    write.table(moral.edge.list, moral.file.name, row.names=FALSE, col.names=FALSE)
  }
  return(list("dag"=edge.list, "moral"=moral.edge.list))
}


weighted_adj <- function(dag){
  eweights <- c(-0.8, -0.6, 0.6, 0.8)
  gg <- graph_from_edgelist(dag)
  adjmat <- as.matrix(get.adjacency(gg))
  nv <- ncol(adjmat)
  
  ## add weights to the adjacency matrix and obtain influence matrix
  adjmat_wgtd <- adjmat * 
    matrix(sample(eweights, nv*nv, replace=T), nv, nv) 
  return(adjmat_wgtd)
}


data_generation_from_B <- function(adjmat_wgtd, interventions, n.samples, n.iterations=1, sav=FALSE){  # generate data from a dag
  sigvec <- c(1, 2, 4)
  nv <- ncol(adjmat_wgtd)
  Ip <- diag(1, nv, nv)
  infmat <- t(solve(Ip - adjmat_wgtd))
  ## covariance matrix for random noise with non-equal variance
  ## using formulas in Shojaie & Michailidis (2010)
  set.seed(nv)
  diagonals <- sample(sigvec, nv, replace=T)
  if(length(interventions)>0) diagonals[interventions] <- runif(length(interventions), 1, 4)
  covmat <- diag(diagonals)
  covmat <- infmat %*% covmat %*% t(infmat)
  
  datmat = c()
  ## generate data and write it into the same folder
  for(jj in 1:n.iterations){
    set.seed(jj)
    datmat[[jj]] <- mvrnorm(n=n.samples, mu=rep(0,nv), Sigma=covmat)
    if (sav){
      datfilename <- paste0(
        paste("data","m",m, "n", nsamples, "iter", jj, sep="_"), ".csv")
      write.table(datmat[[j]], datfilename, sep = ",", 
                  row.names=FALSE, col.names=FALSE)
    }
  }
  return(list("X"=datmat))
}


intervention_target <- function(m){  # generate intervention targets for a dag with m nodes
  k = sample(1:(m%/%2), 1)  # number of variables intervened. # I used sample(1:m%/%2, 1) which is a bug.
  targets <- sample(1:m, k)
  return(list("targets"=targets))
}


interventional_data_generate <- function(adjmat_wgtd, interventions, n.samples, n.iterations=1, sav=FALSE){
  adjmat_wgtd[, interventions] = 0
  X <- data_generation_from_B(adjmat_wgtd, interventions, n.samples, n.iterations, sav)
  return(X)
}


#-------------------------------
# Data folder structure, "/text" means a folder named text. Others are files.
# /Data
#   DAG, Moral.
#  - /SyntheticData
#     - graph1
#         - /environment1
#           data_iter1, data_iter2, ..., data_iter10
#         - /environment2
#           data_iter1, data_iter2, ..., data_iter10
#         - /environment3
#           ....
# 
#  - /RealWorldNetworks

# experiment setting
m_list = c(10, 20, 50, 100, 200)  # list of dimension of graphs
n.environment = 5  # number of environments
n.iter = 10  # number of set of samples for computing the average

graphs <- lapply(m_list, dag_generation, sav=F) # run this line once to save
# graphs <- lapply(m_list, dag_generation)
# graphs[[1]] is the first pair of dag and moral lists.

weighted_adjs <- list()  # assign weights to the adj matrices
for (i in 1:length(m_list)){
  weighted_adjs[[i]] <- weighted_adj(graphs[[i]]$dag)
}

# data1 <- data_generation_from_B(weighted_adjs[[1]], 100, 1) # test command

random_interventions = list()  # generate intervention targets for each enviroment
for (i in 1:n.environment) {
  random_interventions[[i]] = lapply(m_list, intervention_target)
}
for (i in 1:length(m_list)) {
  writeLines(sapply(random_interventions, function(x) paste(x[[i]]$targets, collapse = ",")), paste0("./SyntheticData/","graph",i,"/intervention_targets.txt"))
}

# random_interventions[[1]] stores intervention targets of the first environment


##############data generating####################

# interventional data
for (m_idx in 1:length(m_list)) {
  for (env in 1:n.environment) {
    data_int <- interventional_data_generate(weighted_adjs[[m_idx]], random_interventions[[env]][[m_idx]]$targets, 2*m_list[m_idx], n.iter)
    path_i =  paste0("./SyntheticData", "/graph", m_idx, "/environment", env)
    if (!dir.exists(path_i)) {
      dir.create(path_i, recursive = TRUE)
    }
    for (iter in 1:n.iter) {
      write.table(data_int$X[[iter]], paste0(path_i, "/data_", iter, ".csv"), row.names=FALSE, col.names=FALSE, sep=",")
    }
  }
}
# observational data
for (m_idx in 1:length(m_list)) {
  data_int <- interventional_data_generate(weighted_adjs[[m_idx]], c(), 500, n.iter)
  path_i =  paste0("./SyntheticData", "/graph", m_idx, "/observational")
  if (!dir.exists(path_i)) {
    dir.create(path_i, recursive = TRUE)
  }
  for (iter in 1:n.iter) {
    write.table(data_int$X[[iter]], paste0(path_i, "/data_", iter, ".csv"), row.names=FALSE, col.names=FALSE, sep=",")
  }
}







