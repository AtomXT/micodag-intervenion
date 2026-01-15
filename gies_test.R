# Test gies with true intervention targets.

setwd("/Users/tongxu/Downloads/projects/micodag-intervention")

library(igraph)
library(pcalg)
library(glue)


m = 5
# iter = 1
n.iter = 10
n.environment = 5
# n_e = 500
m_list <- c(10, 20, 50, 100, 200)

d_essential <- c()
times <- c()
for (iter in 1:n.iter) {
  targets <- c()
  dat <- read.csv(glue('./Data/SyntheticData/graph{m}/observational/data_{iter}.csv'), header = F)
  targets <- c(targets, rep(1, dim(dat)[1]))
  for (env in 1:n.environment) {
    dat_temp <- read.csv(glue('./Data/SyntheticData/graph{m}/environment{env}/data_{iter}.csv'), header = F)
    dat <- rbind.data.frame(dat, dat_temp)
    targets <- c(targets, rep(env+1, dim(dat_temp)[1]))
  }
  
  intervention.targets <- read.table(glue("./Data/SyntheticData/graph{m}/intervention_targets.txt"), sep=',', fill=T, header = F)
  names(intervention.targets) = NULL
  intervention.targets_v <- lapply(split(intervention.targets,1:nrow(intervention.targets)),function(v){v[!is.na(v)]})
  names(intervention.targets_v) = NULL
  intervention.targets_v <- c(list(integer(0)), intervention.targets_v)
  start <- Sys.time()
  score <- new("GaussL0penIntScore", dat, intervention.targets_v, targets)
  gies.fit <- gies(score)
  time_i <- Sys.time() - start
  times <- c(times, time_i)
  edges <- gies.fit$essgraph$.in.edges
  gies.adj.mat <- matrix(0, m_list[m], m_list[m])
  for (node in 1:m_list[m]) {
    gies.adj.mat[edges[[node]], node] = 1
  }
  
  
  true.dag.edges <- read.table(glue("./Data/SyntheticData/DAG_{m_list[m]}.txt"))
  true.dag <- graph_from_edgelist(as.matrix(true.dag.edges))
  intervention.targets <- c(t(intervention.targets))
  intervention.targets <- unique(intervention.targets[!is.na(intervention.targets)])
  true.esstional.graph <- dag2essgraph(as_graphnel(true.dag), intervention.targets)
  true.esstional.graph.adj <- as.matrix(get.adjacency(graph_from_graphnel(true.esstional.graph)))
  
  d_ess_i <- sum(abs(gies.adj.mat - true.esstional.graph.adj))
  d_essential <- c(d_essential, d_ess_i)
}

d_essential
mean(d_essential)
mean(times)



# with n_e = 10000
# For m=1, d_essential = c(0 1 0 0 3 1 0 0 1 0), mean is 0.6;
# FOr m=2, d_essential = c(1 1 2 1 2 2 2 1 1 1), mean is 1.4.