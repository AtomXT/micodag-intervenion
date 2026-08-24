# Transfer dags into essdags to evaluate

# setwd("E:/Northwestern/Research/micodag-intervention")
setwd("/Users/tongxu/Downloads/projects/micodag-intervention")
library(igraph)
library(pcalg)
library(glue)

m_list <- c(10, 20, 50, 100, 200)
m = 3
n.iter = 10
lmbda = 0.02

true.dag.edges <- read.table(glue("./Data/SyntheticData/DAG_{m_list[m]}.txt"))
true.dag <- graph_from_edgelist(as.matrix(true.dag.edges))
intervention.targets <- read.table(glue("./Data/SyntheticData/graph{m}/intervention_targets.txt"), sep=',', fill=T, header = F)

names(intervention.targets) = NULL
intervention.targets_v <- lapply(split(intervention.targets,1:nrow(intervention.targets)),function(v){v[!is.na(v)]})
names(intervention.targets_v) = NULL
intervention.targets_v <- c(list(integer(0)), intervention.targets_v)

# intervention.targets <- c(t(intervention.targets))
# intervention.targets <- unique(intervention.targets[!is.na(intervention.targets)])
true.esstional.graph <- dag2essgraph(as_graphnel(true.dag), intervention.targets_v)
true.esstional.graph.adj <- as.matrix(get.adjacency(igraph.from.graphNEL(true.esstional.graph)))

d_essential_gnies_rank <- c()
d_essential_gnies_greedy <- c()
d_essential_micodag_list <- c()
get_indices <- function(row) {
  which(row == 1)
}

for (iter in 1:n.iter) {
  approach1 = 'rank'
  gnies.esstional.graph1 <- read.table(glue("./Results/estimations/gnies_{approach1}_essential_graph{m}_iter{iter}.txt"), sep=',')
  d_essential_gnies1 <- sum(abs(gnies.esstional.graph1 - true.esstional.graph.adj))
  gnies.targets1 <- tryCatch(read.table(glue("./Results/estimations/micodagcd_intervention_targets_graph{m}_lambda{lmbda}_iter{iter}.txt"), sep=','), error=function(e) NULL)
  gnies.targets1_v = apply(gnies.targets1, 1, get_indices)
  temp1 <- dag2essgraph(as_graphnel(graph_from_adjacency_matrix(as.matrix(gnies.esstional.graph1))), gnies.targets1_v)
  temp1 <- as.matrix(get.adjacency(igraph.from.graphNEL(temp1)))
  d_essential_gnies1 <- sum(abs(temp1 - true.esstional.graph.adj))
  
  
  # approach2 = 'greedy'
  # gnies.esstional.graph2 <- read.table(glue("./Results/estimations/gnies_{approach2}_essential_graph{m}_iter{iter}.txt"), sep=',')
  # d_essential_gnies2 <- sum(abs(gnies.esstional.graph2 - true.esstional.graph.adj))
  
  micodag.dag <- read.table(glue("./Results/estimations/micodagcd_dag_graph{m}_lambda{lmbda}_iter{iter}.txt"), sep=',')
  micodag.targets <- tryCatch(read.table(glue("./Results/estimations/micodagcd_intervention_targets_graph{m}_lambda{lmbda}_iter{iter}.txt"), sep=','), error=function(e) NULL)
  
  # micodag.dag <- read.table(glue("./Results/estimations/micodagcd_dag_graph{m}_lambda{lmbda}_iter{iter}.txt"), sep=',')
  # micodag.targets <- tryCatch(read.table(glue("./Results/estimations/micodagcd_intervention_targets_graph{m}_lambda{lmbda}_iter{iter}.txt"), sep=','), error=function(e) NULL)
  
  
  micodag.targets = as.matrix(micodag.targets)
  
  micodag.targets_v = apply(micodag.targets, 1, get_indices)
  micodag.dag <- graph_from_adjacency_matrix(as.matrix(micodag.dag))
  micodag.esstional.graph <- dag2essgraph(as_graphnel(micodag.dag), micodag.targets_v)
  micodag.esstional.graph.adj <- as.matrix(get.adjacency(igraph.from.graphNEL(micodag.esstional.graph)))
  d_essential_micodag <- sum(abs(micodag.esstional.graph.adj - true.esstional.graph.adj))

  d_essential_gnies_rank[iter] <- d_essential_gnies1
  # d_essential_gnies_greedy[iter] <- d_essential_gnies2
  d_essential_micodag_list[iter] <- d_essential_micodag
}

d_essential_gnies_rank
# d_essential_gnies_greedy
d_essential_micodag_list
mean(d_essential_gnies_rank)
# mean(d_essential_gnies_greedy)
mean(d_essential_micodag_list)


