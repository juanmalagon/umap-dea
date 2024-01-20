rm(list=ls())
gc()

suppressWarnings(library("deaR"))
library("tidyverse")

# Set paths
parent_dir <- '/Users/juanmalagon/repos/high-dim-dea/legacy'
mc_simulation_folder <- '/mc_simulation/'
mc_simulation_path <- paste0(parent_dir, mc_simulation_folder)
dea_results_folder <- '/dea_results/'
dea_results_path <- paste0(parent_dir, dea_results_folder)

# Read files with the inputs variables, either the original dataset or a
# dimensionality reduced dataset
# filename <- 'inputs.csv'
filename <- 'inputs_umap_02_dims.csv'
# filename <- 'inputs_umap_05_dims.csv'
# filename <- 'inputs_umap_20_dims.csv'
# filename <- 'inputs_umap_30_dims.csv'
# filename <- 'inputs_umap_40_dims.csv'
inputs <- read.csv(paste0(mc_simulation_path,filename))

# Read file with output variables
filename <- 'outputs.csv'
outputs <- read.csv(paste0(mc_simulation_path,filename))

dataset <- inputs
dataset$Y <- outputs
names(dataset)[length(names(dataset))]<-"Y" 

# Calculate DEA
orientation <- "io"
rts <- "crs"
ni <- ncol(dataset)-2

print("[INFO] Calculating DEA")
datadea <- read_data(dataset
                     , ni=ni
                     , no=1
)
print(paste0("[INFO] Elapsed time DEA ", ni, " inputs ", rts, " ", orientation ))
start_time <- Sys.time()
result <- model_basic(datadea
                      , orientation=orientation
                      , rts=rts)
end_time <- Sys.time()
print(end_time - start_time)

# List the resulting dataframes
output_list <- list()
output_list$efficiencies_df <- as.data.frame(efficiencies(result))
output_list$slacks_df <- as.data.frame(slacks(result))
output_list$targets_df <- as.data.frame(targets(result))
output_list$lambdas_df <- as.data.frame(lambdas(result))
output_list$references_df <- references(result)

export_files <- function(results_list, ni, rts, orientation, path){
  write.csv(results_list$efficiencies_df
            , paste0(dea_results_path, ni, '_inputs_', rts, '_', orientation, '_efficiencies.csv'))
  write.csv(results_list$slacks_df
            , paste0(dea_results_path, ni, '_inputs_', rts, '_', orientation, '_slacks.csv'))
  write.csv(results_list$targets_df
            , paste0(dea_results_path, ni, '_inputs_', rts, '_', orientation, '_targets.csv'))
  write.csv(results_list$lambdas_df
            , paste0(dea_results_path, ni, '_inputs_', rts, '_', orientation, '_lambdas.csv'))
  sink(paste0(dea_results_path, ni, '_inputs_', rts, '_', orientation, '_references.txt'))
  print(results_list$references_df)
  sink()
  print(paste0("[INFO] Exported results for ", rts, " ", orientation ))
}

export_files(output_list, ni, rts, orientation, path)
