# Set working directory to that of the current file
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  # Only when using RStudio
source(file.path("evalUtils.r", fsep = .Platform$file.sep))
source(file.path("MixRF.R", fsep = .Platform$file.sep))

# Environment preparation
library(readr)
library(mefa)
library(lme4)
library(lmerTest)
library(Hmisc)
library(car)
library(sjPlot)
library(optimx)
library(MuMIn)
library(boot)
library(plyr)
library(doParallel)
library(caret)
library(ranger)
library(ROCR)
library(data.table)

registerDoParallel(cores = detectCores())

# Data preparation
context <- read.csv(file.path("data","context.csv", fsep = .Platform$file.sep), row.names = 1)
context$TLOC <- factor(cut(context$LOC, quantile(context$LOC), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NFILES <- factor(cut(context$nfile, quantile(context$nfile), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NCOMMIT <- factor(cut(context$ncommit, quantile(context$ncommit), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NDEV <- factor(cut(context$ndev, quantile(context$ndev), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$nlanguage <- factor(cut(context$nlanguage, c(1,1.9,3), include.lowest = TRUE), labels = c("Least", "Most"))  # There are only 3 values (1,2,3) for this vector, hence we only distinguish 1 and >1
context$git <- NULL
context$LOC <- NULL
context$nfile <- NULL
context$ncommit <- NULL
context$ndev <- NULL

project_names <- row.names(context)
print(length(project_names))
metrics <- c("ns","nd","nf","la","ld","lt","norm_entropy")

projects <- list()
for (i in 1:length(project_names)) {
  projects[[i]] <- read_csv(file.path("data",paste(project_names[i], ".csv", sep = ''), 
                                      fsep = .Platform$file.sep),
                            col_types = cols(contains_bug = col_logical(), 
                                             fix = col_logical(), author_date = col_skip(), 
                                             author_date_unix_timestamp = col_skip(), 
                                             author_email = col_skip(), author_name = col_skip(), 
                                             classification = col_skip(), commit_hash = col_skip(), 
                                             commit_message = col_skip(), fileschanged = col_skip(), 
                                             fixes = col_skip(), glm_probability = col_skip(), 
                                             linked = col_skip(), repository_id = col_skip(),
                                             ndev = col_skip(), age = col_skip(), nuc = col_skip(),
                                             exp = col_skip(), rexp = col_skip(), sexp = col_skip()))
  
  projects[[i]]$loc <- projects[[i]]$la + projects[[i]]$ld
  
  projects[[i]]$norm_entropy <- 0
  tmp_norm_entropy <- projects[[i]]$entrophy / sapply(projects[[i]]$nf, log2) # Normalize entropy
  projects[[i]][projects[[i]]$nf >= 2, "norm_entropy"] <- tmp_norm_entropy[projects[[i]]$nf >= 2]
  
  projects[[i]]$project <- project_names[i]
  
  projects[[i]] <- cbind(projects[[i]], rep(context[project_names[i],], times = nrow(projects[[i]])))
} 

# Correlation and redundancy
#vcobj <- varclus(~., data = all_projects[,c("fix", metrics)], similarity = "spearman", trans = "abs")
#plot(vcobj)
#threshold <- 0.7
#abline(h = 1 - threshold, col = "red", lty = 2, lwd = 2)

#redun_obj <- redun(~ relative_churn + ns + norm_entropy + nf + fix + lt, data = all_projects, nk=5)
#paste(redun_obj$Out, collapse =", ")

scale_metrics <- c("ns","nf","lt","norm_entropy","relative_churn")

for (i in 1:length(project_names)) {
  projects[[i]]$relative_churn <- 0
  tmp_relative_churn <- (projects[[i]]$la + projects[[i]]$ld) / projects[[i]]$lt # (la+ld)/lt
  projects[[i]][projects[[i]]$lt >= 1, "relative_churn"] <- tmp_relative_churn[projects[[i]]$lt >= 1]
  projects[[i]][is.na(projects[[i]])] <- 0
  projects[[i]][,scale_metrics] <- lapply(projects[[i]][,scale_metrics], scale) # Data scaling
}

all_projects <- projects[[1]]
for (i in 2:length(project_names)) {
  all_projects <- rbind(all_projects, projects[[i]])
} 
all_projects$project <- as.factor(all_projects$project) # Merged dataset

# ===================== RQ1 ========================
local.jit <- c()
local.intercept <- c()
local.entropy <- c()

for (i in 1:length(project_names)) {
  # Model training
  trained_model <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, 
                       data = projects[[i]], family = "binomial")
  local.jit[[length(local.jit)+1]] <- trained_model
  
  # Model summary
  print(project_names[i])
  local.intercept[[length(local.intercept)+1]] <- trained_model$coefficients[1]
  local.entropy[[length(local.entropy)+1]] <- trained_model$coefficients[5]
  print(trained_model$coefficients)
  
  # Chisq
  print(Anova(trained_model, Type = 2))
  
  # Goodness-of-fit
  print(r.squaredGLMM(trained_model))
}

# ===================== RQ2 ========================
# Model training
global.jit <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = all_projects, family = "binomial")

# Median Absolute Error for entropy and intercepts
global.jit.entropy <- coef(global.jit)[5]
global.jit.intercept <- coef(global.jit)[1]
diff<- list()
#for(i in 1:length(local.entropy)) {
#  diff<-append(diff, abs(local.entropy[i]-global.jit.entropy))
#}
#median(abs(diff)) # MAE for Entropy
#median(abs(local.intercept - global.jit.intercept)) # MAE for intercepts

# Chisq
Anova(global.jit, Type = 2)

# Goodness-of-fit
print(r.squaredGLMM(global.jit))

# ===================== RQ3 ========================
# Performance of Global JIT Model (Logistic Regression)
print("RQ3")
get_lr_perf <- function(i) {
  training_set <- subset(all_projects, all_projects$project != project_names[i])
  testing_set <- subset(all_projects, all_projects$project == project_names[i])
  
  tmp_lr_model <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = training_set, family="binomial")
  tmp_lr_pred <- predict(tmp_lr_model, newdata = testing_set, type = "response")
  print(evalPredict(testing_set$contains_bug, tmp_lr_pred, testing_set$loc))
  return(evalPredict(testing_set$contains_bug, tmp_lr_pred, testing_set$loc))
}
print("get_lr_perf")
lr_perf <- llply(seq(1, length(project_names), 1), get_lr_perf)
lr_perf <- rbindlist(lr_perf, fill=TRUE)
print("global")