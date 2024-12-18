library(vroom)
library(tidymodels)
library(discrim)
library(themis)
library(embed)
library(rstanarm)

train <- vroom("train.csv")
test <- vroom("test.csv")
glimpse(train)

my_recipe <- recipe(type ~., data = train) %>%
  step_mutate(color = as.factor(color)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = "type") %>% 
  step_smote(all_outcomes(), neighbors = 20)
  

nbmod <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nbwf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nbmod)

tunegrid <- grid_regular(Laplace(),
                         smoothness(),
                         levels = 10)

folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- nbwf %>%
  tune_grid(resamples = folds,
            grid = tunegrid,
            metrics = metric_set(accuracy, roc_auc))

bestTune <- CV_results %>%
  select_best()


tunewf <- nbwf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)


gggpred <- predict(tunewf, new_data = test)

kaggle_submission <- test %>%
  select(id) %>%
  bind_cols(as_tibble(gggpred) %>% rename(type = .pred_class))

# use ID as a variable and make some vars factors, or add smote
vroom_write(x = kaggle_submission, file = "./gggpred.csv", delim = ",")
