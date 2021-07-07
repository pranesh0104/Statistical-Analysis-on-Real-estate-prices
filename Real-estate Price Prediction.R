# Installing required libraries
library(dplyr)
library(skimr)
library(corrplot)
library(gam)
library(tree)
library(randomForest)
library(gbm)
# Reading data
H_sales.df = read.csv("House_Sales_dataset.csv",stringsAsFactors = FALSE)
glimpse(H_sales.df)
# Data Cleaning
skim(H_sales.df)
# There are no missing values in the data as denoted by 'complete_rate' = 1 for all variables
# Defining 2 new columns 'Year' and 'month' which extracts the corresponding information from the date column
H_sales.df$year = as.integer(strtrim(H_sales.df$date,4))
H_sales.df$month = as.integer(substr(H_sales.df$date,5,6))
# Dropping the date column from dataframe as it is redundant now
H_sales.df = subset(H_sales.df,select = -(date))
# Defining a column 'h_age' which calculates the age of the house
H_sales.df$h_age = H_sales.df$year-H_sales.df$yr_built
View(H_sales.df)
# Using 'corrplot' to plot correlation between dependent and independent variables
cor_data = data.frame(H_sales.df[,2:23])
corr = cor(cor_data)
par(mfrow=c(1,1))
corrplot(corr,method="color",type = "upper",title = "Correlation Plot between 'price' and the predictors" )
# The correlation plot shows that price has a good positive correlation  with bedrooms, bathrooms, 
# Sqft_living,waterfront,floor, view , grade, sqft_above, sqft_basement, lat, sqft_living 15.
# These are the important features
imp_features = c("price","bedrooms","bathrooms","sqft_living","waterfront","floors","sqft_above","sqft_basement","view","grade","lat","sqft_living15")
# The cross-validation used here is the 'Validation Set' approach
# Dividing dataset into training and validation set(testing) by splitting 80/20 ratio
set.seed(10)
train = sample(nrow(H_sales.df), floor(nrow(H_sales.df)*.8))
training_HS = H_sales.df[train,imp_features]
validation_HS = H_sales.df[-train,imp_features]
dim(training_HS)
dim(validation_HS)
# Data Modeling 
attach(training_HS)

# Model 1 - Linear Regression Modeling

lm_price = lm(price~.,data = training_HS)
summary(lm_price)
# On further review,some of the variables that weren't statistically relevant were removed 
# such as sqft_above and sqft_basement
lm2_price = lm(price~bedrooms+bathrooms+sqft_living+waterfront+floors+view+grade+lat+sqft_living15,data = training_HS)
summary(lm2_price)
#CROSS-VALIDATION
lm_predict = predict(lm2_price,newdata = validation_HS)
test_lm = data.frame(actual = validation_HS$price,predicted = lm_predict)
lm_mse = mean(((test_lm$actual)-(test_lm$predicted))^2)
sprintf("Linear Model CV Test MSE ->%s",lm_mse)
sprintf("Linear Model CV Test R-MSE ->%s",sqrt(lm_mse))
# [1] "Linear Model CV Test MSE ->59490871572.4092"

# Model 2 - Generalized additive modelling

gam_price = gam(price~poly(sqft_basement,3)+ns(sqft_above,4)+bedrooms+bathrooms+sqft_basement+waterfront+floors+sqft_above+sqft_basement+view+grade+lat+sqft_living15,data=training_HS)
summary(gam_price)
coefficients(gam_price)
#CROSS-VALIDATION
gam_predict = predict.Gam(gam_price,newdata = validation_HS)
test_gam = data.frame(actual = validation_HS$price,predicted = gam_predict)
gam_mse = mean(((test_gam$actual)-(test_gam$predicted))^2)
sprintf("GAM Model CV Test MSE ->%s",gam_mse)
sprintf("GAM Model CV Test R-MSE ->%s",sqrt(gam_mse))
# [1] "GAM Model CV Test MSE ->50788888068.0358"

# Model 3 - Decision Tree

tree_price=tree(price~bedrooms+bathrooms+sqft_living+waterfront+floors+view+grade+lat+sqft_living15,data = training_HS)
summary(tree_price)
plot(tree_price)
text(tree_price,pretty=0)
#CROSS-VALIDATION
tree_predict = predict(tree_price,newdata = validation_HS)
test_tree = data.frame(actual = validation_HS$price,predicted = tree_predict)
tree_mse = mean(((test_tree$actual)-(test_tree$predicted))^2)
sprintf("Decision Tree Model CV Test MSE ->%s",tree_mse)
sprintf("Decision Tree Model CV Test R-MSE ->%s",sqrt(tree_mse))

# Model 4 - Random Forest Modeling

set.seed(10)
rf_price = randomForest(price~bedrooms+bathrooms+sqft_living+waterfront+floors+view+grade+lat+sqft_living15,data=training_HS,mtry=3,importance=TRUE)
importance(rf_price)
varImpPlot(rf_price)
#CROSS-VALIDATION
rf_predict = predict(rf_price,newdata = validation_HS)
test_rf = data.frame(actual = validation_HS$price,predicted = rf_predict)
rf_mse = mean(((test_rf$actual)-(test_rf$predicted))^2)
sprintf("Random Forest Model CV Test MSE ->%s",rf_mse)
sprintf("Random Forest Model CV Test R-MSE ->%s",sqrt(rf_mse))

# Model 5 - Generalized Boosted Regression Modeling
set.seed(10)
gbm_price=gbm(price~bedrooms+bathrooms+sqft_living+waterfront+floors+view+grade+lat+sqft_living15,data=training_HS,distribution="gaussian",n.trees=5000,interaction.depth=3)
summary(gbm_price)
#CROSS-VALIDATION
gbm_predict = predict(gbm_price,newdata = validation_HS)
test_gbm = data.frame(actual = validation_HS$price,predicted = gbm_predict)
gbm_mse = mean(((test_gbm$actual)-(test_gbm$predicted))^2)
sprintf("GBM Model CV Test MSE ->%s",gbm_mse)
sprintf("GBM Model CV Test R-MSE ->%s",sqrt(gbm_mse))

# Gbm_model Accuracy on validation set data
sprintf("Accuracy of GBM model ->%s",(1-mean(abs(test_gbm$actual-test_gbm$predicted)/test_gbm$actual))*100)


