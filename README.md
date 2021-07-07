# Statistical-Analysis-on-Real-estate-prices
Predictive Modelling on Real-estate Prices

For the price prediction dataset, first we start with a simple linear regression 
modeling, followed by generalized additive modeling(gam) and decision tree 
modeling to account for the non-linear data. To achieve an improvised model to 
predict the dataset, we next use random forest modelling which is essential a 
collection of decision trees which are then combined at the end(average) which can 
give us a better predictability over the earlier methods. We follow this up with the 
Generalized boosted regression modeling(gbm) which also uses decision trees but 
combines trees from the start. For the purpose of comparing all the models, we use 
the ‘Validation Set’ cross-validation approach. Here we split the data into 80% for 
training and 20% for validation. The model performance is evaluated using Validation 
(testing data) MSE and RMSE. The best model will be decided on which model 
achieves lowest MSE and RMSE score.
