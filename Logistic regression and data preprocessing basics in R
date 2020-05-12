rm(list = ls())

#1. Import the data ====
setwd("~/Downloads/20190324_Batch59_CSE7302c_LogisticRegression_Lab3")
data1 = read.table("bank.txt", header=T, sep=";")

#2. Understanding & Pre processing the data ====

#2a. Dimensions, structure & summary of data
dim(data1) #dimenssions of data
str(data1) #structure of data
summary(data1) #Summary stats of each variable

#2b. Dealing with missing values

#check for number of 'NA' in data frame
sum(is.na(data1))

#3. Split the data into train validation and test ====

library(caret)
set.seed(1234)

table(data1$y) #Distribution of levels in target data
table(data1$y) / nrow(data1)

# Finding the train rows & subset from data
train_rows = createDataPartition(data1$y, p = 0.7, list = F)
train_data = data1[train_rows, ]

# Find the validation & test rows & subset from the data
Set2 = data1[-train_rows, ]
valrows = createDataPartition(Set2$y, p = 0.5, list = F)
validationdata = Set2[valrows,]
test_data = Set2[-valrows, ]

# Check the dimensions of train, validation & test
dim(train_data) #dimensions of train data
dim(validationdata) #dimensions of validation data
dim(test_data) #dimensions of test data

dim(data1) #dimensions of original data
dim(train_data)[1] + dim(validationdata)[1] + dim(test_data)[1] #sum of rows in train, validation & test

#check for distribution of levels in split data
prop.table(table(data1$y))
prop.table(table(train_data$y))
prop.table(table(validationdata$y))
prop.table(table(test_data$y))

#4. Build logistic regression model & find the predictions ====
log_reg = glm(y ~ ., data = train_data, family = "binomial")
summary(log_reg)

#Finding beta values
train_beta = predict(log_reg,train_data)
train_beta[1]
train_beta
1 / (1+exp(-train_beta[1]))

#Finding probabilities
train_probabilities = predict(log_reg,train_data,type = "response")
train_probabilities[1]

#Finding predictions
levels(data1$y) #understand the order of levels
train_prediction = ifelse((train_probabilities<0.5), "no", "yes")

#Validation predictions
val_probabilities = predict(log_reg,validationdata,type = "response")
val_prediction = ifelse((val_probabilities<0.5), "no", "yes")

#5. Building confusion matrix ====
con_mat = table(train_prediction,train_actual = train_data$y)
val_con_mat = table(val_prediction,val_actual = validationdata$y)

# Finding required metrics on train data
train_accuracy = sum(diag(con_mat)) / sum(con_mat)
train_recall = con_mat[2,2] / sum(con_mat[,2])
train_precision = con_mat[2,2] / sum(con_mat[2,])

# Finding metrics using inbuilt function
#library(caret) #req function available in caret package
confusionMatrix(con_mat)
confusionMatrix(con_mat,positive = "yes")

# Finding validation metrics
confusionMatrix(val_con_mat,positive = "yes")

#6. Identifying collinear variables using 'VIF' & important variables using 'Step-AIC' ====

# 'VIF' for collinear variables
library(car)
vif(log_reg)

# 'Step-AIC' for important variables

library(MASS)
m = stepAIC(log_reg) #using Step-AIC on logistic reg model
summary(m) #check the summary of logistic + Step-AIC model
m$call #get the syntax for best variable model

# Building logistic regression using variables from Step-AIC
log_reg_step_aic = glm(formula = y ~ job + marital + education + loan + contact + 
                         day + month + duration + campaign + poutcome, family = "binomial", 
                       data = train_data)

#7. Finding train and validation pobabilities for the log+step-AIC model ====

# Finding probabilities
step_train_probailities = predict(log_reg_step_aic,train_data,type = "response")
step_val_probailities = predict(log_reg_step_aic,validationdata,type = "response")

# Finding predictions
step_train_predictions = ifelse((step_train_probailities<0.5), "no", "yes")
step_val_predictions = ifelse((step_val_probailities<0.5), "no", "yes")

# Finding confusion matrices
step_con_mat = table(step_train_predictions,train_actual = train_data$y)
step_val_con_mat = table(step_val_predictions,val_actual = validationdata$y)

# Finding metrics
confusionMatrix(step_con_mat,positive = "yes")
confusionMatrix(step_val_con_mat,positive = "yes")

#8. Using ROCR curves to find the best cut-off value ====

library(ROCR)

#Creating prediction object for ROCR
rocpreds = prediction(step_train_probailities, train_data$y)


# Extract performance measures (True Positive Rate and False Positive Rate) using the "performance()" function from the ROCR package
# The performance() function from the ROCR package helps us extract metrics such as True positive rate, False positive rate etc. from the prediction object, we created above.
# Two measures (y-axis = tpr, x-axis = fpr) are extracted
perf = performance(rocpreds, measure="tpr", x.measure="fpr")
slotNames(perf)

perf
# Plot the ROC curve using the extracted performance measures (TPR and FPR)
plot(perf, col = rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))

# Extract the AUC score of the ROC curve and store it in a variable named "auc"
# Use the performance() function on the prediction object created above using the ROCR package, to extract the AUC score
perf_auc = performance(rocpreds,  measure="auc")

# Access the auc score from the performance object
auc = perf_auc@y.values[[1]]
auc

# For different threshold values identifying the tpr and fpr
cutoffs = data.frame(cut= perf@alpha.values[[1]], fpr= perf@x.values[[1]], 
                      tpr=perf@y.values[[1]])

# Sorting the data frame in the decreasing order based on tpr
cutoffs = cutoffs[order(cutoffs$tpr, cutoffs$fpr, decreasing=TRUE),]
head(cutoffs)
class(perf)

# Plotting the true positive rate and false negative rate based on the cutoff       
# increasing from 0.05-0.1
plot(perf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

## Choose a Cutoff Value
# Based on the trade off between TPR and FPR depending on the business domain, a call on the cutoff has to be made.
# A cutoff of 0.1 can be chosen which is in conservative area

#9. Using best cutoff value, find new predictions & new metrics ====
# Finding predictions
new_step_train_predictions = ifelse((step_train_probailities<0.1), "no", "yes")
new_step_val_predictions = ifelse((step_val_probailities<0.1), "no", "yes")

# Finding confusion matrices
new_step_con_mat = table(new_step_train_predictions,train_actual = train_data$y)
new_step_val_con_mat = table(new_step_val_predictions,val_actual = validationdata$y)

# Finding metrics
confusionMatrix(new_step_con_mat,positive = "yes")
confusionMatrix(new_step_val_con_mat,positive = "yes")

#10. Naive Bayes sample problem ====

rm(list = ls())
cancer = c("Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No")

#P("Yes") = count("Yes") / total count
#P("No") = count("No") / total count

P_Yes = length(cancer[cancer == "Yes"]) / length(cancer)
P_No = length(cancer[cancer == "No"]) / length(cancer)
P_Yes
P_No
#Based on this data only, if you had to predict if a new person has cancer / not, what is the answer?
if(P_Yes > P_No){print("cancer = Yes")}else{print("cancer = No")}

# Adding a variable 'smoking' to the data: ====

smoking = c("smoker","smoker","smoker","non_smoker","non_smoker","smoker","smoker","non_smoker","non_smoker","non_smoker","non_smoker")
df = data.frame(cancer,smoking)
dim(df)
# Manual calculation =====
#Posible combinations:

#P(cancer = "Yes" | smoking = "smoker")
#P(cancer = "No" | smoking = "smoker")

#P(cancer = "Yes" | smoking = "non_smoker")
#P(cancer = "No" | smoking = "non_smoker")

#If the smoking information is given for the new person, does the person have cancer?
#If smoking = "somker"?, If smoking = "non_smoker"?

occurances = table(df)
occurances
P_SN_CN = occurances[1,1] / sum(occurances[1,])
P_SY_CN = occurances[1,2] / sum(occurances[1,])

P_SN_CY = occurances[2,1] / sum(occurances[2,])
P_SY_CY = occurances[2,2] / sum(occurances[2,])

P_Yes = sum(occurances[2,]) / sum(occurances)
P_No = sum(occurances[1,]) / sum(occurances)

#For smoker
P_Yes_smoker = P_SY_CY * P_Yes
P_No_smoker = P_SY_CN * P_No

if(P_Yes_smoker > P_No_smoker){print("cancer = Yes")}else{print("cancer = No")}

#For non_smoker
P_Yes_non_smoker = P_SN_CY * P_Yes
P_No_non_smoker = P_SN_CN * P_No

if(P_Yes_non_smoker > P_No_non_smoker){print("cancer = Yes")}else{print("cancer = No")}

prop.table(occurances)

# Using Naive Bayes algorithm =====

df = data.frame(apply(df,2,as.factor))

library(e1071)
cancer_model = naiveBayes(cancer~.,df)

test = data.frame(smoking = c("smoker","non_smoker"),predictions = c("",""))
test$predictions = predict(cancer_model,test)
test
cancer_model$tables
cancer_model$apriori


#Verification
#For smoker
P_Yes_smoker = cancer_model$tables$smoking[,"smoker"][2] * (cancer_model$apriori["Yes"] / sum(cancer_model$apriori)) 
P_No_smoker = cancer_model$tables$smoking[,"smoker"][1] * (cancer_model$apriori["No"] / sum(cancer_model$apriori))

#For non_smoker
P_Yes_non_smoker = cancer_model$tables$smoking[,"non_smoker"][2] * (cancer_model$apriori["Yes"] / sum(cancer_model$apriori)) 
P_No_non_smoker = cancer_model$tables$smoking[,"non_smoker"][1] * (cancer_model$apriori["No"] / sum(cancer_model$apriori))


# Adding another variable 'test_result' to the data ====
test_result = c("positive","positive","negative","positive","negative","positive","negative","positive","negative","negative","negative")
df = cbind(df,test_result)

#If smoking and test_result information is given for the new person, does the person have cancer?

#P(Y|X)= P(x1|Y) * P(x2|Y) * ... * P(Y)
#X = set of independent variables (x1,x2,x3,..)
#Y = dependent variable (Target)

str(df)

#using naive bayes algorithm
cancer_model = naiveBayes(cancer~.,df)

test = data.frame(smoking = c("smoker","smoker","non_smoker","non_smoker"),
                  test_result = c("positive","negative","positive","negative"),
                  predictions = rep("",4))

test$predictions = predict(cancer_model,test)

cancer_model$tables
cancer_model$apriori

#Verification
#smoking = smoker, test_result = positive

P_Yes_smoker_positive = cancer_model$tables$smoking[,"smoker"][2]*
  cancer_model$tables$test_result[,"positive"][2]* 
  (cancer_model$apriori["Yes"] / sum(cancer_model$apriori))
P_No_smoker_positive = cancer_model$tables$smoking[,"smoker"][1]*
  cancer_model$tables$test_result[,"positive"][1]*
  (cancer_model$apriori["No"] / sum(cancer_model$apriori))

P_Yes_smoker_positive / (P_Yes_smoker_positive+P_No_smoker_positive)
P_No_smoker_positive / (P_Yes_smoker_positive+P_No_smoker_positive)

#smoking = non_smoker, test_result = positive
P_Yes_non_smoker_positive = cancer_model$tables$smoking[,"non_smoker"][2]*
  cancer_model$tables$test_result[,"positive"][2]*
  (cancer_model$apriori["Yes"] / sum(cancer_model$apriori))
P_No_non_smoker_positive = cancer_model$tables$smoking[,"non_smoker"][1]*
  cancer_model$tables$test_result[,"positive"][1]*
  (cancer_model$apriori["No"] / sum(cancer_model$apriori))

P_Yes_non_smoker_positive / (P_Yes_non_smoker_positive + P_No_non_smoker_positive)
P_No_non_smoker_positive / (P_Yes_non_smoker_positive + P_No_non_smoker_positive)


#11. Naive Bayes on flights data ====
#Code to clear your global environment
rm(list=ls(all=TRUE))

# Read data ====
flight<-read.csv(file = "FlightDelays.csv",header = T,sep = ",")

#install.packages("e1071")
library(e1071)
#Viewing the head of the data 
head(flight)
tail(flight)

#Checking  the structure of the data
str(flight)

#check the summary of the data
summary(flight)

#Plotting to know levels distribution in each of the factor variables
plot(flight$CARRIER,main="Frequencies of Carriers",xlab="levels in Carrier",ylab="Frequency",col="Red")
plot(flight$DEST,main="Frequencies of Flight to Different Destination",xlab="Different Destinations",ylab="Frequency",col="Green")
plot(flight$ORIGIN,main="Frequencies of  Flight to Different Orgin",xlab="Different Origins",ylab="Frequency",col="Blue")


#Preprocessing ideas
flight$Weather<-as.factor(flight$Weather)
flight$Flight.Status<-as.factor(flight$Flight.Status)

# Binning the fight$DEP_TIME ====
#It is a continuous data and does not fit under traditional binning techniques We need to go for manual binning

nrow(flight)
flight$levels<-ifelse(flight$DEP_TIME>=600 & flight$DEP_TIME<1200,"level1",
                      ifelse(flight$DEP_TIME>=1200 & flight$DEP_TIME<1800,"level2",
                             ifelse(flight$DEP_TIME>=1800 & flight$DEP_TIME<2100,"level3","level4")))

#Converting the DEP_TIME into Factor
flight$DEP_TIME <- NULL
flight$levels <- as.factor(flight$levels)

table(flight$Flight.Status)

#Checking  back the structure
str(flight)

# Splitting the data set ====
#think of the code 

set.seed(123)
#install.packages("caret")
library(caret)
train_rows <- createDataPartition(flight$Flight.Status, p = 0.7, list = F)
train<- flight[train_rows, ]
Set2<-flight[-train_rows, ]
valrows<-createDataPartition(Set2$Flight.Status, p = 0.7, list = F)
validationdata<-Set2[valrows,]
testdata <- Set2[-valrows, ]


#checking the dimensions the partitions 
dim(train)
dim(validationdata)
dim(testdata)

#Study target distribution in train and test 

#Sanity check Check the proportions of classes in all datasets with the given dataset
prop.table(table(train$Flight.Status))
prop.table(table(validationdata$Flight.Status))
prop.table(table(testdata$Flight.Status))
prop.table(table(flight$Flight.Status))

# Building  a model with naive Bayes ====

model_nb<-naiveBayes(train$Flight.Status~.,train)

# Response of the model ====
model_nb
# It lists all likelihoods for attributes -frequency tables
model_nb$tables
# Check for one variable DEST
model_nb$tables$DEST
# sanity check manual calcuation of frequency table

# Prior Probabalities
# Conditional probabalities
a<-table(train$DEST,train$Flight.Status)
a[,1]/table(train$Flight.Status)[1]
a[,2]/table(train$Flight.Status)[2]

# Predict the Flight Status on the train data ====
train_preds<-predict(model_nb,train)
cnf<-table(train$Flight.Status,train_preds)
acc<-sum(diag(cnf))/nrow(train)
print(acc)

# recall
cnf[2,2]/sum(cnf[2,])


#prediction on validation data
preds<-predict(model_nb,validationdata)
cnf<-table(validationdata$Flight.Status,preds)
acc<-sum(diag(cnf))/nrow(validationdata)
print(acc)
# recall
cnf[2,2]/sum(cnf[2,])


#Alternate way of calculating confusion matrix
#Confusion Matrix
library(caret)
confusionMatrix(validationdata$Flight.Status,preds)


# How it is computing the probability values: Look at the following ====

#What the probability of gettin delayes if Carrier=co,Dest=EWR,Origin=DCA,weather=good,Day=3,level=3
## CARRIER=US,DEST=LGA, ORIGIN=DCA,Weather=0,levels=level2,Actual:Flight.Status=0 
# predictesd:train_preds[4]
# Calcualate conditional probabilities for each:
#p(class/carrier,dest,orgin,weather,dayweek,levels)=p(carrier/class)*p(dest/class)*p(origin/class)*p(weather/class)*p(dayweek/class)*p(levels/class)*p(class)

# Now we can compute probabilities for each class
#p(class=0)
class_prob<-prop.table(table(train$Flight.Status))

#p(carrier=US/class=0)
p_0<-model_nb$tables$CARRIER[,"US"][1]*model_nb$tables$DEST[,"LGA"][1]*model_nb$tables$ORIGIN[,"DCA"][1]*model_nb$tables$Weather[,"0"][1]*model_nb$tables$levels[,"level2"][1]*class_prob[1]
p_1<-model_nb$tables$CARRIER[,"US"][2]*model_nb$tables$DEST[,"LGA"][2]*model_nb$tables$ORIGIN[,"DCA"][2]*model_nb$tables$Weather[,"0"][2]*model_nb$tables$levels[,"level2"][2]*class_prob[2]

# These are proportions No we can get the probabilitie as folllowing
p_0/(p_0+p_1)*100
p_1/(p_0+p_1)*100

#This observation got 94.06 for being zero not delayed
#This observation got 5.93 for being one  not delayed
