

---------------------------------------------------------------------------------------------------------------------------
  
# 100 DATA PREPARATION [PART A ]
  
  # start the library

library(tidyverse)

library(readxl)


# READ THE FILE 

china_df<- read.csv("data_china_all.csv")

# delete irrelevant variables 

china_df<- china_df[ -c(1:4, 74:76) ]

str(china_df)

china_df<-drop_na(china_df)

china_df<- replace(china_df,is.na(china_df),0)


# copy of main china file 

china_main<-china_df

china_main<-china_main[-c (1:2,6:8,13:14,16:17, 32)]

str(china_main)

#101 FIND HIGHLY CORRELATED VARIABLES-----

#Import the required libraries 

library(mlbench)
library(caret)


# convert  column into numeric for correlation if needed

str(china_main)

#load the data and calculate the correlation matrix

correlationMatrix <- cor(china_main)


# Summarize the correlation matrix:


print(correlationMatrix)

# Plot the correlation matrix

library(corrplot)

corrplot(correlationMatrix)


-------------------- 


china_main$sale_equity <- arules::discretize(china_main$sale_equity, breaks = 2, labels = c("Low","High"))

#102 PREPARE TRAINING SET AND TEST SET [PART C ]----


#divide the data into train and test

set.seed(1)

train_index <- sample(seq_len(nrow(china_main)),floor(0.7 * nrow(china_main)))

train <- china_main[train_index,]
test <- china_main[-train_index,]


# 103 MODEL CREATION [PART D]----


#103C CREATE A XGBOOST MODEL--

#Create list placeholders for the target, categorical, and numeric variables

target<- "sale_equity"
categorical_columns <- c("sale_equity")
numeric_columns <- setdiff(colnames(train),c(categorical_columns,target))


#Convert the categorical factor variables into character. This will be useful for converting them into one-hot-encoded forms

library(tidyverse)

china_main_XG <- china_main %>% mutate_if(is.factor, as.character)


#Combine numeric variables and the one-hot encoded variables from the third step into a single DataFrame named df_final:

df_final <- cbind(china_main_XG[,numeric_columns])

# Convert the target variable into numeric form, as the XGBoost implementation in R doesn't accept factor or character forms:

y <- ifelse(china_main_XG[,target] == "High",1,0)


#Split the df_final dataset into train (70%) and test (30%) datasets:

set.seed(1)
train_index <- sample(seq_len(nrow(df_final)),floor(0.7 * nrow(df_final)))
xgb.train <- df_final[train_index,]
y_train<- y[train_index]
xgb.test <- df_final[-train_index,]
y_test <- y[-train_index]

# Build an XGBoost model using the xgboost function

library(xgboost)

xgb <- xgboost(data = data.matrix(xgb.train), 
               label = y_train, 
               eta = 0.01,
               max_depth = 6, 
               nround=200, 
               subsample = 1,
               colsample_bytree = 1,
               set.seed = 1,
               eval_metric = "logloss",
               objective = "binary:logistic",
               nthread = 4,
               verbose = 1
)


#Make a prediction using the fitted model on the train dataset and create the confusion matrix 

library(caret)

print("Training data results -")
pred_train <- factor(ifelse(predict(xgb,data.matrix(xgb.train),type="class")>0.5,1,0))
confusionMatrix(pred_train,factor(y_train),positive='1')


# make predictions using the fitted model on the test dataset

print("Test data results -")
pred_test <- factor(ifelse(predict(xgb,data.matrix(xgb.test),
                                   type="class")>0.5,1,0))
confusionMatrix(pred_test,factor(y_test),positive='1')

