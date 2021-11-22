# Machine-Learning-for-Weather-Prediction
#Importing the dataset using the import function on R
library(readxl)
PJ_Solar_Dataset <- read_excel("Semester 1/Machine Learning/Datasets/PJ Solar Dataset.xlsx")
View(PJ_Solar_Dataset)
x <- PJ_Solar_Dataset
#-----------------------------------------------------------------
#Re-naming the dataset as x for simplicity purposes
View(x) 
#-----------------------------------------------------------------
class(x) #View type of data frame
head(x) #View first 6 rows of data
str(x) #Display structure of data
#-------------------------------------------------------------------
#Renaming columns that will be used for analysis
install.packages("tidyverse")
library(tidyverse)
my_data <- as_tibble(x)
colnames(my_data)
names(my_data)[names(my_data) == "LAT"] <- "Latitude"
names(my_data)[names(my_data) == "LON"] <- "Longitude"
names(my_data)[names(my_data) == "MO"] <- "Month"
names(my_data)[names(my_data) == "DY"] <- "Day"
names(my_data)[names(my_data) == "PRECTOT"] <- "Precipitation"
names(my_data)[names(my_data) == "SPECHUM2M"] <- "SpecificHumidity"
names(my_data)[names(my_data) == "RH2M"] <- "RelativeHumidity"
names(my_data)[names(my_data) == "SURPRESSURE"] <- "Pressure"
names(my_data)[names(my_data) == "INSOLATIONCLEARNESS"] <- "Index"
names(my_data)[names(my_data) == "WS10M"] <- "WindSpeed"
names(my_data)[names(my_data) == "T2M"] <- "AmbientTemperature"
names(my_data)[names(my_data) == "CLRSKY_SFC_SW_DWN"] <- "ClearSky"
names(my_data)[names(my_data) == "ALLSKY_SFC_SW_DWN"] <- "Irradiation"
#----------------------------------------------------------------------
str(my_data)
x2=subset(my_data,select = -c(1,2,3)) #Column 1,2 and 3 are repeating values and were removed.
#--------------------------------------------------------------------------
install.packages('DataExplorer')
library(DataExplorer)
library(dplyr)
library(ggplot2)
#-----------------------------------------------------------------------------
#EDA of Dataset
plot_histogram(x2$Precipitation) #Histogram for Precipitation
plot_density(x2$Pressure) #Density Chart for Pressure
barplot(table(x2$SpecificHumidity), main="Specific Humidity", col=c("skyblue","red", "lightgreen"))
barplot(x2$WindSpeed, main="Wind Speed", col = c("skyblue", "red"))
hist(x2$AmbientTemperature, main = "Ambient Temperature", col = c("red"))
boxplot(x2$Precipitation, horizontal=TRUE, main = "Precipitation", col = "lightgreen")
boxplot(x2$Irradiation, main = "Irradiation", col = c("skyblue"))
#---------------------------------------------------------------------------------------
#Impute missing value
# Removing columns with missing Values
sum (is.na(x2))
colSums(sapply(x2,is.na))

#-------------------------------------------------------------------------------
#Identifying Outliers in Precipitation
Q <- quantile(x2$Precipitation, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(x2$Precipitation)
up <-  Q[2]+1.5*iqr # Upper Range  
low <- Q[1]-1.5*iqr # Lower Range
#Renaming dataset as eliminated after eliminating outliers
eliminated <- subset(x2, x2$Precipitation > (Q[1] - 1.5*iqr) & x2$Precipitation < (Q[2]+1.5*iqr))
boxplot(eliminated$Precipitation, horizontal = TRUE, main = "Precipitation Outliers Removed", col = c("red"))
#-----------------------------------------------------------------------------------
# SVM REGRESSION for Dataset

library(ggplot2)
library(e1071)
# Splitting the dataset "eliminated" into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
#Since the dataset is large for calculations, only the first 120 rows were taken.
x3 <- eliminated[-c(116:338),] #Only take data from January to April
df = subset(x3, select = -c(7:14) ) #Remove redundant columns that is not included in the test and training data
# Check missing values
sum(is.na(df))
colSums(is.na(df))

# Imputing missing values using median
library(caret)
preProcValues <- preProcess(df, method = "medianImpute")
ds <- predict(preProcValues, df) #Renaming new dataset as ds with missing values imputed
sum(is.na(ds))
View(ds)

df1 <- ds[,-1] #Remove the Month Column
df2 <- df1[,-1] #Remove the Day Column
split = sample.split(df2$Precipitation, SplitRatio = 0.8)
training_set = subset(df2, split == TRUE)
test_set = subset(df2, split == FALSE)

# ~~~~~~~~~~~~~~~~~~~~  Default SVM Model using the RBF kernel ~~~~~~~~~~~~~~~~~~~~~
svm_rbf <- svm(Precipitation~., data = training_set)
summary(svm_rbf)

pred = predict (svm_rbf, test_set)
pred
table(pred, test_set$Precipitation)

library(caret)
summary(pred)
RMSE(pred, test_set$Precipitation) # Root mean squared error
MAE(pred, test_set$Precipitation) # Mean Absolute Error

# ~~~~~~~~~~~~~~~~~~~~   SVM model using the Linear model  ~~~~~~~~~~~~~~~~~~~~~
svm_linear = svm (Precipitation~., data = training_set, kernel = "linear")
summary (svm_linear)

pred2 = predict (svm_linear, test_set)
pred2

library(caret)
summary(pred2)
RMSE(pred2, test_set$Precipitation) # Root mean squared error
MAE(pred2, test_set$Precipitation) # Mean Absolute Error

# ~~~~~~~~~~~~~~~~~~~~   SVM model using sigmoid kernal  ~~~~~~~~~~~~~~~~~~~~~
svm_sigmoid = svm (Precipitation~., data = training_set, kernel = "sigmoid")
summary (svm_sigmoid)

pred3 = predict (svm_sigmoid, test_set)
pred3
table(pred3, test_set$Precipitation)

library(caret)
summary(pred3)
RMSE(pred3, test_set$Precipitation) # Root mean squared error
MAE(pred3, test_set$Precipitation) # Mean Absolute Error
#--------------------------------------------------------------------
cat("RMSE using RBF Kernal is ", RMSE(pred, test_set$Precipitation))
cat("RMSE using LINEAR Kernal is ", RMSE(pred2, test_set$Precipitation))
cat("RMSE using SIGMOID Kernal is ", RMSE(pred3, test_set$Precipitation))

#---------------------------------------------------------------------------
# Next, fitting Multiple Linear Regression to the Training set is applied
regressor1 = lm(formula = Precipitation ~ ., data = training_set)
summary(regressor1)

# Predicting the Test set results
y_pred = predict(regressor1, newdata = test_set)
y_pred
table(y_pred, test_set$Precipitation) # Comparing the predicted and actual valu

#Error Rate Estimator
sigma(regressor1)/mean(training_set$Precipitation)

# RMSE on test set
sqrt(mean((test_set$Precipitation-y_pred)^2))

library(caret)
summary(y_pred)
RMSE(y_pred, test_set$Precipitation) # Root mean squared error
MAE(y_pred, test_set$Precipitation) # Mean Absolute Error

#---------------------------------------------------------------------
coefficients(regressor1) # model coefficients
confint(regressor1, level=0.95) # CIs for model parameters
fitted(regressor1) # predicted values
residuals(regressor1) # residuals
anova(regressor1) # anova table
vcov(regressor1) # covariance matrix for model parameters
influence(regressor1) # regression diagnostics
#-------------------------------------------------------------------
#Correlation analysis between variables of the dataset
cors <- cor(df2)
print(cors)
#-----------------------------------------------------------------------
#Pearson correlation graph
install.packages("ggpubr")
library("ggpubr")
#Precipitation and Relative Humidity
ggscatter(df2, x = "Precipitation", y = "RelativeHumidity", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Precipitation", ylab = "Relaltive Humidity")

#Precipitation and Surface Pressure
ggscatter(df2, x = "Precipitation", y = "Pressure", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Precipitation", ylab = "Surface Pressure")

#Precipitation and Wind Speed
ggscatter(df2, x = "Precipitation", y = "WindSpeed", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Precipitation", ylab = "Wind Speed")

#Pressure and Wind Speed
ggscatter(df2, x = "Pressure", y = "WindSpeed", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Pressure", ylab = "Wind Speed")

#Ambient Temperature and Wind Speed
ggscatter(df2, x = "AmbientTemperature", y = "WindSpeed", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Temperature", ylab = "Wind Speed")

#Irradiation and Specific Humidity
ggscatter(df2, x = "Irradiation", y = "SpecificHumidity", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Irradiation", ylab = "Specific Humidity")
#-----------------------------------------------------------------------
