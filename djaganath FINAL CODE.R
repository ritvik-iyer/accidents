#STAT 154
#Devan Jaganath Code for final project

##TRAINING SET

train <- read.csv("~/Documents/Berkeley Class/final project/final project data/train.csv")


#PRE-PROCESSING AND EDA 

#First will impute zipcode per Ryan's code
zipcode <- as.character(train$Zipcode)
loc <- data.frame(zipcode = zipcode, county = train$County, state = train$State)
loc[] <- lapply(loc, as.character)


missing_zipcodes <- loc %>% filter(zipcode == "")
dim(missing_zipcodes)
#933 missing zipcodes
head(missing_zipcodes)

missing_zipcodes_index = which(loc$zipcode == "")
for (index in missing_zipcodes_index) {
  row <- loc[index, ]
  row_county <- row$county
  row_state <- row$state
  
  avg_location_df <-loc %>% filter(county == row_county & state == row_state)
  most_common_zip <- names(sort(table(avg_location_df$zipcode), decreasing=T))[[1]]
  loc[index, ]$zipcode <- most_common_zip
}

queen_annes_codes_index = which(loc$zipcode == "")
for (index in queen_annes_codes_index) {
  loc[index, ]$zipcode = "21675"
}

for (index in which(loc$City == "")) {
  loc[index, ]$city = "Queen Anne's"
  
}


sum(loc$zipcode == "")



#first variable is weather_timestamp
library(lubridate)
train$stime <- ymd_hms(train$Start_Time, tz = Sys.timezone())

train$weathertime <- ymd_hms(train$Weather_Timestamp, tz = Sys.timezone())

sum(is.na(train$weathertime))
sum(is.na(train$weathertime))/nrow(train)


#44113 missing (1.5%)
#will impute with start time

train$weathertime[which(is.na(train$weathertime))] <- train$stime[which(is.na(train$weathertime))]

train$weathertime
sum(is.na(train$weathertime))

#extract month
train$weather.mon <- month(train$weathertime)
head(train$weather.mon)

#extract hour
train$weather.hr <- hour(train$weathertime)
head(train$weather.hr)

mon.labels <- c("J", "F", "M", "A", "M", "J", "J", "A", "S","O","N", "D")

#plots of month and hour
library(ggplot2)
ggplot(train, aes(weather.mon, fill = Severity)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Weather Month") +
  xlab("Month") + ylab("Frequency") + scale_x_discrete(limits= mon.labels)

ggplot(train, aes(weather.mon, fill = sevcat)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Weather Month") +
  xlab("Month") + ylab("Frequency") + scale_x_discrete(limits= mon.labels)

ggplot(train, aes(weather.hr, fill = as.factor(Severity))) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Weather Hour") +
  xlab("Hour") + ylab("Frequency")

ggplot(train, aes(weather.hr, fill = as.factor(Severity))) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Weather Hour") +
  xlab("Hour") + ylab("Frequency")

ggplot(train, aes(weather.hr, fill = sevcat)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Weather Hour") +
  xlab("Hour") + ylab("Frequency") 


ggplot(train, aes(weather.hr, fill = as.factor(sevcat))) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Weather Hour") +
  xlab("Hour") + ylab("Frequency")

#WILL GO THROUGH EACH VARIABLE, IMPUTING WITH KNN FOR ZIPCODE AND MON/DAY

#temperature

head(train$Temperature.F.)
sum(is.na(train$Temperature.F.))
sum(is.na(train$Temperature.F.))/nrow(train)
#63,189 (2%)
sum(is.na(train$Temperature.F.))/nrow(train)

##IMPUTATION METHOD 2: create imputation function with knn
  
  zipcode.imp <- loc$zipcode
  #removed last 4 digits when dash used
  zipcode.imp <- substr(zipcode.imp, 1, 5)
  weather.md <- train$weather.md
  weather.md <- gsub("-", "", weather.md)
  
  knn.imp <- function(var, zip, md){
    missing <- which(is.na(var))
    knn.mat <- cbind(var, zip, md)
    knn.mat <- matrix(as.numeric(knn.mat), ncol = ncol(knn.mat))
    test.x <- knn.mat[missing, -1]
    train.x <- knn.mat[-missing, -1]
    train.y <- knn.mat[-missing, 1]
    library(FNN)
    var.knn <- knn.reg(train = train.x, test = test.x, y = train.y, k = 1)
    impute <- as.numeric(var)
    for(i in 1:length(missing)){
      impute[missing[i]] <- var.knn$pred[i]
    }
    return(impute)
  }
  
  temp.imp.zip <- knn.imp(train$Temperature.F., zipcode.imp, weather.md)
  summary(temp.imp.zip)
  train$temp.imp.zip <- temp.imp.zip
  
  train$Temperature.F.[1:10]
  temp.imp.zip[1:10]
  train$temp.imp[1:10]

  
  ggplot(train, aes(temp.imp.zip, fill = as.factor(sevcat))) +
    geom_histogram(alpha = 0.5, position="identity")+ggtitle("Temperature") +
    xlab("Temperature") + ylab("Frequency") + labs(fill = "Severity Category")
  
  ggplot(train, aes(temp.imp.zip, fill = as.factor(sevcat))) +
    geom_density(alpha = 0.5, position="identity")+ggtitle("Temperature") +
    xlab("Temperature") + ylab("Frequency") + labs(fill = "Severity Category")
  
  ggplot(train, aes(x= sevcat, y=temp.imp, fill = as.factor(sevcat))) +
    geom_boxplot()+ggtitle("Temperature") + labs(fill = "Severity Category")
  

  
  #wind chill factor
summary(train$Wind_Chill.F.)
sum(is.na(train$Wind_Chill.F.))
sum(is.na(train$Wind_Chill.F.))/nrow(train)
#44% missing, will EXCLUDE

#humidity

summary(train$Humidity...)
sum(is.na(train$Humidity...))/nrow(train)


humid.imp.zip <- knn.imp(train$Humidity..., zipcode.imp, weather.md)
train$humid.imp.zip <- humid.imp.zip

#graphs
ggplot(train, aes(humid.imp.zip, fill = as.factor(sevcat))) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Humidity") +
  xlab("Humidity") + ylab("Frequency") + labs(fill = "Severity Category")

ggplot(train, aes(humid.imp.zip, fill = as.factor(sevcat))) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Humidity") +
  xlab("Humidity") + ylab("Frequency") + labs(fill = "Severity Category")

ggplot(train, aes(x= sevcat, y=humid.imp.zip, fill = as.factor(sevcat))) + ylab("Humidity (%)") +
  geom_boxplot()+ggtitle("Humidity") + labs(fill = "Severity Category")

#"Pressure.in.

summary(train$Pressure.in.)
sum(is.na(train$Pressure.in.))/nrow(train)
pressure.imp.zip <- knn.imp(train$Pressure.in., zipcode.imp, weather.md)
train$pressure.imp.zip <- pressure.imp.zip

#graphs
ggplot(train, aes(pressure.imp.zip, fill = as.factor(sevcat))) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Pressure") +
  xlab("Pressure") + ylab("In") + labs(fill = "Severity Category")

ggplot(train, aes(pressure.imp.zip, fill = as.factor(sevcat))) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Humidity") +
  xlab("Pressure") + ylab("Frequency") + labs(fill = "Severity Category")

ggplot(train, aes(x= sevcat, y=pressure.imp.zip, fill = as.factor(sevcat))) + ylab("Pressure (in)") +
  geom_boxplot()+ggtitle("Pressure") + labs(fill = "Severity Category")

#remove outliers

ggplot(train[train$pressure.imp.zip<(median(train$pressure.imp.zip)+1.5*IQR(train$pressure.imp.zip, na.rm = TRUE)) & train$pressure.imp.zip>(median(train$pressure.imp.zip)-1.5*IQR(train$pressure.imp.zip, na.rm = TRUE)), ], aes(x= sevcat, y=pressure.imp.zip, fill = as.factor(sevcat))) + ylab("Pressure (in)") +
  geom_boxplot()+ggtitle("Pressure without Outliers") + labs(fill = "Severity Category")


summary(train$wind.speed.imp.zip[train$sevcat==0 & train$wind.speed.imp.zip<(median(train$wind.speed.imp.zip)+(1.5*IQR(train$wind.speed.imp.zip))) &  train$wind.speed.imp.zip>(median(train$wind.speed.imp.zip)-(1.5*IQR(train$wind.speed.imp.zip)))])
summary(train$wind.speed.imp.zip[train$sevcat==1 & train$wind.speed.imp.zip<(median(train$wind.speed.imp.zip)+(1.5*IQR(train$wind.speed.imp.zip))) &  train$wind.speed.imp.zip>(median(train$wind.speed.imp.zip)-(1.5*IQR(train$wind.speed.imp.zip)))])

summary(train$wind.speed.imp.zip[train$sevcat==1 & train$wind.speed.imp.zip<(median(train$wind.speed.imp.zip)+(1.5*IQR(train$wind.speed.imp.zip))) &  train$wind.speed.imp.zip>(median(train$wind.speed.imp.zip)-(1.5*IQR(train$wind.speed.imp.zip)))])


#visitibility.mi

summary(train$Visibility.mi.)
sum(is.na(train$Visibility.mi.))/nrow(train)

summary(train$Wind_Direction)
sum(is.na(train$Wind_Direction))/nrow(train) 


visibility.imp.zip <- knn.imp(train$Visibility.mi., zipcode.imp, weather.md)
train$visibility.imp.zip <- visibility.imp.zip

#graphs

summary(train$visibility.imp.zip[train$sevcat==0 & train$visibility.imp.zip<(median(train$visibility.imp.zip)+(1.5*IQR(train$visibility.imp.zip))) &  train$visibility.imp.zip>(median(train$visibility.imp.zip)-(1.5*IQR(train$wind.speed.imp.zip)))])
ggplot(train, aes(x= sevcat, y=visibility.imp.zip, fill = as.factor(sevcat))) + ylab("Visibility (miles)") +
  geom_boxplot()+ggtitle("Visibility") + labs(fill = "Severity Category")


ggplot(train[train$visibility.imp.zip<(median(train$visibility.imp.zip)+1.5*IQR(train$visibility.imp.zip, na.rm = TRUE)), ], aes(x= sevcat, y=visibility.imp.zip, fill = as.factor(sevcat))) + ylab("Visibility (miles)") +
  geom_boxplot()+ggtitle("Visibility without Outliers") + labs(fill = "Severity Category")


## WIND DIRECTION

#need cleaner variable, some are calm or variable as opposed to directions?

wind.d <- train$Wind_Direction

library(forcats)
library(dplyr)
wind.d <- wind.d %>% fct_collapse(S = c("South", "S"), W = c("West", "W"), N = c("North", "N"), E = c("East", "E"),  Calm = c("Calm", "CALM"), Var = c("Variable", "VAR"))
levels(wind.d)
summary(wind.d)


wind.d.matrix <- cbind(as.numeric(wind.d), zipcode.imp, weather.md)
dim(wind.d.matrix)
wind.d.matrix <- matrix(as.numeric(wind.d.matrix), ncol = ncol(wind.d.matrix))

test.wind.d.index <- which(wind.d == "")

wind.d.test.x <- wind.d.matrix[test.wind.d.index, -1]
wind.d.train.x <- wind.d.matrix[-test.wind.d.index, -1]
wind.d.train.y <- wind.d.matrix[-test.wind.d.index, 1]

wind.d.knn.zip <- knn.reg(train = wind.d.train.x, test = wind.d.test.x, y = wind.d.train.y, k = 1)
wind.d.knn.zip$pred

wind.d.imp.zip <- as.numeric(wind.d)
for(i in 1:length(test.wind.d.index)){
  wind.d.imp.zip[test.wind.d.index[i]] <- wind.d.knn.zip$pred[i]
}

wind.d.imp.zip <- factor(wind.d.imp.zip, levels = c(1:19), labels = wind.d.levels)
summary(wind.d.new)

train$wind.d.imp.zip <- wind.d.imp.zip

#graphs
ggplot(train, aes(wind.d.imp.zip, fill = as.factor(sevcat))) +
  geom_histogram(alpha = 0.5, position="identity", stat = "count")+ggtitle("Wind Direction") +
  xlab("Wind Direction") + ylab("Direction") + labs(fill = "Severity Category")

#Wind_Speed.mph

summary(train$Wind_Speed.mph)
sum(is.na(train$Wind_Speed.mph))/nrow(train) 

wind.speed.imp.zip <- knn.imp(train$Wind_Speed.mph., zipcode.imp, weather.md)
train$wind.speed.imp.zip <- wind.speed.imp.zip

#graphs
ggplot(train, aes(wind.speed.imp.zip, fill = as.factor(sevcat))) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Wind Speed") +
  xlab("Wind Speed") + ylab("Frequency") + labs(fill = "Severity Category")

ggplot(train, aes(wind.speed.imp.zip, fill = as.factor(sevcat))) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Wind Speed") +
  xlab("Wind Speed") + ylab("Frequency") + labs(fill = "Severity Category")

ggplot(train, aes(x= sevcat, y=wind.speed.imp.zip, fill = as.factor(sevcat))) + ylab("MPH") +
  geom_boxplot()+ggtitle("Wind Speed") + labs(fill = "Severity Category")

ggplot(train[train$wind.speed.imp.zip<1.5*IQR(train$wind.speed.imp.zip, na.rm = TRUE),], aes(x= sevcat, y=wind.speed.imp.zip, fill = as.factor(sevcat))) + ylab("MPH") +
  geom_boxplot()+ggtitle("Wind Speed without Outliers") + labs(fill = "Severity Category")


ggplot(train[train$wind.speed.imp<1.5*IQR(train$wind.speed.imp, na.rm = TRUE), ], aes(x= as.factor(sevcat), y =wind.speed.imp)) +
  geom_boxplot()

ggplot(train[train$wind.speed.imp.zip<(median(train$wind.speed.imp.zip)+1.5*IQR(train$wind.speed.imp.zip, na.rm = TRUE)) & train$wind.speed.imp.zip>(median(train$wind.speed.imp.zip)-1.5*IQR(train$wind.speed.imp.zip, na.rm = TRUE)), ], aes(x= sevcat, y=wind.speed.imp.zip, fill = as.factor(sevcat))) + ylab("Speed (mph)") +
  geom_boxplot()+ggtitle("Wind Speed without Outliers") + labs(fill = "Severity Category")


##PRECIPITATION


Precipitation.in.
summary(train$Precipitation.in.)
sum(is.na(train$Precipitation.in.))/nrow(train)
#48% will not use

ggplot(train, aes(Precipitation.in., fill = Severity)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Precipitation") +
  xlab("in") + ylab("Frequency")



#Weather_Condition
summary(train$Weather_Condition)
levels(train$Weather_Condition)
69146/nrow(train) 

#will make a series of logical variables with key words

light <- grepl("light", train$Weather_Condition, ignore.case = TRUE)
heavy <- grepl("heavy", train$Weather_Condition, ignore.case = TRUE)
rain <- grepl("rain|precip", train$Weather_Condition, ignore.case = TRUE)
snow <- grepl("snow|wintry|hail|sleet|ice|freez", train$Weather_Condition, ignore.case = TRUE)
storm <- grepl("storm|thunder|tornado", train$Weather_Condition, ignore.case = TRUE)
windy <-  grepl("wind|blow|tornado", train$Weather_Condition, ignore.case = TRUE)
haze <-  grepl("fog|smoke|volcano|ash|sand|dust", train$Weather_Condition, ignore.case = TRUE)

train <- cbind(train, light, heavy, rain, snow, storm, windy, haze)

save(train, file = "train43020.Rdata")


ggplot(train, aes(rain, fill = as.factor(sevcat))) +
  geom_histogram(alpha = 0.5, position="identity", stat = "count")+ggtitle("Rainy Weather")

colnames(train)

#final variables, ultimately used knn imputation approach as completed all data points and faster
#issue with first approach is that if county was missing, it became missing
#dropped Wind_Chill.F. and Precipitiation due to large amount of missing data
weather.vars <- train[ ,c(56, 57, 58, 78, 74, 75, 76, 73, 77, 65, 66, 67, 68, 69, 70, 71)]


write.csv(weather.vars, "stat154_weathervars.csv")



#TEST SET CLEANING

test <- read.csv("~/Documents/Berkeley Class/final project/final project data/test.csv")

#First will impute zipcode per Ryan's code
zipcode <- as.character(test$Zipcode)
loc <- data.frame(zipcode = zipcode, county = test$County, state = test$State)
loc[] <- lapply(loc, as.character)


missing_zipcodes <- loc %>% filter(zipcode == "")
dim(missing_zipcodes)
#359 missing zipcodes
head(missing_zipcodes)

missing_zipcodes_index = which(loc$zipcode == "")
for (index in missing_zipcodes_index) {
  row <- loc[index, ]
  row_county <- row$county
  row_state <- row$state
  
  avg_location_df <-loc %>% filter(county == row_county & state == row_state)
  most_common_zip <- names(sort(table(avg_location_df$zipcode), decreasing=T))[[1]]
  loc[index, ]$zipcode <- most_common_zip
}



sum(loc$zipcode == "")

#no missing zipcodes

#first variable is weather_timestamp
library(lubridate)
test$stime <- ymd_hms(test$Start_Time, tz = Sys.timezone())

test$weathertime <- ymd_hms(test$Weather_Timestamp, tz = Sys.timezone())

sum(is.na(test$weathertime))
sum(is.na(test$weathertime))/nrow(test)


#18531 missing, 1.5%
#will impute with start time

test$weathertime[which(is.na(test$weathertime))] <- test$stime[which(is.na(test$weathertime))]

test$weathertime
sum(is.na(test$weathertime))

#extract month
test$weather.mon <- month(test$weathertime)
head(test$weather.mon)

#extract hour
test$weather.hr <- hour(test$weathertime)
head(test$weather.hr)

mon.labels <- c("J", "F", "M", "A", "M", "J", "J", "A", "S","O","N", "D")

#will use knn imputation

#temperature

head(test$Temperature.F.)
sum(is.na(test$Temperature.F.))
sum(is.na(test$Temperature.F.))/nrow(test)
#26.7K (2%)
sum(is.na(test$Temperature.F.))/nrow(test)

#IMPUTATION
weather.ymd <- format(as.Date(test$weathertime), "%Y-%m-%d")
weather.md <- format(as.Date(test$weathertime), "%m-%d")

test$weather.ymd <- weather.ymd
test$weather.md <- weather.md

zipcode.imp <- loc$zipcode
#removed last 4 digits when dash used
zipcode.imp <- substr(zipcode.imp, 1, 5)
weather.md <- test$weather.md
weather.md <- gsub("-", "", weather.md)

knn.imp <- function(var, zip, md){
  missing <- which(is.na(var))
  knn.mat <- cbind(var, zip, md)
  knn.mat <- matrix(as.numeric(knn.mat), ncol = ncol(knn.mat))
  test.x <- knn.mat[missing, -1]
  train.x <- knn.mat[-missing, -1]
  train.y <- knn.mat[-missing, 1]
  library(FNN)
  var.knn <- knn.reg(train = train.x, test = test.x, y = train.y, k = 1)
  impute <- as.numeric(var)
  for(i in 1:length(missing)){
    impute[missing[i]] <- var.knn$pred[i]
  }
  return(impute)
}

temp.imp.zip <- knn.imp(test$Temperature.F., zipcode.imp, weather.md)
summary(temp.imp.zip)
test$temp.imp.zip <- temp.imp.zip

#wind chill factor
summary(test$Wind_Chill.F.)
sum(is.na(test$Wind_Chill.F.))
sum(is.na(test$Wind_Chill.F.))/nrow(test)
#45% missing, will not include

#humidity

summary(test$Humidity...)
sum(is.na(test$Humidity...))/nrow(test)


humid.imp.zip <- knn.imp(test$Humidity..., zipcode.imp, weather.md)
test$humid.imp.zip <- humid.imp.zip
summary(test$humid.imp.zip)


#"Pressure.in.

summary(test$Pressure.in.)
sum(is.na(test$Pressure.in.))/nrow(test)
pressure.imp.zip <- knn.imp(test$Pressure.in., zipcode.imp, weather.md)
test$pressure.imp.zip <- pressure.imp.zip
summary(test$pressure.imp.zip)


#visitibility.mi

summary(test$Visibility.mi.)
sum(is.na(test$Visibility.mi.))/nrow(test)
visibility.imp.zip <- knn.imp(test$Visibility.mi., zipcode.imp, weather.md)
test$visibility.imp.zip <- visibility.imp.zip
summary(test$visibility.imp.zip)



## WIND DIRECTION

#need cleaner variable, some are calm or variable as opposed to directions?

wind.d <- test$Wind_Direction

library(forcats)
library(dplyr)
wind.d <- wind.d %>% fct_collapse(S = c("South", "S"), W = c("West", "W"), N = c("North", "N"), E = c("East", "E"),  Calm = c("Calm", "CALM"), Var = c("Variable", "VAR"))
levels(wind.d)
summary(wind.d)

wind.d.matrix <- cbind(as.numeric(wind.d), zipcode.imp, weather.md)
dim(wind.d.matrix)
wind.d.matrix <- matrix(as.numeric(wind.d.matrix), ncol = ncol(wind.d.matrix))

test.wind.d.index <- which(wind.d == "")

wind.d.test.x <- wind.d.matrix[test.wind.d.index, -1]
wind.d.train.x <- wind.d.matrix[-test.wind.d.index, -1]
wind.d.train.y <- wind.d.matrix[-test.wind.d.index, 1]

wind.d.knn.zip <- knn.reg(train = wind.d.train.x, test = wind.d.test.x, y = wind.d.train.y, k = 1)
wind.d.knn.zip$pred

wind.d.imp.zip <- as.numeric(wind.d)
for(i in 1:length(test.wind.d.index)){
  wind.d.imp.zip[test.wind.d.index[i]] <- wind.d.knn.zip$pred[i]
}

wind.d.imp.zip <- factor(wind.d.imp.zip, levels = c(1:19), labels = wind.d.levels)
summary(wind.d.new)

test$wind.d.imp.zip <- wind.d.imp.zip


#Wind_Speed.mph

summary(test$Wind_Speed.mph)
sum(is.na(test$Wind_Speed.mph))/nrow(test) 
wind.speed.imp.zip <- knn.imp(test$Wind_Speed.mph., zipcode.imp, weather.md)
test$wind.speed.imp.zip <- wind.speed.imp.zip
summary(test$wind.speed.imp.zip)



##PRECIPITATION
summary(test$Precipitation.in.)
sum(is.na(test$Precipitation.in.))/nrow(test)
#49% will not use


#Weather_Condition


#will make a series of logical variables with key words

light <- grepl("light", test$Weather_Condition, ignore.case = TRUE)
heavy <- grepl("heavy", test$Weather_Condition, ignore.case = TRUE)
rain <- grepl("rain|precip", test$Weather_Condition, ignore.case = TRUE)
snow <- grepl("snow|wintry|hail|sleet|ice|freez", test$Weather_Condition, ignore.case = TRUE)
storm <- grepl("storm|thunder|tornado", test$Weather_Condition, ignore.case = TRUE)
windy <-  grepl("wind|blow|tornado", test$Weather_Condition, ignore.case = TRUE)
haze <-  grepl("fog|smoke|volcano|ash|sand|dust", test$Weather_Condition, ignore.case = TRUE)

test <- cbind(test, light, heavy, rain, snow, storm, windy, haze)

save(test, file = "test5421.Rdata")

colnames(test)

#final variables, ultimately used knn imputation approach as completed all data points and faster
#issue with first approach is that if county was missing, it became missing
#dropped Wind_Chill.F. and Precipitiation due to large amount of missing data
weather.test.vars <- test[ ,c(1, 50:67)]


write.csv(weather.test.vars, "stat154_weather_test_vars.csv")


#ALGORITHM BUILDING AND TESTING

#load data
data <-read.csv("train_final.csv")
val <- read.csv("val_final.csv")


test <- read.csv("test_final.csv")

data$Weekday <- as.integer(as.logical(data$Weekday))
data$Rush.Hour <- as.integer(as.logical(data$Rush.Hour))
data$Severity <- as.factor(data$Severity)
val$Weekday <- as.integer(as.logical(val$Weekday))

val$Rush.Hour <- as.integer(as.logical(val$Rush.Hour))
val$Severity <- as.factor(val$Severity)

test$Weekday <- as.integer(as.logical(test$Weekday))

test$Rush.Hour <- as.integer(as.logical(test$Rush.Hour))

x.train <- as.matrix(data[,-81])
y.train <- as.factor(data[,81])
x.val <- as.matrix(val[,-81])
y.val <- val[,81]

#elastic net
library(glmnet)
data.lasso <- cv.glmnet(x=x.train, y= y.train, family = "binomial", standardize = FALSE, type.measure = "class", nfolds = 5, alpha = 1)
pred.lasso <- predict(data.lasso, x.val, s = "lambda.min", type = "class")
pred.lasso.train <- predict(data.lasso, x.train, s = "lambda.min", type = "class")


table(pred.lasso, y.val)
sum(pred.lasso.train ==y.train)/length(y.train)

sum(pred.lasso ==y.val)/length(y.val)



library(MASS)

data.step <- stepAIC()

library(caret)


fitControl <- trainControl(method = "cv", number = 5, search = "random")


#run in parallel 

library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

set.seed(1234)


#lasso
library(glmnet)
data.lasso <- cv.glmnet(x=x.train, y= y.train, family = "binomial", standardize = FALSE, type.measure = "class", nfolds = 5, alpha = 1)
pred.lasso <- predict(data.lasso, x.val, s = "lambda.min", type = "class")
pred.lasso.train <- predict(data.lasso, x.train, s = "lambda.min", type = "class")


table(pred.lasso, y.val)
sum(pred.lasso.train ==y.train)/length(y.train)

sum(pred.lasso ==y.val)/length(y.val)
coef(data.lasso)
plot(data.lasso)


#elastic net
data.elastic <- train(Severity ~ ., data = data, 
                      method = "glmnet", 
                      trControl = fitControl, family = "binomial", standardize = FALSE, type.measure = "class")

elastic.pred <-predict(data.elastic, x.val, type = "prob")

elastic.pos <- NA
elastic.pos[elastic.pred[,2]>=0.5] <- 1
elastic.pos[elastic.pred[,2]<0.5] <- 0
table(elastic.pos, y.val)
sum(elastic.pos ==y.val)/length(y.val)

train.small.index <- sample(1:nrow(data), 0.1*nrow(data))
length(train.small.index)

elastic.test.pred <-predict(data.elastic, test, type = "prob")
elastic.test.pos <- NA
elastic.test.pos[elastic.test.pred[,2]>=0.5] <- 1
elastic.test.pos[elastic.test.pred[,2]<0.5] <- 0
table(elastic.test.pos)
elastic.test.df <- data.frame(ID = test$ID, Severity = as.numeric(elastic.test.pos))
write.csv(elastic.test.df, file = "elasticpreds.csv", row.names = FALSE, quote = FALSE)

sum(elastic.test.pos ==y.val)/length(y.val)

#keep vars sig from lasso c(presure, wind.speed, distance, blocked, weekday, traffic signal, x3, x4, CA, mapquest)]

train.small <- data[train.small.index, c(3,5,6,8,13,26,28,29,32, 79, 81)]

data.class.xgb <- train(Severity ~ ., data = train.small, 
                        method = "xgbTree", 
                        trControl = fitControl)

xgb.pred <- predict(data.class.xgb, x.val, type = "prob")

#use 0.5 probability cut-off
xgb.pos <- NA
xgb.pos[xgb.pred[,2]>=0.5] <- 1
xgb.pos[xgb.pred[,2]<0.5] <- 0
table(xgb.pos, y.val)
sum(xgb.pos ==y.val)/length(y.val)

#predict in test data
xgb.test <- predict(data.class.xgb, test, type = "prob")
xgb.test.class <- NA
xgb.test.class[xgb.test[,2]>=0.5] <- 1
xgb.test.class[xgb.test[,2]<0.5] <- 0
table(xgb.test.class)
xgb.test.preds <- data.frame(ID = test$ID, Severity = as.numeric(xgb.test.class))
write.csv(xgb.test.preds, file = "xgbpreds.csv", row.names = FALSE, quote = FALSE)

#try with 25% data
set.seed(1234)
train.small.25pct <- sample(1:nrow(data), 0.25*nrow(data))
train.25pct <- data[train.small.25pct, c(3,5,6,8,13,26,28,29,32, 79, 81)]

data.25.xgb <- train(Severity ~ ., data = train.25pct, 
                     method = "xgbTree", 
                     trControl = fitControl)
xgb.25.pred <- predict(data.25.xgb, x.val, type = "prob")

#predict in test data
xgb.25.pos <- NA
xgb.25.pos[xgb.25.pred[,2]>=0.5] <- 1
xgb.25.pos[xgb.25.pred[,2]<0.5] <- 0
table(xgb.25.pos, y.val)
sum(xgb.25.pos ==y.val)/length(y.val)

#overall performance is similar whether 25% or 10%


#kernel svm - tried but ran out of memory
library(kernlab)
data.ksvm <- train(Severity ~ ., data = train.small, 
                   method = "svmRadial", 
                   trControl = fitControl)

# stop parallelization
stopCluster(cl)

