#EDA
#Devan Jaganath

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


#I imputed two ways, one by taking average of variable among accidents at the same county and month/day
#second approach I did 1 neearest neighbors to predict missing values with zipcode and month/day

#temperature

head(train$Temperature.F.)
sum(is.na(train$Temperature.F.))
sum(is.na(train$Temperature.F.))/nrow(train)
#63,189 (2%)
sum(is.na(train$Temperature.F.))/nrow(train)

#IMPUTATION APPROACH 1
#will take the average temperature from the same zipcode on the same day

weather.ymd <- format(as.Date(train$weathertime), "%Y-%m-%d")
weather.md <- format(as.Date(train$weathertime), "%m-%d")

train$weather.ymd <- weather.ymd
train$weather.md <- weather.md
train$weather.mon <- train$weather.mon


temp[9]
train$City[9]
train$weather.mon[9]

mean(train$Temperature.F.[train$City==train$City[9] & train$weather.md == train$weather.md[9]], na.rm = TRUE)

#write an imputing function

temp.impute <- function(a){
  impute <- a
  for(i in 1:length(a)){
    if(is.na(a[i]) == TRUE){
      impute[i] <- mean(train$Temperature.F.[train$County==train$County[i] & train$weather.md == train$weather.md[i]], na.rm = TRUE)
    }
  }
  return(impute)  
}

temp.impute <- function(a){
  missing.index <- which(is.na(a))
  impute <- a
  for(i in 1:length(missing.index)){
    impute[missing.index[i]]<-  mean(train$Temperature.F.[train$County==train$County[missing.index[i]] & train$weather.md == train$weather.md[missing.index[i]]], na.rm = TRUE)
    }
  return(impute)  
}
  
temp.new <- temp.impute(temp)
train$temp.imp <- temp.new

library(ggplot2)
  
  ggplot(train, aes(temp.imp, fill = as.factor(Severity))) +
    geom_histogram(alpha = 0.5, position="identity")+ggtitle("Temperature") +
    xlab("Temperature") + ylab("Frequency") 
  
  ggplot(train, aes(temp.imp, fill = as.factor(Severity))) +
    geom_density(alpha = 0.5, position="identity")+ggtitle("Temperature") +
    xlab("Temperature") + ylab("Frequency") 
  
  ggplot(train, aes(x= Severity, y=temp.imp, fill = as.factor(Severity))) +
    geom_boxplot()+ggtitle("Temperature")
  
  ggplot(train, aes(x= sevcat, y=temp.imp, fill = as.factor(sevcat))) +
    geom_boxplot()+ggtitle("Temperature")
  
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
  
  #there is a diffference  
  
  
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
#44% missing, will not include

#humidity

summary(train$Humidity...)
sum(is.na(train$Humidity...))/nrow(train)

#IMPUTE METHOD 1

humidity.impute <- function(a){
  missing.index <- which(is.na(a))
  impute <- a
  for(i in 1:length(missing.index)){
    impute[missing.index[i]]<-  mean(train$Humidity...[train$City==train$City[missing.index[i]] & train$weather.md == train$weather.md[missing.index[i]]], na.rm = TRUE)
  }
  return(impute)  
}

humidity <- train$Humidity...

humidity.new <- humidity.impute(humidity)
summary(humidity.new)
sum(is.na(humidity.new))/length(humidity.new)
#down to 0.7%

train$humid.imp <- humidity.new

#IMPUTE #2
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

#METHOD 1

pressure.impute <- function(a){
  missing.index <- which(is.na(a))
  impute <- a
  for(i in 1:length(missing.index)){
    impute[missing.index[i]]<-  mean(train$Pressure.in.[train$City==train$City[missing.index[i]] & train$weather.md == train$weather.md[missing.index[i]]], na.rm = TRUE)
  }
  return(impute)  
}

pressure <- train$Pressure.in.

pressure.new <- pressure.impute(pressure)
summary(pressure.new)
sum(is.na(pressure.new))/length(pressure.new)

train$pressure.imp <- pressure.new

save(train, file = "train.Rdata")


#METHOD 2
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

ggplot(train[train$pressure.imp.zip<1.5*IQR(train$pressure.imp.zip, na.rm = TRUE), ], aes(x= sevcat, y=pressure.imp.zip, fill = as.factor(sevcat))) + ylab("Pressure (in)") +
  geom_boxplot()+ggtitle("Pressure without Outliers") + labs(fill = "Severity Category")



#visitibility.mi

summary(train$Visibility.mi.)
sum(is.na(train$Visibility.mi.))/nrow(train)

summary(train$Wind_Direction)
sum(is.na(train$Wind_Direction))/nrow(train) 

#METHOD 1

visibiity.impute <- function(a){
  missing.index <- which(is.na(a))
  impute <- a
  for(i in 1:length(missing.index)){
    impute[missing.index[i]]<-  mean(train$Visibility.mi.[train$City==train$City[missing.index[i]] & train$weather.md == train$weather.md[missing.index[i]]], na.rm = TRUE)
  }
  return(impute)  
}

visibility <- train$Visibility.mi.

visibility.new <- visibiity.impute(visibility)
summary(visibility.new)
sum(is.na(visibility.new))/length(visibility.new)

train$vis.imp <- visibility.new

#METHOD 2
visibility.imp.zip <- knn.imp(train$Visibility.mi., zipcode.imp, weather.md)
train$visibility.imp.zip <- visibility.imp.zip

#graphs
ggplot(train, aes(pressure.imp.zip, fill = as.factor(sevcat))) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Pressure") +
  xlab("Pressure") + ylab("In") + labs(fill = "Severity Category")

ggplot(train, aes(pressure.imp.zip, fill = as.factor(sevcat))) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Humidity") +
  xlab("Pressure") + ylab("Frequency") + labs(fill = "Severity Category")

ggplot(train, aes(x= sevcat, y=pressure.imp.zip, fill = as.factor(sevcat))) + ylab("Pressure (in)") +
  geom_boxplot()+ggtitle("Pressure") + labs(fill = "Severity Category")


## WIND DIRECTION

#need cleaner variable, some are calm or variable as opposed to directions?

wind.d <- train$Wind_Direction

library(forcats)
library(dplyr)
wind.d <- wind.d %>% fct_collapse(S = c("South", "S"), W = c("West", "W"), N = c("North", "N"), E = c("East", "E"),  Calm = c("Calm", "CALM"), Var = c("Variable", "VAR"))
levels(wind.d)
summary(wind.d)

#method 1, this time used city rather than make a matrix with wind.d, city and month-day
wind.d.levels <- levels(wind.d)

wind.d.matrix <- cbind(as.numeric(wind.d), as.numeric(train$City), train$weather.md)
dim(wind.d.matrix)

wind.d.matrix[,3] <- gsub("-", "", wind.d.matrix[,3])
wind.d.matrix <- matrix(as.numeric(wind.d.matrix), ncol = ncol(wind.d.matrix))

head(wind.d.matrix)

which(wind.d == "")

wind.d[1:10]

test.wind.d.index <- which(wind.d == "")

wind.d.test.x <- wind.d.matrix[test.wind.d.index, -1]
dim(wind.d.test.x)
wind.d.train.x <- wind.d.matrix[-test.wind.d.index, -1]
dim(wind.d.train.x)

wind.d.train.y <- wind.d.matrix[-test.wind.d.index, 1]

library(FNN)

wind.d.knn <- knn.reg(train = wind.d.train.x, test = wind.d.test.x, y = wind.d.train.y, k = 1)
wind.d.knn$pred

wind.d.new <- as.numeric(wind.d)
for(i in 1:length(test.wind.d.index)){
  wind.d.new[test.wind.d.index[i]] <- wind.d.knn$pred[i]
}

wind.d.new <- factor(wind.d.new, levels = c(1:19), labels = wind.d.levels)
summary(wind.d.new)

train$wind.d.imp <- wind.d.new



#METHOD 2

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


#METHOD 1 with city/date approach
wind.speed <- train$Wind_Speed.mph.

wind.speed.mat <- cbind(wind.speed, train$City, train$weather.md)
wind.speed.mat[,3] <- gsub("-", "", wind.speed.mat[,3])
wind.speed.mat <- matrix(as.numeric(wind.speed.mat), ncol = ncol(wind.speed.mat))
head(wind.speed.mat)

test.wind.speed.index <- which(is.na(wind.speed))
wind.speed.test.x <- wind.speed.mat[test.wind.speed.index, -1]
dim(wind.speed.test.x)
wind.speed.train.x <- wind.speed.mat[-test.wind.speed.index, -1]
dim(wind.speed.train.x)

wind.speed.train.y <- wind.speed.mat[-test.wind.speed.index, 1]

wind.speed.knn <- knn.reg(train = wind.speed.train.x, test = wind.speed.test.x, y = wind.speed.train.y, k = 1)
wind.speed.knn$pred

wind.speed.new <- wind.speed
for(i in 1:length(test.wind.speed.index)){
  wind.speed.new[test.wind.speed.index[i]] <- wind.speed.knn$pred[i]
}

summary(wind.speed.new)
train$wind.speed.imp <- wind.speed.new

ggplot(train, aes(wind.speed.imp, fill = sevcat)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Wind Speed") +
  xlab("mph") + ylab("Frequency")

ggplot(train[train$wind.speed.imp<1.5*IQR(train$wind.speed.imp, na.rm = TRUE), ], aes(x= as.factor(sevcat), y =wind.speed.imp)) +
  geom_boxplot()

train %>% boxplot(wind.speed.imp~sev.cat)

#METHOD 2
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






