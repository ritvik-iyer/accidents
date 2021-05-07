
#I will focus on variables 23-32
eda.vars <- train[,c(4,23:32, 50)]
eda.vars$sevcat <- as.factor(eda.vars$sevcat)
eda.vars$Severity <- as.factor(eda.vars$Severity)

dim(eda.vars)
colnames(eda.vars)

summary(eda.vars)

#first variable is weather_timestamp
library(lubridate)
train$stime <- ymd_hms(train$Start_Time, tz = Sys.timezone())

eda.vars$weathertime <- ymd_hms(eda.vars$Weather_Timestamp, tz = Sys.timezone())

sum(is.na(eda.vars$weathertime))
sum(is.na(eda.vars$weathertime))/nrow(eda.vars)
#44113 missing (1.5%)
#will impute with start time

eda.vars$weathertime[which(is.na(eda.vars$weathertime))] <- train$stime[which(is.na(eda.vars$weathertime))]

eda.vars$weathertime
sum(is.na(eda.vars$weathertime))

#extract month
eda.vars$weather.mon <- month(eda.vars$weathertime)
head(eda.vars$weather.mon)

#extract hour
eda.vars$weather.hr <- hour(eda.vars$weathertime)
head(eda.vars$weather.hr)

mon.labels <- c("J", "F", "M", "A", "M", "J", "J", "A", "S","O","N", "D")

#plots of month and hour
library(ggplot2)
ggplot(eda.vars, aes(weather.mon, fill = Severity)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Weather Month") +
  xlab("Month") + ylab("Frequency") + scale_x_discrete(limits= mon.labels)

ggplot(eda.vars, aes(weather.mon, fill = sevcat)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Weather Month") +
  xlab("Month") + ylab("Frequency") + scale_x_discrete(limits= mon.labels)

ggplot(eda.vars, aes(weather.hr, fill = as.factor(Severity))) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Weather Hour") +
  xlab("Hour") + ylab("Frequency")

ggplot(eda.vars, aes(weather.hr, fill = as.factor(Severity))) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Weather Hour") +
  xlab("Hour") + ylab("Frequency")

ggplot(eda.vars, aes(weather.hr, fill = sevcat)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Weather Hour") +
  xlab("Hour") + ylab("Frequency") 


ggplot(eda.vars, aes(weather.hr, fill = as.factor(sevcat))) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Weather Hour") +
  xlab("Hour") + ylab("Frequency")


#temperature

head(eda.vars$Temperature.F.)
sum(is.na(eda.vars$Temperature.F.))
sum(is.na(eda.vars$Temperature.F.))/nrow(eda.vars)
#63,189 (2%)
sum(is.na(eda.vars$Temperature.F.))/nrow(eda.vars)
#will take the average temperature from the same zipcode on the same day

weather.ymd <- format(as.Date(eda.vars$weathertime), "%Y-%m-%d")
weather.md <- format(as.Date(eda.vars$weathertime), "%m-%d")

train$weather.ymd <- weather.ymd
train$weather.md <- weather.md
train$weather.mon <- eda.vars$weather.mon


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
  
#just do with missing points, keep index
temp.new <- temp.impute(temp)


  
  
  ggplot(eda.vars, aes(Temperature.F., fill = Severity)) +
    geom_histogram(alpha = 0.5, position="identity")+ggtitle("Temperature") +
    xlab("Temperature") + ylab("Frequency") 
  
  ggplot(eda.vars, aes(Temperature.F., fill = Severity)) +
    geom_density(alpha = 0.5, position="identity")+ggtitle("Temperature") +
    xlab("Temperature") + ylab("Frequency") 
  
  ggplot(eda.vars, aes(x= Severity, y=Temperature.F., fill = Severity)) +
    geom_boxplot()+ggtitle("Temperature")
  
  ggplot(eda.vars, aes(x= sevcat, y=Temperature.F., fill = sevcat)) +
    geom_boxplot()+ggtitle("Temperature")
  
  #wind chill factor
summary(eda.vars$Wind_Chill.F.)
sum(is.na(eda.vars$Wind_Chill.F.))
sum(is.na(eda.vars$Wind_Chill.F.))/nrow(eda.vars)
#44% missing 

#humidity

summary(eda.vars$Humidity...)
sum(is.na(eda.vars$Humidity...))/nrow(eda.vars)


ggplot(eda.vars, aes(Humidity..., fill = Severity)) +
  geom_density(alpha = 0.5, position="identity")+ggtitle("Humidity") +
  xlab("Humidity") + ylab("Frequency")

ggplot(eda.vars, aes(Humidity..., fill = sevcat)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Humidity") +
  xlab("Humidity") + ylab("Frequency")

ggplot(eda.vars, aes(x= sevcat, y=Humidity..., fill = sevcat)) +
  geom_boxplot()+ggtitle("Humidity")

#"Pressure.in.

summary(eda.vars$Pressure.in.)
sum(is.na(eda.vars$Pressure.in.))/nrow(eda.vars)

#visitibility.mi

Visibility.mi.
summary(eda.vars$Visibility.mi.)
sum(is.na(eda.vars$Visibility.mi.))/nrow(eda.vars)

summary(eda.vars$Wind_Direction)
sum(is.na(eda.vars$Wind_Direction))/nrow(eda.vars) 

#need cleaner variable, some are calm or variable as opposed to directions?

eda.vars$wind.d <- factor()

#probably need to impute with majority rule
wind_dir.impute <- function(a){
  impute <- a
  for(i in 1:length(a)){
    if(is.na(a[i]) == TRUE){
      impute[i] <- mean(train$Temperature.F.[train$County==train$County[i] & train$weather.md == train$weather.md[i]], na.rm = TRUE)
    }
  }
  return(impute)  
}



Wind_Speed.mph

summary(eda.vars$Wind_Speed.mph)
sum(is.na(eda.vars$Wind_Speed.mph))/nrow(eda.vars) 


Precipitation.in.
summary(eda.vars$Precipitation.in.)
sum(is.na(eda.vars$Precipitation.in.))/nrow(eda.vars) 

ggplot(eda.vars, aes(Precipitation.in., fill = Severity)) +
  geom_histogram(alpha = 0.5, position="identity")+ggtitle("Precipitation") +
  xlab("in") + ylab("Frequency")

Weather_Condition
summary(eda.vars$Weather_Condition)
levels(eda.vars$Weather_Condition)
69146/nrow(eda.vars) 
