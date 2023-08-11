# import the year-level csv files from kaggle:
library(readxl)
X2009 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2009.csv")
X2010 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2010.csv")
X2011 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2011.csv")
X2012 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2012.csv")
X2013 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2013.csv")
X2014 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2014.csv")
X2015 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2015.csv")
X2016 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2016.csv")
X2017 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2017.csv")
X2018 <- read.csv("C:\\Users\\cmart\\Documents\\EC 382\\project\\2018.csv")

# create a list of all the data frames
dfs <- list(X2009[, c("FL_DATE", "DEP_DELAY")],
            X2010[, c("FL_DATE", "DEP_DELAY")],
            X2011[, c("FL_DATE", "DEP_DELAY")],
            X2012[, c("FL_DATE", "DEP_DELAY")],
            X2013[, c("FL_DATE", "DEP_DELAY")],
            X2014[, c("FL_DATE", "DEP_DELAY")],
            X2015[, c("FL_DATE", "DEP_DELAY")],
            X2016[, c("FL_DATE", "DEP_DELAY")],
            X2017[, c("FL_DATE", "DEP_DELAY")],
            X2018[, c("FL_DATE", "DEP_DELAY")])

# combine all data frames into a single data frame
combined_df <- do.call(rbind, dfs)

library(dplyr)
library(magrittr)

# Assuming that your original data frame is called combined_df
# Calculate the mean departure delay for each date
aggregate(DEP_DELAY ~ FL_DATE, combined_df, mean)



# Convert the result to a data frame with two columns: "Date" and "mean_delay"
averages_df <- as.data.frame(averages_df)


# averages_df <- combined_df %>% group_by(FL_DATE) %>% summarise(mean_delay = mean(DepartureDelay))

# Convert the result to a data frame with two columns: "Date" and "mean_delay"
averages_df <- as.data.frame(averages_df)



