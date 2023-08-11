# connor martins
# ec 382 term project - flight delays time series analysis
# importing and cleaning the data into one dataframe:

library(data.table)
year_files <- list("C:\\Users\\cmart\\Documents\\EC 382\\project\\2009.csv",
                "C:\\Users\\cmart\\Documents\\EC 382\\project\\2010.csv",
                "C:\\Users\\cmart\\Documents\\EC 382\\project\\2011.csv",
                "C:\\Users\\cmart\\Documents\\EC 382\\project\\2012.csv",
                "C:\\Users\\cmart\\Documents\\EC 382\\project\\2013.csv",
                "C:\\Users\\cmart\\Documents\\EC 382\\project\\2014.csv",
                "C:\\Users\\cmart\\Documents\\EC 382\\project\\2016.csv",
                "C:\\Users\\cmart\\Documents\\EC 382\\project\\2017.csv",
                "C:\\Users\\cmart\\Documents\\EC 382\\project\\2018.csv")

sample_size <- 100


# Define a function to read and sample data from each file
read_and_sample <- function(file) {
  dt <- fread(file)
  dt[, .(DEP_DELAY = sample(DEP_DELAY, min(sample_size, .N), replace = TRUE)), by = .(FL_DATE)]
}

# Read and sample data from each file
all_results <- rbindlist(lapply(year_files, read_and_sample), use.names = TRUE, fill = TRUE)

# Convert the flight delay column to numeric
all_results[, DEP_DELAY := as.numeric(DEP_DELAY)]

# Compute the mean delay for each day
daily_mean <- all_results[, .(mean_delay = mean(DEP_DELAY, na.rm = TRUE)), by = .(FL_DATE)]


# daily_mean will serve as the main dataframe from which the TS variable will be generated:
delay <- ts(daily_mean$mean_delay, frequency = 365, start = c(2009, 1), end = c(2018, 365))

# plot the time series:
plot.ts(delay, main = "American Flight Delays Between 2009 and 2018", col = "lightblue",
        xlab = "Time", ylab = "Flight Delay in Minutes")
# time series decomposition due to evidence of seasonality:
delays_composed <- decompose(delay)
plot(delays_composed)
# plot the original series along with its trend component from the decomp:
ts.plot(delay, delays_composed$trend, gpars=list(xlab="Year", ylab="Flight Delay in Minutes", col = 3:4))
legend("topleft", legend = c("Flight Delays", "Flight Delays Trend Component"), col = 3:4, lty = 1)
# plotting the seasonally adjusted TS:
delays_seasonally_adj <- delay - delays_composed$seasonal
plot(delays_seasonally_adj, main = "Seasonally Adjusted Flight Delays", 
     col = "lightblue",
     xlab = "Time", ylab = "Flight Delay in Minutes")

# plotting the SMA of the TS:
library("TTR")
delays_SMA3 <- SMA(delay,n=3)
plot.ts(delays_SMA3, main = "Smooth Moving Average (Order 3) of Flight Delays",
     col = "lightblue")
delays_SMA9 <- SMA(delay,n=9)
plot.ts(delays_SMA9, main = "Smooth Moving Average (Order 9) of Flight Delays",
        col = "lightblue")
delays_SMA12 <- SMA(delay,n=12)
plot.ts(delays_SMA12, main = "Smooth Moving Average (Order 12) of Flight Delays",
        col = "lightblue")
# conducting an ADF test for a unit root:
library(tseries)
Log_Delay <- ts(log(delay)) # generates NaNs because you cannot take natural log of negative values
## Warning in log(delay): NaNs produced
FirstDiff_Delay <- diff(delay) # the first difference of the delay TS will be used for the ADF test
adf.test(FirstDiff_Delay)
adf_output <- adf.test(FirstDiff_Delay)
## Warning in adf.test(FirstDiff_Delay): p-value smaller than printed p-value
adf_output$critical[,"1%"]
## NULL
# adf.test() not returning a critical ADF statistic, so use urca: 
library(urca)
adf_urca_output <- ur.df(FirstDiff_Delay, type = "trend", selectlags = "AIC")
summary(adf_urca_output)
## 
## ############################################### 
## # Augmented Dickey-Fuller Test Unit Root Test # 
## ############################################### 
## 
## Test regression trend 
## 
## 
## Call:
## lm(formula = z.diff ~ z.lag.1 + 1 + tt + z.diff.lag)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -24.360  -4.239  -0.422   3.670  37.558 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -2.914e-03  2.229e-01  -0.013    0.990    
## z.lag.1     -1.726e+00  2.601e-02 -66.361   <2e-16 ***
## tt          -1.966e-07  1.058e-04  -0.002    0.999    
## z.diff.lag   2.860e-01  1.587e-02  18.017   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 6.727 on 3643 degrees of freedom
## Multiple R-squared:  0.6979, Adjusted R-squared:  0.6977 
## F-statistic:  2805 on 3 and 3643 DF,  p-value: < 2.2e-16
## 
## 
## Value of test-statistic is: -66.3614 1467.948 2201.921 
## 
## Critical values for test statistics: 
##       1pct  5pct 10pct
## tau3 -3.96 -3.41 -3.12
## phi2  6.09  4.68  4.03
## phi3  8.27  6.25  5.34
# plotting the ACF and PACF's of the time series' first difference:
acf(na.omit(FirstDiff_Delay), lag.max = 3000, main = "ACF for the First Difference of Flight Delays", col = "darkblue")
pacf(na.omit(FirstDiff_Delay), lag.max = 3000, main = "PACF for the First Difference of Flight Delays", col = "darkred")
# very granular, convert TS to monthly to see the effects on the ACF and PACF:
library(xts)
delay_xts <- xts(daily_mean$mean_delay, order.by = as.Date(daily_mean$FL_DATE))
monthly_delay <- apply.monthly(delay_xts, mean)
diff_monthly_delay <- diff(monthly_delay)
acf(na.omit(diff_monthly_delay), lag.max = 15, main = "ACF for the First Difference of Monthly Flight Delays", col = "darkblue")
pacf(na.omit(diff_monthly_delay), lag.max = 15, main = "PACF for the First Difference of Monthly Flight Delays", col = "darkred")
# AR(2) predictions, monthly:
ts.plot(monthly_delay, main = "Monthly Flight Delays, 5-Step-Ahead AR(2) Forecast")
AR_2 <- arima(monthly_delay, order = c(2,0,0))
predict_AR2 <- predict(AR_2)
AR2_forecast <- predict(AR_2, n.ahead = 5)$pred
AR2_forecast_se <- predict(AR_2, n.ahead = 5)$se
points(AR2_forecast, type = "l", col = 2)
points(AR2_forecast - 2*AR2_forecast_se, type = "l", col = 2, lty = 2)
points(AR2_forecast + 2*AR2_forecast_se, type = "l", col = 2, lty = 2)
# MA(2) predictions, monthly:
ts.plot(monthly_delay, main = "Monthly Flight Delays, 5-Step-Ahead MA(2) Forecast")
MA_2 <- arima(monthly_delay, order = c(0,0,2))
predict_MA2 <- predict(MA_2)
MA2_forecast <- predict(MA_2, n.ahead = 5)$pred
MA2_forecast_se <- predict(MA_2, n.ahead = 5)$se
points(MA2_forecast, type = "l", col = 2)
points(MA2_forecast - 2*MA2_forecast_se, type = "l", col = 2, lty = 2)
points(MA2_forecast + 2*MA2_forecast_se, type = "l", col = 2, lty = 2)
# ARMA(15,15) predictions, monthly:
ts.plot(monthly_delay, main = "Monthly Flight Delays, 5-Step-Ahead ARMA(2,2) Forecast")
ARMA_22 <- arima(monthly_delay, order = c(2,0,2))
predict_ARMA22 <- predict(ARMA_22)
ARMA22_forecast <- predict(ARMA_22, n.ahead = 5)$pred
ARMA22_forecast_se <- predict(ARMA_22, n.ahead = 5)$se
points(ARMA22_forecast, type = "l", col = 2)
points(ARMA22_forecast - 2*ARMA22_forecast_se, type = "l", col = 2, lty = 2)
points(ARMA22_forecast + 2*ARMA22_forecast_se, type = "l", col = 2, lty = 2)
AIC(AR_2)
## [1] 521.1414
AIC(MA_2)
## [1] 520.7793
AIC(ARMA_22)
## [1] 517.5043
BIC(AR_2)
## [1] 531.7952
BIC(MA_2)
## [1] 531.433
BIC(ARMA_22)
## [1] 533.4849
