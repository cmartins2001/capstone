cd "C:\Users\cmart\OneDrive - Bentley University\EC 483"

log using prel__gravity_model.out, text append
use "C:\Users\cmart\OneDrive - Bentley University\EC 483\stata\gravity_data_final.dta", clear

/// dealing_country variable cleanup:
replace dealing_country = lower(dealing_country)
gen dealing_country_final = trim(dealing_country)
keep if dealing_country_final == "england" | dealing_country_final == "spain" | dealing_country_final == "germany" | dealing_country_final == "italy" | dealing_country_final == "france" | dealing_country_final == "portugal" | dealing_country_final == "netherlands" | dealing_country_final == "russia" | dealing_country_final == "belgium" | dealing_country_final == "scotland"

/// creating a buying_country variable:
gen buying_country = ""
replace buying_country = "england" if buying_league == "PL"
replace buying_country = "spain" if buying_league == "LL"
replace buying_country = "germany" if buying_league == "BL"
replace buying_country = "italy" if buying_league == "SA"
replace buying_country = "france" if buying_league == "L1"
replace buying_country = "portugal" if buying_league == "LP"
replace buying_country = "netherlands" if buying_league == "EV"
replace buying_country = "russia" if buying_league == "Pliga_Rus"
replace buying_country = "belgium" if buying_league == "JPL"
replace buying_country = "scotland" if buying_league == "SP"


/// create a winter/summer dummy variable:
gen season_final = trim(window)
gen summer_transfer = cond(season_final == "summer", 1, 0)


///creating a distance variable:
gen distance = acos(sin(Buying_Lat*(_pi/180))*sin(Dealing_Lat*(_pi/180)) + cos(Buying_Lat*(_pi/180))*cos(Dealing_Lat*(_pi/180))*cos((Dealing_Lng - Buying_Lng)*(_pi/180)))*6371

/// fixing france-france distance issue:
replace distance = 0 if distance == .

/// creating common language dummy:
gen lang_overlap = 0

replace lang_overlap = 1 if (buying_country=="england" & dealing_country_final=="scotland") | (buying_country=="scotland" & dealing_country_final=="england") | (buying_country=="england"& dealing_country_final=="england") | (buying_country=="scotland" & dealing_country_final=="scotland") | (buying_country=="france" & dealing_country_final=="belgium") | (buying_country=="belgium" & dealing_country_final=="france") | (buying_country=="france" & dealing_country_final=="france") | (buying_country=="netherlands" & dealing_country_final=="belgium") | (buying_country=="belgium" & dealing_country_final=="netherlands") | (buying_country=="netherlands" & dealing_country_final=="netherlands") | (buying_country=="germany" & dealing_country_final=="belgium") | (buying_country=="belgium" & dealing_country_final=="germany") | (buying_country=="germany" & dealing_country_final=="germany") | (buying_country=="belgium" & dealing_country_final=="belgium") | (buying_country=="portugal" & dealing_country_final=="portugal") | (buying_country=="portugal" & dealing_country_final=="spain") | (buying_country=="spain" & dealing_country_final=="portugal") | (buying_country=="italy" & dealing_country_final=="italy") | (buying_country=="spain" & dealing_country_final=="spain") | (buying_country=="russia" & dealing_country_final=="russia")

/// generating a common border dummy:
gen contig = 0

replace contig = 1 if (buying_country=="france"& dealing_country_final=="france") | (buying_country=="spain"& dealing_country_final=="spain") | (buying_country=="italy"& dealing_country_final=="italy") | (buying_country=="england"& dealing_country_final=="england") | (buying_country=="scotland" & dealing_country_final=="scotland") | (buying_country=="germany" & dealing_country_final=="germany") | (buying_country=="belgium" & dealing_country_final=="belgium") | (buying_country=="netherlands" & dealing_country_final=="netherlands") | (buying_country=="portugal" & dealing_country_final=="portugal") | (buying_country=="russia" & dealing_country_final=="russia") | (buying_country=="france"& dealing_country_final=="spain") | (buying_country=="spain"& dealing_country_final=="france") | (buying_country=="france"& dealing_country_final=="italy") | (buying_country=="italy"& dealing_country_final=="france") | (buying_country=="france"& dealing_country_final=="germany") | (buying_country=="germany"& dealing_country_final=="france") | (buying_country=="portugal" & dealing_country_final=="spain") | (buying_country=="spain" & dealing_country_final=="portugal") | (buying_country=="netherlands" & dealing_country_final=="france") | (buying_country=="france" & dealing_country_final=="netherlands") | (buying_country=="netherlands" & dealing_country_final=="germany") | (buying_country=="germany" & dealing_country_final=="netherlands") | (buying_country=="netherlands" & dealing_country_final=="belgium") | (buying_country=="belgium" & dealing_country_final=="netherlands")

/// create a buying_size * dealing_size variable and taking natural log:
gen size_product = buying_league_size * dealing_league_size
gen ln_size_product = ln(size_product)

/// create a variable of abs value (fee - mkt value) and take LN:
gen mkt_value_dist = fee - market_value
gen mkt_value_minus_fee = abs(fee - market_value)
gen log_mkt_value_minus_fee = ln(mkt_value_minus_fee)

/// calculate total transfer fees paid and received by each league and dealing country, group by year, transfer in/out, position, and summer/winter window:
collapse (sum) fee, by(buying_league dealing_country_final year summer_transfer movement_binary GK DEF MID ATT distance brent_euro ln_size_product lang_overlap contig)

/// calculate bilateral trade volumes:
bysort buying_league dealing_country_final: gen trade_flow = sum(movement_binary * fee) + sum((1-movement_binary) * fee)

/// create log(trade_flow) variables:
gen ln_trade_flow = ln(trade_flow)
gen trade_flow_invh = ln(trade_flow + sqrt((trade_flow^2) + 1))


/// trade flow descriptives:
histogram trade_flow_invh, width(0.4) normal fraction
summarize trade_flow_invh, detail
histogram size_product
histogram ln_size_product

// summarize continuous variables:
summarize fee, detail
summarize trade_flow, detail
summarize distance, detail
summarize buying_league_size, detail
histogram ln_size_product, width(0.4) normal fraction
summarize dealing_league_size, detail

/// summarize dummies:
summarize lang_overlap, detail
summarize contig, detail
summarize movement_binary, detail
summarize GK, detail
summarize DEF, detail
summarize MID, detail
summarize ATT, detail
summarize summer_transfer, detail
tabulate summer_transfer

/// naive OLS models:
regress trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro

/// robustness checks:
estat hettest, iid
regress trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro, robust
pwcorr ln_size_product distance brent_euro
estat vif

/// FE models:

* buying_league FE model:
encode buying_league, generate(buying_league_code)
regress trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro i.buying_league_code
testparm i.buying_league_code
* evidence of buying_league level heterogeneity

* time-level FE model:
regress trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro i.year
testparm i.year
* evidence of time level heterogeneity

* TWFE model, using buying_league and year FE's:
xtreg trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro i.year, fe i(buying_league_code)
estimates store twfe_model
* no change in significance of variables, rho = 0.21 for league-level FE
* test for heteroskedasticity and get robust SE's:
hettest, twoway
xtreg trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro i.year, fe i(buying_league_code) cluster(buying_league_code)

 
/// RE model:
xtreg trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro i.year, re theta

/// HAUSMAN TEST for RE vs FE:
quietly xtreg trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro i.year, fe i(buying_league_code)
est store fe

quietly xtreg trade_flow_invh ln_size_product distance lang_overlap contig GK DEF MID ATT summer_transfer brent_euro i.year, re theta
est store re

hausman fe re, sigmamore
* TWFE is the better model


/// VISUALS post-estimation:
sysuse auto, clear
graph heat trade_flow_invh ln_size_product, row(foreign) col(rep78)
* scatter for the TWFE model:
twoway scatter trade_flow_invh ln_size_product || lfit trade_flow_invh ln_size_product
predict trade_flow_invh_hat if e(sample)
gen twfe_resid = trade_flow_invh - trade_flow_invh_hat
gen twfe_resid1 = trade_flow - twfe_trade_flow_hat
scatter twfe_resid trade_flow_invh_hat
gen size_product = exp(ln_size_product)
gen twfe_trade_flow_hat = exp(trade_flow_invh_hat)
twoway scatter twfe_resid1 twfe_trade_flow_hat

/// other model: regress mkt value diff variable against gravity regressors:
regress mkt_value_minus_fee distance ln_size_product GK DEF MID ATT lang_overlap contig brent_euro

