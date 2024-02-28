import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from haversine import haversine, Unit
import statsmodels.formula.api as smf
import statsmodels.api as sm
from linearmodels import RandomEffects
# use the above library to create a spinner for certain functions if you can figure that out


def haversine_dist(row):
    buying_coord = (row['Buying_Lat'], row['Buying_Lng'])
    dealing_coord = (row['Dealing_Lat'], row['Dealing_Lng'])
    return haversine(buying_coord, dealing_coord, unit=Unit.KILOMETERS)


# making an altair histogram of transfer fees and LN(transfer fees):
def histograms(dataframe):
    fee_hist = alt.Chart(dataframe).mark_bar().encode(
        x=alt.X('log_transfer_fee', bin=True),
        y='count()'
    )
    st.altair_chart(fee_hist, use_container_width=True)


def filtered_dataframe(dataframe, league, purchase):
    filtered_df = dataframe[(dataframe['League 1'] == league) & (dataframe['Purchase'] == purchase)]
    st.dataframe(filtered_df)


def gravity_dataframe(dataframe):
    st.subheader("Simple OLS Regressions:")
    df1 = dataframe.rename(
        columns={'year': 'Year', 'buying_league': 'League_1', 'dealing_country_final': 'Country 2',
                 'brent_euro': 'Brent', 'movement_binary': 'Purchase', 'summer_transfer': 'Summer', 'fee': 'Transfer Fee',
                 'distance': 'Distance', 'lang_overlap': 'Common_Language', 'contig': 'Common_Border',
                 'ln_size_product': 'Log_of_Sizes_Product', 'trade_flow': 'Trade_Volume', 'trade_flow_invh':
                     'Log_of_Trade_Volume'})
    scatter_plot1 = alt.Chart(df1).mark_point().encode(
        x=alt.X('Log_of_Sizes_Product', scale=alt.Scale(domain=[6.0, 21.0])),
        y='Log_of_Trade_Volume',
        color=alt.Color('Year:O', scale=alt.Scale(scheme='reds')),
        tooltip=['Year', 'League_1', 'Country 2', 'Trade_Volume']
    ).properties(
        width=700,
        height=550,
        title=f"Trade Volume vs. Product of League Sizes (Log Transformation)"
    )
    reg_line = scatter_plot1.transform_regression('Log_of_Sizes_Product',
                                                  'Log_of_Trade_Volume').mark_line(color='#329ea8')
    st.altair_chart(scatter_plot1 + reg_line)
    st.divider()
    st.subheader("Panel Data Analyses:")
    model_type = st.radio("Choose a panel data analysis method:", ('Naive Pooled OLS Model',
                                                                   'League-Level Fixed Effects Model',
                                                                   'Time-Level Fixed Effects Model',
                                                                   'Two-Way Fixed Effects (TWFE) Model',
                                                                   'Random Effects (RE) Model'))
    if st.button("Run Regressions"):
        # Show a spinner while the regressions are running
        with st.spinner("Running regressions..."):
            st.write(f"{model_type} Results:")
            if model_type == "Naive Pooled OLS Model":
                pooled_ols_model = smf.ols(
                    'Log_of_Trade_Volume ~ Log_of_Sizes_Product + Distance + Common_Language + Common_Border + GK + '
                    'DEF + MID + ATT + Summer + Brent', data=df1)
                results_pooled_ols_model = pooled_ols_model.fit()
                st.write(results_pooled_ols_model.summary())
                df1['Prediction'] = results_pooled_ols_model.predict()
                # Scatter plot of observations vs. predictions:
                scatter_plot = alt.Chart(df1).mark_point().encode(
                    x=alt.X('Prediction', scale=alt.Scale(domain=[0, 27])),
                    y=alt.Y('Log_of_Trade_Volume', scale=alt.Scale(domain=[0, 27])),
                    color=alt.Color('Year:O', scale=alt.Scale(scheme='blues')),
                    tooltip=['Log_of_Trade_Volume', 'Prediction']
                ).properties(
                    width=700,
                    height=550,
                    title=f"Observed Trade Volumes vs. {model_type} Predictions"
                )
                # Make a 45-degree reference line
                reference_line = alt.Chart(pd.DataFrame({'x': [0, 27], 'y': [0, 27]})).mark_line(
                    color='red',
                    strokeDash=[3, 3]
                ).encode(
                    x='x',
                    y='y'
                )
                st.divider()
                st.altair_chart(scatter_plot + reference_line)
            elif model_type == "League-Level Fixed Effects Model":
                formula = 'Log_of_Trade_Volume ~ Log_of_Sizes_Product + Distance + Common_Language + ' \
                          'Common_Border + GK + DEF + MID + ATT + Summer + Brent + C(League_1)'
                fe_model1 = sm.formula.ols(formula, data=df1)
                results_fe_model1 = fe_model1.fit()
                st.write(results_fe_model1.summary())
                df1['Prediction'] = results_fe_model1.predict()
                # Scatter plot of observations vs. predictions:
                scatter_plot = alt.Chart(df1).mark_point().encode(
                    x=alt.X('Prediction', scale=alt.Scale(domain=[0, 27])),
                    y=alt.Y('Log_of_Trade_Volume', scale=alt.Scale(domain=[0, 27])),
                    color=alt.Color('Year:O', scale=alt.Scale(scheme='greens')),
                    tooltip=['Log_of_Trade_Volume', 'Prediction']
                ).properties(
                    width=700,
                    height=550,
                    title=f"Observed Trade Volumes vs. {model_type} Predictions"
                )
                # Make a 45-degree reference line
                reference_line = alt.Chart(pd.DataFrame({'x': [0, 27], 'y': [0, 27]})).mark_line(
                    color='red',
                    strokeDash=[3, 3]
                ).encode(
                    x='x',
                    y='y'
                )
                st.divider()
                st.altair_chart(scatter_plot + reference_line)
            elif model_type == "Time-Level Fixed Effects Model":
                formula = 'Log_of_Trade_Volume ~ Log_of_Sizes_Product + Distance + Common_Language + ' \
                          'Common_Border + GK + DEF + MID + ATT + Summer + Brent + C(Year)'
                fe_model2 = sm.formula.ols(formula, data=df1)
                results_fe_model2 = fe_model2.fit()
                st.write(results_fe_model2.summary())
            elif model_type == "Two-Way Fixed Effects (TWFE) Model":
                formula = 'Log_of_Trade_Volume ~ Log_of_Sizes_Product + Distance + Common_Language + ' \
                          'Common_Border + GK + DEF + MID + ATT + Summer + Brent + C(Year) + C(League_1)'
                twfe_model = sm.formula.ols(formula, data=df1)
                results_twfe_model = twfe_model.fit()
                st.write(results_twfe_model.summary())
                df1['Prediction'] = results_twfe_model.predict()
                # Scatter plot of observations vs. predictions:
                scatter_plot = alt.Chart(df1).mark_point().encode(
                    x=alt.X('Prediction', scale=alt.Scale(domain=[0, 27])),
                    y=alt.Y('Log_of_Trade_Volume', scale=alt.Scale(domain=[0, 27])),
                    color=alt.Color('Year:O', scale=alt.Scale(scheme='blues')),
                    tooltip=['Log_of_Trade_Volume', 'Prediction']
                ).properties(
                    width=700,
                    height=550,
                    title=f"Observed Trade Volumes vs. {model_type} Predictions"
                )
                # Make a 45-degree reference line
                reference_line = alt.Chart(pd.DataFrame({'x': [0, 27], 'y': [0, 27]})).mark_line(
                    color='red',
                    strokeDash=[3, 3]
                ).encode(
                    x='x',
                    y='y'
                )
                st.divider()
                st.altair_chart(scatter_plot + reference_line)
                # Make a residual plot":
                residuals = results_twfe_model.resid
                fitted_values = results_twfe_model.fittedvalues
                data = pd.DataFrame({'Fitted Values': fitted_values, 'Residuals': residuals})
                scatter_plot_resids = alt.Chart(data).mark_circle().encode(
                    x='Fitted Values',
                    y='Residuals',
                    tooltip=['Fitted Values', 'Residuals']
                ).properties(
                    width=700,
                    height=550,
                    title=f'{model_type} Residual Plot'
                )
                # Add a horizontal reference line for the residual plot:
                reference_line1 = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[3, 3]).encode(
                    y='y')
                residual_plot = scatter_plot_resids + reference_line1
                st.altair_chart(residual_plot)
            elif model_type == "Random Effects (RE) Model":
                model = RandomEffects(df1['Log_of_Trade_Volume'], df1[['Log_of_Sizes_Product', 'Distance', 'Common_Language',
                                                       'Common_Border', 'GK', 'DEF', 'MID', 'ATT', 'Summer', 'Brent']])
                re_results = model.fit()
                st.write(re_results.summary)
                df1['Prediction'] = re_results.predict()
                # Scatter plot of observations vs. predictions:
                scatter_plot = alt.Chart(df1).mark_point().encode(
                    x=alt.X('Prediction', scale=alt.Scale(domain=[0, 27])),
                    y=alt.Y('Log_of_Trade_Volume', scale=alt.Scale(domain=[0, 27])),
                    color=alt.Color('Year:O', scale=alt.Scale(scheme='blues')),
                    tooltip=['Log_of_Trade_Volume', 'Prediction']
                ).properties(
                    width=700,
                    height=550,
                    title=f"Observed Trade Volumes vs. {model_type} Predictions"
                )
                # Make a 45-degree reference line
                reference_line = alt.Chart(pd.DataFrame({'x': [0, 27], 'y': [0, 27]})).mark_line(
                    color='red',
                    strokeDash=[3, 3]
                ).encode(
                    x='x',
                    y='y'
                )
                st.divider()
                st.altair_chart(scatter_plot + reference_line)

        # Clear the spinner once the regressions are finished
        st.spinner()


def transfer_trends(dataframe, league):
    transfer_sums = dataframe[dataframe['League 1'] == league]
    transfer_sums['Spending'] = np.where(transfer_sums['Purchase'] == 1, transfer_sums['Transfer Fee'], 0)
    transfer_sums['Income'] = np.where(transfer_sums['Purchase'] == 0, transfer_sums['Transfer Fee'], 0)
    transfer_sums['Year'] = pd.to_datetime(transfer_sums['Year'], format='%Y')
    transfer_sums = transfer_sums.groupby('Year').agg({'Spending': 'sum', 'Income': 'sum'}).reset_index()
    area_chart1 = alt.Chart(transfer_sums).transform_fold(
        ['Spending', 'Income'],
        as_=['Category', 'Value']
    ).mark_area(opacity=0.7).encode(
        x='Year:T',
        y=alt.Y('Value:Q', stack=None),
        color=alt.Color('Category:N', scale=alt.Scale(domain=['Spending', 'Income'], range=['#c75f5f', '#5fa6c7'])),
        tooltip=['Year:T', 'Value:Q', 'Category:N']
    ).properties(
        width=700,
        height=550,
        title=f"{league} Transfer Spending and Income Over Time"
    )
    st.altair_chart(area_chart1)


def run_ols(dataframe):
    # plot value difference variable against log_product_size
    df1 = dataframe[dataframe['Purchase'] == 1]
    scatter_plot1 = alt.Chart(df1).mark_point().encode(
        x=alt.X('log_size_product', scale=alt.Scale(domain=[6.0, 21.0])),
        y='value_diff',
        color=alt.Color('Year:O', scale=alt.Scale(scheme='blues')),
        tooltip=['Year', 'Club 1', 'Club 2', 'value_diff', 'Name']
    ).properties(
        width=700,
        height=550,
        title=f"Value-Transfer Difference (Euros) vs. League Size Product"
    )
    st.altair_chart(scatter_plot1)
    scatter_plot2 = alt.Chart(df1).mark_point().encode(
        x='Distance',
        y='value_diff',
        color=alt.Color('Year:O', scale=alt.Scale(scheme='greens')),
        tooltip=['Year', 'Club 1', 'Club 2', 'value_diff', 'Name', 'Distance']
    ).properties(
        width=700,
        height=550,
        title=f"Value-Transfer Difference (Euros) vs. Trade Distance (km)"
    )
    st.altair_chart(scatter_plot2)
    scatter_plot2 = alt.Chart(df1).mark_point().encode(
        x='Brent',
        y='value_diff',
        color=alt.Color('Year:O', scale=alt.Scale(scheme='purples')),
        tooltip=['Year', 'Club 1', 'Club 2', 'value_diff', 'Name', 'Brent']
    ).properties(
        width=700,
        height=550,
        title=f"Brent Crude Oil (Euros) vs. Trade Distance (km)"
    )
    st.altair_chart(scatter_plot2)


def main():
    # Making the main dataframe:
    df = pd.read_excel('gravity.xlsx')
    # Getting rid of unwanted columns and renaming the wanted columns:
    df1 = df[['year', 'name', 'buying_league', 'club', 'short_pos', 'market_value',
              'dealing_club', 'dealing_league', 'movement_binary', 'window', 'fee',
              'is_loan', 'brent_euro', 'buying_league_size', 'dealing_league_size',
              'Buying_Lat', 'Buying_Lng', 'Dealing_Lat', 'Dealing_Lng',
              'GK', 'DEF', 'MID', 'ATT']]
    df1 = df1.rename(
        columns={'year': 'Year', 'name': 'Name', 'buying_league': 'League 1', 'club': 'Club 1', 'short_pos': 'Position',
                 'market_value': 'Market Value', 'dealing_club': 'Club 2', 'dealing_league': 'League 2',
                 'movement_binary': 'Purchase', 'window': 'Window', 'fee': 'Transfer Fee',
                 'brent_euro': 'Brent',
                 'buying_league_size': 'League 1 Size', 'dealing_league_size': 'League 2 Size',
                 'Buying_Lat': 'Lat 1', 'Buying_Lng': 'Lng 1',
                 'Dealing_Lat': 'Lat 2', 'Dealing_Lng': 'Lng 2'})
    # Getting rid of loan deals and free transfers:
    df1 = df1[round(df1['is_loan'], 1) == 0.0]
    df1 = df1[df1['Transfer Fee'] != 0]
    # Taking natural logs of relevant variables:
    df1['log_transfer_fee'] = np.log(df1['Transfer Fee'])
    df1['log_size_product'] = np.log((df1['League 1 Size'] * df1['League 2 Size']))
    # Taking the difference between market value and transfer fee:
    df1['value_diff'] = (df1['Transfer Fee'] - df1['Market Value'])
    # Adding a distance variable by calling the haversine distance formula:
    df1['Distance'] = df1.apply(lambda row: haversine((row['Lat 1'], row['Lng 1']),
                                                      (row['Lat 2'], row['Lng 2']), unit=Unit.KILOMETERS),
                                axis=1)
    # Making an aggregate dataframe for the gravity modeling:
    gravity_df = pd.read_stata('gravity_agg.dta')
    # Streamlit outputs:
    st.title("Gravity Model of Trade for Elite European Soccer Players")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Histogram", "Raw Data", "Transfer Trends", "Scatterplots", "Gravity"])
    tab1.subheader("Histogram:")
    tab2.subheader("League Filtering:")
    tab3.subheader("Transfer Trends:")
    tab4.subheader("Scatterplots:")
    with tab1:
        # Transfer fee histogram in streamlit:
        st.write("Natural Log of Transfers (Free Transfers Omitted):")
        histograms(df1)
    with tab2:
        # Creating a filtered dataframe based on a league name and
        # printing it along with a graph of total spending/purchases:
        league_name = st.selectbox("Select a league:", df1['League 1'].unique(), key="first")
        purchase = st.radio("See player purchases (1) or sales (0): ", (0, 1), key="first_purchase")
        filtered_dataframe(df1, league_name, purchase)
    with tab3:
        league_name = st.selectbox("Select a league:", df1['League 1'].unique())
        transfer_trends(df1, league_name)
    with tab4:
        run_ols(df1)
    with tab5:
        gravity_dataframe(gravity_df)


if __name__ == '__main__':
    main()
