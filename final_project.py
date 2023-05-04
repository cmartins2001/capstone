"""
Class: CS230--Section 2
Name: Connor Martins
Description: Final Project - Streamlit Web App for the U.S. Roller Coaster Dataset
I pledge that I have completed the programming assignment independently. 
I have not copied the code from a student or any source.
I have not given my code to any student. 
"""
import streamlit as st
import pandas as pd
import altair as alt
import base64
import folium
import math
from streamlit_folium import folium_static
# could not diagnose the reference error with streamlit_folium, it still works and renders the folium map
# in streamlit. see interactive_map() for more


# function that allows the user to select a year using a streamlit slider and filter the entire dataframe; will show
# only rows for coasters opened up to the selected year
def raw_data(dataframe, year):
    df1 = dataframe[dataframe['Year_Opened'] <= year]
    st.dataframe(df1)


# function that shows national average coaster metrics for all coasters up to a selected year, along with the percent
# change from the coasters up to the previous year. year selected using a streamlit slider
def get_metrics(dataframe, year):
    df1 = dataframe[dataframe['Year_Opened'] <= year]
    mean_top_speed = df1['Top_Speed'].mean()
    mean_drop = df1['Drop'].mean()
    mean_length = df1['Length'].mean()
    df2 = dataframe[dataframe['Year_Opened'] <= (year-1)]
    mean_top_speed_prev = df2['Top_Speed'].mean()
    mean_drop_prev = df2['Drop'].mean()
    mean_length_prev = df2['Length'].mean()
    top_speed_percent_change = ((mean_top_speed - mean_top_speed_prev) / mean_top_speed_prev) * 100
    drop_percent_change = ((mean_drop - mean_drop_prev) / mean_drop_prev) * 100
    length_percent_change = ((mean_length - mean_length_prev) / mean_length_prev) * 100
    return mean_top_speed, mean_drop, mean_length, top_speed_percent_change, drop_percent_change, length_percent_change


# function that renders a folium map of american roller coasters. hovering over each marker shows basic info about
# the coaster including location and specification info
def interactive_map(dataframe):
    coaster_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    for _, row in dataframe.iterrows():
        name = row['Coaster']
        speed = row['Top_Speed']
        height = row['Max_Height']
        length = row['Length']
        theme_park = row['Park']
        lat, long = row['Latitude'], row['Longitude']
        info = f"{name}:\n " \
               f"Top Speed: {speed}\n" \
               f"Coaster Height: {height} ft.\n" \
               f"Track Length: {length} ft.\n" \
               f"Theme Park: {theme_park}"
        folium.Marker([lat, long], popup=info).add_to(coaster_map)
    folium_static(coaster_map, width=700, height=550)


# function that takes the main dataframe and filters it based on a user-inputted year range and coaster specification
# and returns a histogram of the specification (nation-wide). histogram color varies based on chosen specification and
# the histogram itself is generated with the altair library
def spec_histogram(dataframe, year_range, spec):
    df1 = dataframe[['Year_Opened', 'Top_Speed', 'Drop', 'Length', 'Duration']]
    state_year_specs = df1[df1['Year_Opened'].between(year_range[0], year_range[1] + 1)]
    state_year_specs = state_year_specs.rename(columns={"Top_Speed": "Top Speed, MPH", "Drop": "Maximum Drop, ft",
                                                        "Length": "Length, ft",
                                                        "Duration": "Ride Duration, seconds"})
    color_dict = {'Top Speed, MPH': 'steelblue', 'Maximum Drop, ft': 'forestgreen', 'Length, ft': 'orange',
                  'Ride Duration, seconds': 'firebrick'}
    color = color_dict[spec]
    histogram = alt.Chart(state_year_specs).mark_bar(color=color).encode(
        x=alt.X(spec, bin=True),
        y=alt.Y('count()', title='Number of Roller Coasters')
    ).properties(
        title=f"Histogram of {spec}, Between the Years of {year_range[0]} and {year_range[1]}"
    )
    st.altair_chart(histogram, use_container_width=True)


# function that filters the main dataframe before or after a user-inputted year (slider) and state, and then returns the
# fastest or slowest roller coaster in that state before or after the year based on user choice
def get_avg_height(dataframe, state, before, year):
    df1 = dataframe[['Coaster', 'State', 'Year_Opened', 'Max_Height']]
    if before == 'Before':
        df1 = df1[df1['Year_Opened'] < year]
    else:
        df1 = df1[df1['Year_Opened'] > year]
    avg_height = df1[df1['State'] == state]['Max_Height'].mean()
    return avg_height


# function that filters the dataframe up to the user-inputted year and returns a pie chart showing the national
# proportions of roller coaster type up to that year (types are wooden or steel), pie chart from altair
def material_pie_chart(dataframe, year):
    df1 = dataframe[dataframe['Year_Opened'] <= year]
    df2 = df1[['Year_Opened', 'Coaster', 'Type']]
    state_types = df2.groupby('Type').agg({'Coaster': 'count'}).reset_index()
    state_types = state_types.rename(columns={"Coaster": "Coaster Count"})
    pie_chart = alt.Chart(state_types).mark_arc().encode(
        theta='Coaster Count:Q',
        color=alt.Color('Type:N', scale=alt.Scale(range=['steelblue', '#EC9756'])),
        tooltip=['Type', 'Coaster Count']
    )
    _left, mid, _right = st.columns(3)
    with mid:
        st.altair_chart(pie_chart)


# function that filters the dataframe by user-inputted state and returns the maximum top speed of a roller coaster in
# that state
def fastest_or_slowest(dataframe, fastest, state):
    df1 = dataframe[dataframe['State'] == state]
    df2 = df1[['Coaster', 'State', 'City', 'Year_Opened', 'Top_Speed']][df1['Top_Speed'].notnull()]
    if fastest == 'Fastest':
        speed = int(df2['Top_Speed'].max())
        _left, mid, _right = st.columns(3)
        with mid:
            st.subheader(
                f':blue[{speed}] MPH is the :green[{fastest.lower()}] top speed of a roller coaster in :blue[{state}].')
    elif fastest == 'Slowest':
        speed = int(df2['Top_Speed'].min())
        _left, mid, _right = st.columns(3)
        with mid:
            st.subheader(
                f':blue[{speed}] MPH is the :red[{fastest.lower()}] top speed of a roller coaster in :blue[{state}].')


# dataframe that generates an altair scatterplot with ride duration on the y-axis and track length on the x-axis,
# the color scale of the data points reflect the top_speed of the given coaster, and hovering over the points
# tells the user basic info about that coaster, including name, location, and a few specifications
def scatter_specs(dataframe):
    df1 = dataframe[['State', 'City', 'Coaster', 'Top_Speed', 'Length', 'Duration']]
    df1 = df1.rename(columns={"Coaster": "Coaster Name", "Top_Speed": "Top Speed, MPH",
                              "Length": "Length, ft", "Duration": "Duration, seconds"})
    scatter_plot = alt.Chart(df1).mark_point().encode(
        x='Length, ft',
        y='Duration, seconds',
        color=alt.Color('Top Speed, MPH', scale=alt.Scale(scheme='reds')),
        tooltip=['Coaster Name', 'State', 'City', 'Top Speed, MPH', 'Duration, seconds', 'Length, ft']
    ).properties(
        width=700,
        height=550
    )
    st.altair_chart(scatter_plot)


# main function that calls all the visualization and information-generating functions in the program
def main():
    main_df = pd.read_csv('RollerCoasters-Geo.csv')
    st.write('<div style="text-align:center"><h1>American Roller Coaster Data</h1></div>', unsafe_allow_html=True)
    # intro metrics about roller coasters:
    st.write('<div style="text-align:center"><h3>Coaster Specs at a Glance:</h3></div>', unsafe_allow_html=True)
    year1 = st.slider("Select a year:", min_value=(int(main_df['Year_Opened'].min() + 1)),
                      max_value=int(main_df['Year_Opened'].max()))
    mean_top_speed, mean_drop, mean_length, ts_perchng, drp_prchng, lngth_prchng = get_metrics(main_df, year1)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Average Top Speed (2016)', f"{mean_top_speed:.2f} MPH", f"{ts_perchng:.2f}%")
    with col2:
        st.metric('Average Maximum Drop', f"{mean_drop:.2f} FT", f"{drp_prchng:.2f}%")
    with col3:
        st.metric('Average Coaster Length', f"{mean_length:.2f} FT", f"{lngth_prchng:.2f}%")
    gif = open("coaster_gif.gif", "rb")
    contents = gif.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    gif .close()
    _left, mid, _right = st.columns(3)
    with mid:
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="coaster gif">',
            unsafe_allow_html=True,
        )
    st.write(f'\n\n\n')
    # start of functions output:
    st.divider()
    st.title("American Roller Coasters:")
    tab1, tab2 = st.tabs(["Map", "Raw Data"])
    tab1.subheader("Map of U.S. Roller Coasters")
    with tab1:
        interactive_map(main_df)
    with tab2:
        year_df = st.slider("Filter the dataset to before the year of your choice:", min_value=int(main_df['Year_Opened'].min()),
                            max_value=int(main_df['Year_Opened'].max()), key="key1")
        raw_data(main_df, year_df)
    st.divider()
    st.title("Histograms of Roller Coaster Specs for Different Time Periods:")
    year_range = st.radio("Choose a time period you'd like to know more about:", ((1915, 1950), (1951, 1995),
                                                                                  (1996, 2016)))
    coaster_spec = st.radio("Choose the roller coaster spec you want to know more "
                            "about:", ('Top Speed, MPH', 'Maximum Drop, ft', 'Length, ft', 'Ride Duration, seconds'))
    spec_histogram(main_df, year_range, coaster_spec)
    st.divider()
    st.title("Select a state and year and see what its average roller coaster height was before or after that year:")
    state = st.selectbox("Select a state:", main_df['State'].unique(), key="first_time")
    year = st.slider("Select a year:", min_value=int(main_df['Year_Opened'].min()),
                     max_value=int(main_df['Year_Opened'].max()))
    before = st.radio("Choose whether you'd like to see the average roller coaster height before or "
                      "after your selected year:", ('Before', 'After'))
    avg_height = get_avg_height(main_df, state, before, year)
    if math.isnan(avg_height):
        st.subheader(f"There were :red[no] roller coasters in :blue[{state}]{' before' if before else ' after'} {year}.")
    else:
        st.subheader(
            f"The average roller coaster height for :blue[{state}]{' before' if before else ' after'} {year} "
            f"is :green[{avg_height:.2f}] feet.")
    st.divider()
    st.title("Select a year and create a pie chart showing the percentage "
             "of American roller coasters by construction type:")
    year = st.slider("Select a year:", min_value=int(main_df['Year_Opened'].min()),
                     max_value=int(main_df['Year_Opened'].max()), key="second_time")
    material_pie_chart(main_df, year)
    st.divider()
    st.title("Select a state and see its fastest or slowest roller coaster:")
    state = st.selectbox("Select a state:", main_df['State'].unique(), key="third_time")
    fastest = st.radio(f"Choose whether you'd like to see the fastest or slowest top speed in "
                       f"{state}:", ('Fastest', 'Slowest'))
    fastest_or_slowest(main_df, fastest, state)
    st.divider()
    st.title("Speed, Track Length, and Ride Duration Scattered:")
    scatter_specs(main_df)
    st.divider()
    _left, mid, _right = st.columns(3)
    with mid:
        st.subheader("Thank You!")
    return main_df


if __name__ == '__main__':
    main()
