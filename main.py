import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import folium
from streamlit.components.v1 import html
import geopy 
from geopy.geocoders import Nominatim
import time
from sklearn.linear_model import LinearRegression
import numpy as np
from PIL import Image
warnings.filterwarnings('ignore')


# Initialize geocoder
geolocator = Nominatim(user_agent="geoapiExercises")

# Static coordinates for testing
def static_lat_lon(location):
    example_data = {
        'Delhi': (28.6139, 77.2090),
        'Maharashtra': (19.6633, 75.3302),
        'Karnataka': (15.3173, 75.7139),
        'Tamil Nadu': (11.1271, 78.6569),
        'Uttar Pradesh': (26.8467, 80.9462),
        'West Bengal': (22.9868, 87.8550),
        'Rajasthan': (27.0238, 74.2176),
        'Gujarat': (22.2587, 71.1924),
        'Kerala': (10.8505, 76.2711),
        'Bihar': (25.0961, 85.3131),
        'Assam': (26.2006, 92.9376),
        'Punjab': (31.1471, 75.3412),
        'Haryana': (29.0588, 76.0856),
        'Jharkhand': (23.6102, 85.2799),
        'Odisha': (20.9517, 85.0985),
        'Chhattisgarh': (21.2787, 81.8661),
        'Himachal Pradesh': (31.1048, 77.1734),
        'Uttarakhand': (30.0668, 79.0193),
        'Sikkim': (27.5330, 88.5126),
        'Arunachal Pradesh': (27.0984, 93.6167),
        'Meghalaya': (25.4670, 91.3662),
        'Nagaland': (26.1584, 94.5624),
        'Manipur': (24.6637, 93.9063),
        'Tripura': (23.8364, 91.2791),
        'Andaman and Nicobar Islands': (11.7401, 92.6586),
        'Dadra and Nagar Haveli': (20.1809, 73.0169),
        'Daman and Diu': (20.4250, 72.8238),
        'Lakshadweep': (10.5626, 72.6370),
        'Puducherry': (11.9416, 79.8083)
    }
    return example_data.get(location, (None, None))

logo = "logo.jpeg"
st.set_page_config(page_title="India EV Market", page_icon=logo, layout="wide")

col1, col2 = st.columns([1, 4])

with col1:
    st.image(logo, width=120)

with col2:
    st.markdown(
        """
        <style>
        .title-container {
            display: flex;
            align-items: center;
        }
        .title-container h1 {
            font-size: 4.5rem; /* Increase font size */
            margin-left: 0.0rem; /* Adjust spacing between logo and title */
        }
        </style>
        <div class="title-container">
            <h1>India EV Market</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
st.write('This dashboard presents key insights for the Indian Electric Vehicle Market.')



@st.cache_data
def load_data():
    ev_makers_df = pd.read_excel("cleaned_EV_Maker_by_Place.xlsx")
    ev_cat_df = pd.read_excel("cleaned_EV_Cat.xlsx")
    ev_sales_df = pd.read_excel("cleaned_EV_Sales.xlsx")
    operational_pc_df = pd.read_excel("cleaned_Operational.xlsx")
    vehicle_class_df = pd.read_excel("cleaned_vehice_class.xlsx")
    requirement_vs_having = pd.read_excel("requirement_vs_having.xlsx")

    return ev_makers_df, ev_cat_df, ev_sales_df, operational_pc_df, vehicle_class_df, requirement_vs_having

ev_makers_df, ev_cat_df, ev_sales_df, operational_pc_df, vehicle_class_df, requirement_vs_having = load_data()

# Add latitude and longitude using static coordinates
if 'latitude' not in ev_makers_df.columns or 'longitude' not in ev_makers_df.columns:
    #st.write("Adding Static Coordinates for EV Makers...")
    ev_makers_df[['latitude', 'longitude']] = ev_makers_df['State'].apply(lambda x: static_lat_lon(x) if pd.notnull(x) else (None, None)).apply(pd.Series)
    ev_makers_df.dropna(subset=['latitude', 'longitude'], inplace=True)  # Drop rows with missing coordinates
    time.sleep(1)  # Adding a delay to avoid potential issues

if 'latitude' not in operational_pc_df.columns or 'longitude' not in operational_pc_df.columns:
    #st.write("Adding Static Coordinates for Public Charging Stations...")
    operational_pc_df[['latitude', 'longitude']] = operational_pc_df['State'].apply(lambda x: static_lat_lon(x) if pd.notnull(x) else (None, None)).apply(pd.Series)
    operational_pc_df.dropna(subset=['latitude', 'longitude'], inplace=True)  # Drop rows with missing coordinates
    time.sleep(1) 

# Custom CSS for font and colors
st.markdown(
    """
    <style>
    /* Change overall font and background color */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f5f5f5;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ff6347;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stSelectbox, .stMultiselect, .stSlider {
        font-size: 18px;
        color: #1f77b4;
    }
    .stCheckbox label {
        font-size: 18px;
        color: #1f77b4;
    }
    .stTable {
        background-color: #ff6347;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Prediction function for vehicle categories
def predict_ev_sales(year, category):
    # Filter data based on category
    if category != 'All':
        filtered_sales = ev_sales_df[ev_sales_df['Cat'] == category]
    else:
        filtered_sales = ev_sales_df
    
    # Initialize lists to store predictions
    predictions = []

    for _, row in filtered_sales.iterrows():
        # Extract historical sales data
        X = np.array(range(2015, 2025)).reshape(-1, 1)  # Years from 2015 to 2024
        y = row[2:].values  # Sales data from 2015 to 2024

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict sales for the specified future year
        future_year = np.array([[year]])
        predicted_sales = model.predict(future_year)

        # Append prediction to the list
        predictions.append(predicted_sales[0])
    
    # Return the mean prediction if multiple categories or makers are present
    return np.mean(predictions)

def display_vehicle_requirements_and_production(requirement_vs_having):
    st.title('Vehicle Requirements and EV Production in India')

    All_Category = ['TWO WHEELER(NT)', 'LIGHT MOTOR VEHICLE', 'THREE WHEELER(T)',
                    'LIGHT GOODS VEHICLE', 'HEAVY GOODS VEHICLE',
                    'LIGHT PASSENGER VEHICLE', 'OTHER THAN MENTIONED ABOVE',
                    'MEDIUM GOODS VEHICLE', 'HEAVY GOODS VEHICLE',
                    'THREE WHEELER(NT)', 'MEDIUM PASSENGER VEHICLE',
                    'MEDIUM MOTOR VEHICLE', 'TWO WHEELER(T)']

    top_5_categories = ['TWO WHEELER(NT)', 'LIGHT MOTOR VEHICLE', 'THREE WHEELER(T)',
                        'LIGHT GOODS VEHICLE', 'HEAVY GOODS VEHICLE']

    # Select vehicle categories
   

    # Multiselect option with "Select All" functionality
    all_categories = requirement_vs_having['Vehicle Class'].tolist()
    all_categories.append('Select All')  # Add the "Select All" option

    selected_categories = st.multiselect('Select Vehicle Categories', all_categories)

    
    # Button to generate plot
    if st.button("Total Number of Vehicles in India"):
        
        # Filter the dataframe based on selected categories
        filtered_df = requirement_vs_having[requirement_vs_having['Vehicle Class'].isin(selected_categories)]
        
        # Plotting
        width = 0.35
        length = filtered_df['Vehicle Class']
        indices = np.arange(len(length))

        plt.figure(figsize=(10, 9))
        plt.bar(indices - width, filtered_df['Total Registrations'], width, label='Overall Requirement in the Country')
        plt.bar(indices, filtered_df['Total Across Years'], width, label='EV Category')

        plt.title('Total number of vehicles produced in India')
        plt.xlabel('Vehicle Class')
        plt.ylabel('Requirement and Production')
        plt.xticks(indices, length, rotation=90)
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)
# Sidebar Navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select one:', 
                           ['Dashboard', 
                            'EV Sales Analysis',
                            'EV Makers by Place',  
                            'Public Charging Stations', 
                            'Vehicle Class Registration'])

# Dashboard as the default homepage
if options == 'Dashboard':
    st.header("EV Market Overview")

    # Total EV Sales so far 
    total_sales = ev_sales_df.iloc[:, 2:].sum().sum()

    # Company that sold the most number of EVs
    top_company = ev_sales_df.groupby('Maker').sum().iloc[:, 1:].sum(axis=1).idxmax()

    # State with the most number of EVs
    top_state = ev_makers_df['State'].value_counts().idxmax()

    # Most sold EV category
    most_sold_category = ev_sales_df.groupby('Cat').sum().iloc[:, 1:].sum(axis=1).idxmax()

    first_column, second_column, third_column, fourth_column = st.columns(4)
    with first_column:
        st.markdown(
            f"""
            <h3 style='color: #ff6347;'>Total EV Sales:</h3>
            <h3 style='color: #ffffff;'>{total_sales:,}</h3>
            """, unsafe_allow_html=True)

    with second_column:
        st.markdown(
            f"""
            <h3 style='color: #ff6347;'>Top EV Company:</h3>
            <h3 style='color: #ffffff;'>{top_company}</h3>
            """, unsafe_allow_html=True)

    with third_column:
        st.markdown(
            f"""
            <h3 style='color: #ff6347;'>State with Most EVs:</h3>
            <h3 style='color: #ffffff;'>{top_state}</h3>
            """, unsafe_allow_html=True)

    with fourth_column:
        st.markdown(
            f"""
            <h3 style='color: #ff6347;'>Most Sold EV Category:</h3>
            <h3 style='color: #ffffff;'>{most_sold_category}</h3>
            """, unsafe_allow_html=True)

    # Future Sales Prediction (Placeholder - Example)
    st.header("Future Sales Prediction")
    year = st.slider('Select Year', 2025, 2030, 2025)
    # Define the category selection for prediction
    category = st.selectbox('Select Vehicle Category for Prediction', ['All'] + list(ev_sales_df['Cat'].unique()))

    # Make prediction
    predicted_sales = predict_ev_sales(year, category)

    # Display predicted sales
    st.write(f"Predicted EV sales for {category} in {year} will be approximately {predicted_sales:.0f} units.")

    # Display Vehicle Requirements and Production in India
    display_vehicle_requirements_and_production(requirement_vs_having)
    
if 'EV Sales Analysis' in options:
    st.header('EV Sales by Category and Year')

    st.subheader('Sales by Manufacturer')
    maker_filter = st.selectbox('Select a Manufacturer', ev_sales_df['Maker'].unique())
    filtered_sales = ev_sales_df[ev_sales_df['Maker'] == maker_filter]
    st.bar_chart(filtered_sales.set_index('Cat').T)

    year = st.slider('Select Year', 2015, 2024, 2021)
    filtered_yearly_sales = ev_sales_df[['Cat', 'Maker', (year)]]
    st.table(filtered_yearly_sales[filtered_yearly_sales['Maker'] == maker_filter])

    # Instruction text
    st.write("Click the button below to visualize the trends in EV sales over the years.")
    
    # Button to trigger the plot
    if st.button("Show EV Sales Trends"):
        
        Manufactured_ev_yearwise=ev_cat_df[['FOUR WHEELER (INVALID CARRIAGE)', 'HEAVY GOODS VEHICLE',
        'HEAVY MOTOR VEHICLE', 'HEAVY PASSENGER VEHICLE', 'LIGHT GOODS VEHICLE',
        'LIGHT MOTOR VEHICLE', 'LIGHT PASSENGER VEHICLE',
        'MEDIUM GOODS VEHICLE', 'MEDIUM PASSENGER VEHICLE',
        'MEDIUM MOTOR VEHICLE', 'OTHER THAN MENTIONED ABOVE',
        'THREE WHEELER(NT)', 'TWO WHEELER (INVALID CARRIAGE)',
        'THREE WHEELER(T)', 'TWO WHEELER(NT)', 'TWO WHEELER(T)','Year']].groupby("Year").sum().reset_index()

        # Prepare the data
        columns = ['TWO WHEELER(NT)', 'THREE WHEELER(T)', 'LIGHT MOTOR VEHICLE',
            'LIGHT PASSENGER VEHICLE', 'TWO WHEELER(T)', 'LIGHT GOODS VEHICLE',
            'HEAVY PASSENGER VEHICLE', 'OTHER THAN MENTIONED ABOVE',
            'THREE WHEELER(NT)', 'MEDIUM PASSENGER VEHICLE']

        Prime_years_top_vehicles=Manufactured_ev_yearwise[Manufactured_ev_yearwise["Year"]>=2010][['Year','TWO WHEELER(NT)',
            'THREE WHEELER(T)', 'LIGHT MOTOR VEHICLE',
        'LIGHT PASSENGER VEHICLE', 'TWO WHEELER(T)', 'LIGHT GOODS VEHICLE',
        'HEAVY PASSENGER VEHICLE', 'OTHER THAN MENTIONED ABOVE',
        'THREE WHEELER(NT)', 'MEDIUM PASSENGER VEHICLE']]

        # Step 2: Set 'Year' as the index
        Prime_years_top_vehicles.set_index("Year", inplace=True)

        #Step 3: Create the line chart using Matplotlib
        plt.figure(figsize=(10, 6))  # Set the figure size
        for column in columns:
            plt.plot(Prime_years_top_vehicles.index, Prime_years_top_vehicles[column], label=column)

        # Adding labels, legend, and grid
        plt.xlabel("Year")
        plt.ylabel("Number of Vehicles Manufactured")
        plt.title("Yearly Manufactured EVs by Category (2010 and Later)")
        plt.legend(title="Vehicle Categories")
        plt.grid(True)

        #Step 4: Display the plot in Streamlit
        st.pyplot(plt)

        #Step 5: Add headers and additional text
        st.header("Yearly Sales of  EVs by Category (2010 and Later)")
        st.write("The line chart shows the trends that demonstrate the evolution of the EV market in India since 2015.")
    st.title("Description")
    st.write("This section of the dashboard provides an analysis of EV sales by category and year. Users can filter sales data by selecting a manufacturer and viewing the corresponding sales breakdown in a bar chart. Additionally, a slider allows users to select a specific year to see the sales details for that year. By clicking a button, users can visualize trends in EV manufacturing across various vehicle categories over the years.")
if 'EV Makers by Place' in options:
    
    # Function to plot an interactive bar chart using Plotly
    def plot_interactive_bar_chart(data, column, title, x_label, y_label):
        fig = px.bar(
            data,
            x=column,
            y='Unique EV Makers',
            title=title,
            labels={column: x_label, 'Unique EV Makers': y_label},
            hover_name=column
        )
        st.plotly_chart(fig)

    # Create a function for the selection and plotting process to avoid redundancy
    def render_plot(section_name, column_name):
        # Group the data by the given column and count unique 'EV Maker'
        grouped_data = ev_makers_df.groupby(column_name)['EV Maker'].nunique().reset_index()
        grouped_data.columns = [column_name, 'Unique EV Makers']
        st.header("Setect locations to see No. of EV Makers")
        # Add a "Select All" checkbox
        select_all = st.checkbox(f'Select All {section_name}s')

        # Create a multiselect widget with all options selected if "Select All" is checked
        if select_all:
            selected_options = st.multiselect(f'Select {section_name}s', grouped_data[column_name].unique(), grouped_data[column_name].unique())
        else:
            selected_options = st.multiselect(f'Select {section_name}s', grouped_data[column_name].unique())

        # Filter the data based on selected options and create the chart
        if selected_options:
            filtered_data = grouped_data[grouped_data[column_name].isin(selected_options)]
            plot_interactive_bar_chart(
                filtered_data,
                column_name,
                f'Unique EV Makers by {section_name}',
                section_name,
                'Number of Unique EV Makers'
            )

    # Section for EV Makers by State and Place
    st.header("EV Makers Data")
    st.subheader("States")
    render_plot('State', 'State')

    st.subheader("Places")
    render_plot('Place', 'Place')


    st.header('EV Makers by Place')
    st.map(ev_makers_df[['latitude', 'longitude']])

    

    maker_filter = st.selectbox('Select an EV Maker', ev_makers_df['EV Maker'].unique())
    filtered_map_data = ev_makers_df[ev_makers_df['EV Maker'] == maker_filter]
    st.map(filtered_map_data[['latitude', 'longitude']])

    st.title("Description")
    st.write("This section provides an insight called 'Place by Makers' which helps identify the number of EV makers in specific states and locations. The interactive bar graph above shows the number of makers by state, and you can explore it further by downloading, zooming, or interacting with the graph. Below the bar graph, the map displays the locations of EV makers. You can select an EV maker from the list below the map to see where they are located.")

 
if 'Public Charging Stations' in options:
    st.header('Public Charging Stations in a State')

    # Filter data based on selected state
    state_filter = st.selectbox('Select State', operational_pc_df['State'].unique())
    filtered_pc_data = operational_pc_df[operational_pc_df['State'] == state_filter]

    # Display bar chart
    st.bar_chart(filtered_pc_data.set_index('State'))

    # Display the number of operational public charging stations
    st.subheader('Number of Operational Public Charging Stations')
    st.write(f"The number of operational public charging stations in {state_filter} is {filtered_pc_data['No. of Operational PCS'].values[0]}.")


if 'Vehicle Class Registration' in options:
    st.header('Vehicle Class Registration')

    st.subheader('Total Registrations by Vehicle Class')
    vehicle_class_filter = st.multiselect('Select Vehicle Classes', vehicle_class_df['Vehicle Class'].unique(), default=vehicle_class_df['Vehicle Class'].unique())
    filtered_vehicle_class_df = vehicle_class_df[vehicle_class_df['Vehicle Class'].isin(vehicle_class_filter)]
    st.bar_chart(filtered_vehicle_class_df.set_index('Vehicle Class'))


if not options:
    st.header("Welcome to the EV Market Dashboard")
    st.write("Please select one or more options from the sidebar to view detailed information")
