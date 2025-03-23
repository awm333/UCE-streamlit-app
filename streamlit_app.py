import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

vehicles_df = pd.read_csv('limited_vehicles.csv')
grid_emissions_df = pd.read_csv('grid_emissions_forecast.csv')

def emissions_intersection_point(df_EV, df_ICEV):
    #Takes two dataframes, one for EV and one for ICEV, and returns the year and emissions at which the two intersect
    #Returns the year AFTER the intersction point
    #If no intersection point is found, returns (0,0)
    df_EV_comp = df_EV[['years_of_ownership', 'running_emissions']]
    df_EV_comp.rename(columns={'running_emissions': 'EV_running_emissions'}, inplace=True)

    df_ICEV_comp = df_ICEV[['years_of_ownership', 'running_emissions']]
    df_ICEV_comp.rename(columns={'running_emissions': 'ICEV_running_emissions'}, inplace=True)

    df_comp = pd.merge(df_EV_comp, df_ICEV_comp, on='years_of_ownership', how='outer')
    df_comp['diff'] = df_comp['EV_running_emissions'] - df_comp['ICEV_running_emissions']
    df_comp['diff_sign'] = np.sign(df_comp['diff'])
    df_comp['ly_diff_sign'] = df_comp['diff_sign'].shift(1)
    df_comp['intersection_point'] = np.where((df_comp['diff_sign'] != df_comp['ly_diff_sign']) & (df_comp['years_of_ownership'] > 1), 
                                            1, 
                                            0)
    #if there is an exact match, set the intersection point to 2
    df_comp['intersection_point'] = np.where((df_comp['diff_sign'] == 0),
                                            2,
                                            df_comp['intersection_point'])

    df_return = df_comp[df_comp['intersection_point'] > 0][['years_of_ownership', 'EV_running_emissions', 'ICEV_running_emissions']]
    #select row with max intersection point
    df_return = df_comp[df_comp['intersection_point'] == df_comp['intersection_point'].max()]
    try:
            return_yr = df_return['years_of_ownership'].values[0]
            return_EV_emissions = df_return['EV_running_emissions'].values[0]
            return_ICEV_emissions = df_return['ICEV_running_emissions'].values[0]
    except:
            return_yr = 0
            return_EV_emissions = 0
            return_ICEV_emissions = 0
    return_emissions = np.average([return_EV_emissions, return_ICEV_emissions])

    return_intersection_point = (return_yr, return_emissions)
    return return_intersection_point

def cost_intersection_point(df_EV, df_ICEV):
    #Takes two dataframes, one for EV and one for ICEV, and returns the year and emissions at which the two intersect
    #Returns the year AFTER the intersction point
    #If no intersection point is found, returns (0,0)
    df_EV_comp = df_EV[['years_of_ownership', 'running_cost_of_ownership']]
    df_EV_comp.rename(columns={'running_cost_of_ownership': 'EV_running_cost'}, inplace=True)

    df_ICEV_comp = df_ICEV[['years_of_ownership', 'running_cost_of_ownership']]
    df_ICEV_comp.rename(columns={'running_cost_of_ownership': 'ICEV_running_cost'}, inplace=True)

    df_comp = pd.merge(df_EV_comp, df_ICEV_comp, on='years_of_ownership', how='outer')
    df_comp['diff'] = df_comp['EV_running_cost'] - df_comp['ICEV_running_cost']
    df_comp['diff_sign'] = np.sign(df_comp['diff'])
    df_comp['ly_diff_sign'] = df_comp['diff_sign'].shift(1)
    df_comp['intersection_point'] = np.where((df_comp['diff_sign'] != df_comp['ly_diff_sign']) & (df_comp['years_of_ownership'] > 1), 
                                            1, 
                                            0)
    #if there is an exact match, set the intersection point to 2
    df_comp['intersection_point'] = np.where((df_comp['diff_sign'] == 0),
                                            2,
                                            df_comp['intersection_point'])

    df_return = df_comp[df_comp['intersection_point'] > 0][['years_of_ownership', 'EV_running_cost', 'ICEV_running_cost']]
    #select row with max intersection point
    df_return = df_comp[df_comp['intersection_point'] == df_comp['intersection_point'].max()]
    try:
            return_yr = df_return['years_of_ownership'].values[0]
            return_EV_cost = df_return['EV_running_cost'].values[0]
            return_ICEV_cost = df_return['ICEV_running_cost'].values[0]
    except:
            return_yr = 0
            return_EV_cost = 0
            return_ICEV_cost = 0
    return_cost = np.average([return_EV_cost, return_ICEV_cost])

    return_intersection_point = (return_yr, return_cost)
    return return_intersection_point

def plot_cars(model_1, model_2, gas_price, kwh_price, grid_emissions_option, miles_per_year):
    #fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 20))

    fig1 = plt.figure(figsize=(20, 10))
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure(figsize=(20, 10))
    ax2 = fig2.add_subplot(111)

    EV_model_df = vehicles_df[vehicles_df['model'] == model_1]
    ICEV_model_df = vehicles_df[vehicles_df['model'] == model_2]
    
    years_of_ownership_df = pd.DataFrame({'years_of_ownership': range(0,21)})
    EV_model_df = pd.merge(EV_model_df, years_of_ownership_df, how='cross')
    ICEV_model_df = pd.merge(ICEV_model_df, years_of_ownership_df, how='cross')

    grid_emissions_df['merge_year'] = grid_emissions_df['forecast_year'] - 2023
    EV_model_df = pd.merge(EV_model_df, grid_emissions_df, how='left', left_on=['years_of_ownership'], right_on=['merge_year'])
    ICEV_model_df = pd.merge(ICEV_model_df, grid_emissions_df, how='left', left_on=['years_of_ownership'], right_on=['merge_year'])

    #calculate running cost of ownership
    EV_model_df['running_cost_of_ownership'] = EV_model_df['MSRP'] + (1013 * 0.66 * EV_model_df['years_of_ownership']) + (miles_per_year * EV_model_df['combE'] / 100 * kwh_price * EV_model_df['years_of_ownership'])
    ICEV_model_df['running_cost_of_ownership'] = ICEV_model_df['MSRP'] + (1013 * ICEV_model_df['years_of_ownership']) + (miles_per_year / ICEV_model_df['comb08'] * gas_price * ICEV_model_df['years_of_ownership'])
    
    #calculate running emissions
    EV_model_df['ty_emissions'] = np.where(EV_model_df['years_of_ownership'] != 0,
                                           EV_model_df['emissions_tCO2MWh'] / 1000 * EV_model_df['combE'] / 100 * miles_per_year,
                                           0)
    EV_model_df['starting_emissions'] = np.where(EV_model_df['years_of_ownership'] == 0, 
                                                 EV_model_df['Manufacture_Emissions_tCO2'], 
                                                 0)
    if grid_emissions_option == 1:
        EV_model_df['ty_plus'] = EV_model_df['ty_emissions'] + EV_model_df['starting_emissions']
        EV_model_df['running_emissions'] = EV_model_df['ty_plus'].cumsum()
    elif grid_emissions_option == 2:
        EV_model_df['running_emissions'] = EV_model_df['Manufacture_Emissions_tCO2'] + ((0.532954 / 1000 * EV_model_df['combE'] / 100 * miles_per_year * EV_model_df['years_of_ownership']))
        #'combE' = kWh/100 miles, 0.532954 tCO2/MWh (from Logan)
    elif grid_emissions_option == 3:
        EV_model_df['running_emissions'] = EV_model_df['Manufacture_Emissions_tCO2'] + ((1.047798 / 1000 * EV_model_df['combE'] / 100 * miles_per_year * EV_model_df['years_of_ownership']))
        #'combE' = kWh/100 miles, 1.047798 tCO2/MWh (from EIA 2.31 lbs CO2/kWh generated from Coal)

    ICEV_model_df['running_emissions'] = ICEV_model_df['Manufacture_Emissions_tCO2'] + (8887 / ICEV_model_df['comb08'] * miles_per_year * ICEV_model_df['years_of_ownership']/1000/1000)
                                                  #8,887 gCO2/gallon
    
    intersection_point_emissions = emissions_intersection_point(EV_model_df, ICEV_model_df)
    intersection_point_cost = cost_intersection_point(EV_model_df, ICEV_model_df)
    #union of the two dataframes
    plot_func_df = pd.concat([EV_model_df, ICEV_model_df])

    #Interesection Point Annotations
    if intersection_point_cost[0] != 0:
        # axs[0].annotate('Breakeven by year ' + str(intersection_point_cost[0]), 
        #                 xy= (intersection_point_cost[0], intersection_point_cost[1]), 
        #                 xytext=(intersection_point_cost[0] - 6, intersection_point_cost[1] + 4000), 
        #                 arrowprops=dict(facecolor='black', shrink=0.05))
        # axs[0].text(x = intersection_point_cost[0] - 4, 
        #             y = intersection_point_cost[1] + 2000, 
        #             s = f'${round(intersection_point_cost[1], 2)}', fontsize=12)
        ax1.vlines(x = intersection_point_cost[0], ymin = 0, ymax = intersection_point_cost[1], color = 'lightgrey', linestyles='dashed')
        ax1.hlines(y = intersection_point_cost[1], xmin = 0, xmax = intersection_point_cost[0], color = 'lightgrey', linestyles='dashed')
    
    if intersection_point_emissions[0] != 0:
        # axs[1].text(x = intersection_point_emissions[0] - 4, 
        #             y = intersection_point_emissions[1] + 0.25, 
        #             s = f'{round(intersection_point_emissions[1], 2)} tCO2', fontsize=12)
        ax2.vlines(x = intersection_point_emissions[0], ymin = 0, ymax = intersection_point_emissions[1], color = 'lightgrey', linestyles='dashed')
        ax2.hlines(y = intersection_point_emissions[1], xmin = 0, xmax = intersection_point_emissions[0], color = 'lightgrey', linestyles='dashed')

   
    #Cost Plot
    #sns.set_style("white")
    #sns.set_palette("muted")
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 24
    #axs[0].set_facecolor('#F2F0EA')
    sns.lineplot(data=plot_func_df, 
                x='years_of_ownership', 
                y='running_cost_of_ownership', 
                hue='model',
                ax=ax1)
    
    #axs[0].set_title(f'Cost of Ownership')# for {model_1} and {model_2}', fontname='Sans Serif')
    ax1.set_xlabel('Years of Ownership', fontsize=28)
    ax1.set_ylabel('Cost of Ownership ($)', fontsize=28)
    ax1.set_ylim(bottom = 0)
    ax1.set_xlim(0, 20)
    ax1.set_xticks(range(0, 21, 2))
    ax1.lines[0].set_linewidth(3)
    ax1.lines[1].set_linewidth(3)

    #Emissions Plot
    sns.lineplot(data=plot_func_df, 
                x='years_of_ownership', 
                y='running_emissions', 
                hue='model',
                ax=ax2)
    ax2.set_title(f'Emissions for {model_1} and {model_2}')
    ax2.set_xlabel('Years of Ownership')
    ax2.set_ylabel('Emissions (tCO2)')
    ax2.set_ylim(bottom = 0)
    ax2.set_ylim(0,140)
    ax2.set_xlim(0, 20)
    ax2.set_xticks(range(0, 21, 2))
    ax2.lines[0].set_linewidth(3)
    ax2.lines[1].set_linewidth(3)

    st.markdown("<h2 style='text-align: center;'>Cost of Ownership</h2>", unsafe_allow_html=True)
    st.pyplot(fig1)

    st.markdown("<h2 style='text-align: center;'>Emissions</h2>", unsafe_allow_html=True)
    st.pyplot(fig2)
    # st.metric("Breakeven Year", intersection_point_cost[0], delta="2") ## could be used for showing the dashboarding metrics that Kelbe wants

def add_make_to_model(model):
    make = vehicles_df[vehicles_df['model'] == model]['make'].values[0]
    return make + " " + model

def radio_button_output(grid_emissions_option):
    if grid_emissions_option == 1:
        return 'Actual Projected Grid'
    elif grid_emissions_option == 2:
        return 'Today\'s Grid'
    elif grid_emissions_option == 3:
        return 'All-Coal'

vehicles_df['running_cost_of_ownership'] = 0.00
vehicles_df['running_emissions'] = 0.00

st.title('Utah Clean Energy EV vs ICEV Cost and Emissions Calculator')

st.write('Created By Adrian Martino')

col1, col2 = st.columns(2)
with col1:
    EV_dropdown = st.selectbox(
        label='Select EV',
        options=vehicles_df[vehicles_df['fuelType'] == 'Electricity']['model'],
        format_func=add_make_to_model
    )

    gas_price_slider = st.slider(
         label='Gas Price ($/gallon):', 
         min_value=2.00, 
         max_value=5.00, 
         value=3.15,
         help='Here is why we made this choice',
         label_visibility="visible")
    
    electricity_slider = st.slider(
        label='Electricity Price ($/kWh):',
        min_value=0.00, 
        max_value=0.30, 
        value=0.12,
        help='Here is why we made this choice',
        label_visibility="visible")

    tax_credit_link = 'https://homes.rewiringamerica.org/federal-incentives/30d-new-ev-tax-incentive'
    tax_credit_checkbox = st.checkbox(
         label='Include Federal Tax Credit',
         help='Check this box to include the Federal Tax Credit for EVs.  For more information, [click here](' + tax_credit_link + ').',
         )

with col2:
    ICEV_dropdown = st.selectbox(
        label = 'Select ICEV:',
        options = vehicles_df[vehicles_df['fuelType'] == 'Regular']['model'],
        format_func = add_make_to_model
    )

    grid_emissions_radio_buttons = st.radio(
     label='Grid Emissions Options:', 
     options=[1,2,3],
     captions=["PacifiCorp's Forecasts", "Based on 2023 Actuals", "Hypothetical All-Coal Grid"],
     format_func=radio_button_output
     )
    
    miles_input_box = st.number_input(
         label='Miles/Year',
         min_value=0, 
         max_value=20000, 
         value=11000, 
         step=500)

#st.header('Check out these charts!')

plot_cars(model_1=EV_dropdown, model_2=ICEV_dropdown, gas_price=gas_price_slider, kwh_price=electricity_slider, grid_emissions_option=grid_emissions_radio_buttons, miles_per_year=miles_input_box)
