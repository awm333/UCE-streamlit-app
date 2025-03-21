import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from ipywidgets import widgets, interactive
#from ipywidgets.embed import embed_minimal_html

vehicles_df = pd.read_csv('data/limited_vehicles.csv')
## topselling_df = pd.read_excel('data/topsellingcars.xlsx')
grid_emissions_df = pd.read_excel('data/grid_emissions_forecast.xlsx')

def plot_cars(model_1, model_2, gas_price, kwh_price, grid_emissions_option, miles_per_year):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    
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
    if grid_emissions_option == 'Forecasts':
        EV_model_df['ty_plus'] = EV_model_df['ty_emissions'] + EV_model_df['starting_emissions']
        EV_model_df['running_emissions'] = EV_model_df['ty_plus'].cumsum()
    elif grid_emissions_option == 'Current Emissions':
        EV_model_df['running_emissions'] = EV_model_df['Manufacture_Emissions_tCO2'] + ((0.532954 / 1000 * EV_model_df['combE'] / 100 * miles_per_year * EV_model_df['years_of_ownership']))
        #'combE' = kWh/100 miles, 0.532954 tCO2/MWh (from Logan)
    elif grid_emissions_option == 'All-Coal':
        EV_model_df['running_emissions'] = EV_model_df['Manufacture_Emissions_tCO2'] + ((1.047798 / 1000 * EV_model_df['combE'] / 100 * miles_per_year * EV_model_df['years_of_ownership']))
        #'combE' = kWh/100 miles, 1.047798 tCO2/MWh (from EIA 2.31 lbs CO2/kWh generated from Coal)

    ICEV_model_df['running_emissions'] = ICEV_model_df['Manufacture_Emissions_tCO2'] + (8887 / ICEV_model_df['comb08'] * miles_per_year * ICEV_model_df['years_of_ownership']/1000/1000)
                                                  #8,887 gCO2/gallon
    
    # intersection_point_emissions = emissions_intersection_point(EV_model_df, ICEV_model_df)
    # intersection_point_cost = cost_intersection_point(EV_model_df, ICEV_model_df)
    # print(intersection_point_emissions)
    # print(intersection_point_cost)
    #union of the two dataframes
    plot_func_df = pd.concat([EV_model_df, ICEV_model_df])


    #Interesection Point Annotations
    # if intersection_point_cost[0] != 0:
    #     # axs[0].annotate('Breakeven by year ' + str(intersection_point_cost[0]), 
    #     #                 xy= (intersection_point_cost[0], intersection_point_cost[1]), 
    #     #                 xytext=(intersection_point_cost[0] - 6, intersection_point_cost[1] + 4000), 
    #     #                 arrowprops=dict(facecolor='black', shrink=0.05))
    #     # axs[0].text(x = intersection_point_cost[0] - 4, 
    #     #             y = intersection_point_cost[1] + 2000, 
    #     #             s = f'${round(intersection_point_cost[1], 2)}', fontsize=12)
    #     axs[0].vlines(x = intersection_point_cost[0], ymin = 0, ymax = intersection_point_cost[1], color = 'lightgrey', linestyles='dashed')
    #     axs[0].hlines(y = intersection_point_cost[1], xmin = 0, xmax = intersection_point_cost[0], color = 'lightgrey', linestyles='dashed')
    
    # if intersection_point_emissions[0] != 0:
    #     # axs[1].text(x = intersection_point_emissions[0] - 4, 
    #     #             y = intersection_point_emissions[1] + 0.25, 
    #     #             s = f'{round(intersection_point_emissions[1], 2)} tCO2', fontsize=12)
    #     axs[1].vlines(x = intersection_point_emissions[0], ymin = 0, ymax = intersection_point_emissions[1], color = 'lightgrey', linestyles='dashed')
    #     axs[1].hlines(y = intersection_point_emissions[1], xmin = 0, xmax = intersection_point_emissions[0], color = 'lightgrey', linestyles='dashed')


    #Cost Plot
    sns.lineplot(data=plot_func_df, 
                x='years_of_ownership', 
                y='running_cost_of_ownership', 
                hue='model',
                ax=axs[0])
    axs[0].set_title(f'Cost of Ownership for {model_1} and {model_2}')
    axs[0].set_xlabel('Years of Ownership')
    axs[0].set_ylabel('Cost of Ownership ($)')
    axs[0].set_ylim(bottom = 0)
    axs[0].set_xlim(0, 20)
    axs[0].set_xticks(range(0, 21, 2))


    #Emissions Plot
    sns.lineplot(data=plot_func_df, 
                x='years_of_ownership', 
                y='running_emissions', 
                hue='model',
                ax=axs[1])
    axs[1].set_title(f'Emissions for {model_1} and {model_2}')
    axs[1].set_xlabel('Years of Ownership')
    axs[1].set_ylabel('Emissions (tCO2)')
    axs[1].set_ylim(bottom = 0)
    axs[1].set_ylim(0,140)
    axs[1].set_xlim(0, 20)
    axs[1].set_xticks(range(0, 21, 2))
    
    st.pyplot(fig)
    return plot_func_df

vehicles_df = vehicles_df[['year','make', 'model', 'comb08', 'combE', 'fuelCost08', 'cylinders', 'fuelType', 'id', 'VClass']]
vehicles_df['running_cost_of_ownership'] = 0.00
vehicles_df['running_emissions'] = 0.00

vehicles_df = pd.merge(topselling_df, vehicles_df, how='left', left_on=['epa_id'], right_on=['id'])

st.title('Utah Clean Energy EV vs ICEV Cost and Emissions Calculator')

st.write('Created By Adrian Martino')

col1, col2 = st.columns(2)

with col1:
    EV_dropdown = st.selectbox(
        'Select EV',
        vehicles_df[vehicles_df['fuelType'] == 'Electricity']['model']
    )

with col2:
    ICEV_dropdown = st.selectbox(
        'Select ICEV:',
        vehicles_df[vehicles_df['fuelType'] == 'Regular']['model']
    )

st.header('Check out these charts!')

vehicles_df.head(2)

plot_cars(model_1=EV_dropdown, model_2=ICEV_dropdown, gas_price=3.15, kwh_price=0.12, grid_emissions_option='Forecasts', miles_per_year=11000)

