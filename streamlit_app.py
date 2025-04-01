import streamlit as st

st.set_page_config(page_title = 'Utah Clean Energy EV Tool',
                   layout="wide",
                   menu_items={
                       'Get help': 'https://utahcleanenergy.org'})

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

vehicles_df = pd.read_csv('data/limited_vehicles.csv')
grid_emissions_df = pd.read_csv('data/grid_emissions_forecast.csv')

def emissions_intersection_point_months(df_EV, df_ICEV):
    #Takes two dataframes, one for EV and one for ICEV, and returns the month and emissions at which the two intersect
    #Returns the month AFTER the intersction point
    #If no intersection point is found, returns (0,0)
    df_EV_comp = df_EV[['cumulative_months', 'running_emissions']]
    df_EV_comp.rename(columns={'running_emissions': 'EV_running_emissions'}, inplace=True)

    df_ICEV_comp = df_ICEV[['cumulative_months', 'running_emissions']]
    df_ICEV_comp.rename(columns={'running_emissions': 'ICEV_running_emissions'}, inplace=True)

    df_comp = pd.merge(df_EV_comp, df_ICEV_comp, on='cumulative_months', how='outer')
    df_comp['diff'] = df_comp['EV_running_emissions'] - df_comp['ICEV_running_emissions']
    df_comp['diff_sign'] = np.sign(df_comp['diff'])
    df_comp['ly_diff_sign'] = df_comp['diff_sign'].shift(1)
    df_comp['is_intersection_point'] = np.where((df_comp['diff_sign'] != df_comp['ly_diff_sign']) & (df_comp['cumulative_months'] > 1), 
                                            1, 
                                            0)
    #if there is an exact match, set the intersection point to 2
    df_comp['is_intersection_point'] = np.where((df_comp['diff_sign'] == 0),
                                            2,
                                            df_comp['is_intersection_point'])

    df_return = df_comp[df_comp['is_intersection_point'] > 0][['cumulative_months', 'EV_running_emissions', 'ICEV_running_emissions']]
    #select row with max intersection point
    df_return = df_comp[df_comp['is_intersection_point'] == df_comp['is_intersection_point'].max()]
    try:
            return_mth = df_return['cumulative_months'].values[0]
            return_EV_emissions = df_return['EV_running_emissions'].values[0]
            return_ICEV_emissions = df_return['ICEV_running_emissions'].values[0]
    except:
            return_mth = 0
            return_EV_emissions = 0
            return_ICEV_emissions = 0
    return_emissions = np.average([return_EV_emissions, return_ICEV_emissions])

    return_intersection_point = (return_mth, return_emissions)
    return return_intersection_point

def cost_intersection_point_months(df_EV, df_ICEV):
    #Takes two dataframes, one for EV and one for ICEV, and returns the year and emissions at which the two intersect
    #Returns the year AFTER the intersction point
    #If no intersection point is found, returns (0,0)
    df_EV_comp = df_EV[['cumulative_months', 'running_cost_of_ownership']]
    df_EV_comp.rename(columns={'running_cost_of_ownership': 'EV_running_cost'}, inplace=True)

    df_ICEV_comp = df_ICEV[['cumulative_months', 'running_cost_of_ownership']]
    df_ICEV_comp.rename(columns={'running_cost_of_ownership': 'ICEV_running_cost'}, inplace=True)

    df_comp = pd.merge(df_EV_comp, df_ICEV_comp, on='cumulative_months', how='outer')
    df_comp['diff'] = df_comp['EV_running_cost'] - df_comp['ICEV_running_cost']
    df_comp['diff_sign'] = np.sign(df_comp['diff'])
    df_comp['ly_diff_sign'] = df_comp['diff_sign'].shift(1)
    df_comp['is_intersection_point'] = np.where((df_comp['diff_sign'] != df_comp['ly_diff_sign']) & (df_comp['cumulative_months'] > 1), 
                                            1, 
                                            0)
    #if there is an exact match, set the intersection point to 2
    df_comp['is_intersection_point'] = np.where((df_comp['diff_sign'] == 0),
                                            2,
                                            df_comp['is_intersection_point'])

    df_return = df_comp[df_comp['is_intersection_point'] > 0][['cumulative_months', 'EV_running_cost', 'ICEV_running_cost']]
    #select row with max intersection point
    df_return = df_comp[df_comp['is_intersection_point'] == df_comp['is_intersection_point'].max()]
    try:
            return_mth = df_return['cumulative_months'].values[0]
            return_EV_cost = df_return['EV_running_cost'].values[0]
            return_ICEV_cost = df_return['ICEV_running_cost'].values[0]
    except:
            return_mth = 0
            return_EV_cost = 0
            return_ICEV_cost = 0
    return_cost = np.average([return_EV_cost, return_ICEV_cost])

    return_intersection_point = (return_mth, return_cost)
    return return_intersection_point

def calculate_lifetime_savings(df_EV, df_ICEV):
    try:
        EV_lifetime_cost = df_EV[df_EV['cumulative_months'] == 180]['running_cost_of_ownership'].values[0]
        ICEV_lifetime_cost = df_ICEV[df_ICEV['cumulative_months'] == 180]['running_cost_of_ownership'].values[0]
        lifetime_savings = ICEV_lifetime_cost - EV_lifetime_cost
    except:
        lifetime_savings = 0
    lifetime_savings = "{:,.0f}".format(lifetime_savings)
    return lifetime_savings


def plot_cars(model_1, model_2, gas_price=3.15, kwh_price=0.12, grid_emissions_option=1, miles_per_year=11000, apply_tax_credit=False):

    fig1 = plt.figure(figsize=(30, 15))
    fig1.patch.set_linewidth(2)
    fig1.patch.set_edgecolor('black')
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure(figsize=(30, 15))
    fig2.patch.set_linewidth(2)
    fig2.patch.set_edgecolor('black')
    ax2 = fig2.add_subplot(111)

    EV_model_df = vehicles_df[vehicles_df['model'] == model_1]
    ICEV_model_df = vehicles_df[vehicles_df['model'] == model_2]

    years_of_ownership_df = pd.DataFrame({'years_of_ownership': range(0,21)})
    months_of_ownership_df = pd.DataFrame({'months_of_ownership': range(1, 13)})
    time_of_ownership_df = pd.merge(years_of_ownership_df, months_of_ownership_df, how='cross')
    time_of_ownership_df['cumulative_months'] = time_of_ownership_df['months_of_ownership'] + (time_of_ownership_df['years_of_ownership'] * 12)
    new_row = pd.DataFrame([[0, 0, 0]], columns=time_of_ownership_df.columns)
    time_of_ownership_df = pd.concat([new_row, time_of_ownership_df], ignore_index=True)
    time_of_ownership_df.drop(columns=['months_of_ownership'], inplace=True)
    
    EV_model_df = pd.merge(EV_model_df, time_of_ownership_df, how='cross')
    ICEV_model_df = pd.merge(ICEV_model_df, time_of_ownership_df, how='cross')

    grid_emissions_df['merge_year'] = grid_emissions_df['forecast_year'] - 2023

    EV_model_df = pd.merge(EV_model_df, grid_emissions_df, how='left', left_on=['years_of_ownership'], right_on=['merge_year'])
    ICEV_model_df = pd.merge(ICEV_model_df, grid_emissions_df, how='left', left_on=['years_of_ownership'], right_on=['merge_year'])

    EV_model_df['running_cost_of_ownership'] = EV_model_df['MSRP'] + (1013 * 0.65 * EV_model_df['cumulative_months']/12) + (miles_per_year * EV_model_df['combE'] / 100 * kwh_price * EV_model_df['cumulative_months']/12) + np.where(((1.11 / 100 * miles_per_year) < 138.5),
                                                                                                                                                                                                                                      (1.11 / 100 * miles_per_year / 12),
                                                                                                                                                                                                                                      (138.5 / 12))
    ICEV_model_df['running_cost_of_ownership'] = ICEV_model_df['MSRP'] + (1013 * ICEV_model_df['cumulative_months']/12) + (miles_per_year / ICEV_model_df['comb08'] * gas_price * ICEV_model_df['cumulative_months']/12)

    EV_model_df['ty_emissions'] = np.where(EV_model_df['cumulative_months'] != 0,
                                           EV_model_df['emissions_tCO2MWh'] / 1000 * EV_model_df['combE'] / 100 * miles_per_year / 12,
                                           0)
    EV_model_df['starting_emissions'] = np.where(EV_model_df['cumulative_months'] == 0, 
                                                 EV_model_df['Manufacture_Emissions_tCO2'], 
                                                 0)

    if grid_emissions_option == 1:
        EV_model_df['ty_plus'] = EV_model_df['ty_emissions'] + EV_model_df['starting_emissions']
        EV_model_df['running_emissions'] = EV_model_df['ty_plus'].cumsum()
    elif grid_emissions_option == 2:
        EV_model_df['running_emissions'] = EV_model_df['Manufacture_Emissions_tCO2'] + ((0.532954 / 1000 * EV_model_df['combE'] / 100 * miles_per_year * EV_model_df['cumulative_months']/12))
        #'combE' = kWh/100 miles, 0.532954 tCO2/MWh (from Logan)
    elif grid_emissions_option == 3:
        EV_model_df['running_emissions'] = EV_model_df['Manufacture_Emissions_tCO2'] + ((1.047798 / 1000 * EV_model_df['combE'] / 100 * miles_per_year * EV_model_df['cumulative_months']/12))
        #'combE' = kWh/100 miles, 1.047798 tCO2/MWh (from EIA 2.31 lbs CO2/kWh generated from Coal)

    ICEV_model_df['running_emissions'] = ICEV_model_df['Manufacture_Emissions_tCO2'] + (8887 / ICEV_model_df['comb08'] * miles_per_year * ICEV_model_df['cumulative_months']/12/1000/1000)
                                                  #8,887 gCO2/gallon
  
    intersection_point_emissions = emissions_intersection_point_months(EV_model_df, ICEV_model_df)
    intersection_point_cost = cost_intersection_point_months(EV_model_df, ICEV_model_df)
    lifetime_savings = calculate_lifetime_savings(EV_model_df, ICEV_model_df)
    
    plot_func_df = pd.concat([EV_model_df, ICEV_model_df])

    UCE_blue = '#016495'
    UCE_red = '#D84829'
    color_mapping = {'Electricity': f'{UCE_blue}', 'Regular': f'{UCE_red}'}


    ### Emissions Plot ###

    sns.lineplot(data=plot_func_df, 
                x='cumulative_months', 
                y='running_emissions', 
                hue='fuelType',
                palette=color_mapping,
                ax=ax2)
    ax2.set_title('\n')
    ax2.set_xlabel('\n Years of Ownership \n', fontsize=32)#, fontweight='bold')
    ax2.set_ylabel('\n Emissions (tCO2) \n', fontsize=40)
    ax2.set_ylim(0,140)
    ax2.set_xlim(0, 182)
    xtick_positions = range(12, 181, 12)
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels([f"{m//12}" for m in xtick_positions])
    ax2.tick_params(axis='x', colors='black')
    ax2.tick_params(axis='y', colors='black')
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        #label.set_fontweight("bold")
        label.set_fontsize(30)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.lines[0].set_linewidth(3)
    ax2.lines[1].set_linewidth(3)
    ax2.legend_.remove()
    legend_patches = [mpatches.Patch(color=color, 
                                     label=plot_func_df.loc[plot_func_df['fuelType'] == label1, 'model'].iloc[0])
                                    for label1, color in color_mapping.items()]
    ax2.legend(handles=legend_patches, title="Model", fontsize=28, title_fontsize=32)

    if apply_tax_credit:
        tax_credit_EV_df = EV_model_df
        tax_credit_EV_df['running_cost_of_ownership'] = tax_credit_EV_df['running_cost_of_ownership'] - 7500
        tax_credit_EV_df['fuelType'] = '(w/ Tax Credit)'
        plot_func_df.loc[plot_func_df['fuelType'] == 'Electricity', 'fuelType'] = '(w/o Tax Credit)'        
        plot_func_df = pd.concat([plot_func_df, tax_credit_EV_df])
        color_mapping = {'(w/ Tax Credit)': f'{UCE_blue}', '(w/o Tax Credit)': 'grey', 'Regular': f'{UCE_red}'}
        intersection_point_cost = cost_intersection_point_months(tax_credit_EV_df, ICEV_model_df)
        lifetime_savings = calculate_lifetime_savings(tax_credit_EV_df, ICEV_model_df)


    ### Interesection Points ###

    if intersection_point_cost[0] != 0:
        # axs[0].annotate('Breakeven by year ' + str(intersection_point_cost[0]), 
        #                 xy= (intersection_point_cost[0], intersection_point_cost[1]), 
        #                 xytext=(intersection_point_cost[0] - 6, intersection_point_cost[1] + 4000), 
        #                 arrowprops=dict(facecolor='black', shrink=0.05))
        # axs[0].text(x = intersection_point_cost[0] - 4, 
        #             y = intersection_point_cost[1] + 2000, 
        #             s = f'${round(intersection_point_cost[1], 2)}', fontsize=12)
        ax1.vlines(x = intersection_point_cost[0], 
                   ymin = 0, 
                   ymax = intersection_point_cost[1], 
                   color = 'grey', 
                   #linestyles='dashed',
                   linestyle=(0, (10, 4)) #dashed line
                   )
        ax1.hlines(y = intersection_point_cost[1], 
                   xmin = 0, 
                   xmax = intersection_point_cost[0], 
                   color = 'lightgrey', 
                   #linestyles='dashed',
                   linestyle=(0, (10, 4))
                   )

    if intersection_point_emissions[0] != 0:
        # axs[1].text(x = intersection_point_emissions[0] - 4, 
        #             y = intersection_point_emissions[1] + 0.25, 
        #             s = f'{round(intersection_point_emissions[1], 2)} tCO2', fontsize=12)
        ax2.vlines(x = intersection_point_emissions[0], 
                   ymin = 0, 
                   ymax = intersection_point_emissions[1], 
                   color = 'grey', 
                   linestyle=(0, (10, 4)) #dashed line
                   )
        ax2.hlines(y = intersection_point_emissions[1], 
                   xmin = 0, 
                   xmax = intersection_point_emissions[0], 
                   color = 'lightgrey', 
                   linestyle=(0, (10, 4))
                   ) #dashed line

   
    ### Cost Plot ###

    sns.lineplot(data=plot_func_df, 
                x='cumulative_months', 
                y='running_cost_of_ownership', 
                hue='fuelType',
                palette=color_mapping,
                ax=ax1)
    cost_formatter = ticker.FuncFormatter(lambda x, pos: f'${x:,.0f}')
    ax1.yaxis.set_major_formatter(cost_formatter)
    ax1.yaxis.set_minor_formatter(cost_formatter)
    ax1.set_title('\n')
    ax1.set_xlabel('\n Years of Ownership \n', fontsize=32)#, fontweight='bold')
    ax1.set_ylabel('\n Cost of Ownership ($) \n', fontsize=40)
    ax1.set_ylim(bottom = 0)
    ax1.set_xlim(0, 182)
    #xtick_positions = range(12, 181, 12)
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels([f"{m//12}" for m in xtick_positions])
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        #label.set_fontweight("bold")
        label.set_fontsize(30)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.lines[0].set_linewidth(3)
    ax1.lines[1].set_linewidth(3)
    if apply_tax_credit:
        ax1.lines[0].set_linewidth(1) #set old EV line to thin
        ax1.lines[2].set_linewidth(3)
        legend_patches = [mpatches.Patch(color=color, 
                                         label=(plot_func_df.loc[plot_func_df['fuelType'] == label1, 'model'].iloc[0] + " " + label1)
                                         if not plot_func_df.loc[plot_func_df['fuelType'] == label1, 'model'].empty else label1) 
                                         for label1, color in color_mapping.items()]
    else:
        legend_patches = [mpatches.Patch(color=color, 
                                         label=plot_func_df.loc[plot_func_df['fuelType'] == label1, 'model'].iloc[0]
                                         if not plot_func_df.loc[plot_func_df['fuelType'] == label1, 'model'].empty else label1)
                                         for label1, color in color_mapping.items()]
    ax1.legend_.remove()
    ax1.legend(handles=legend_patches, title="Model", fontsize=28, title_fontsize=32)


    ### Streamlit ###

    #col1, col_space, col2 = st.columns([12,1,4])
    col1, col2 = st.columns([3.5,1])
    with col1:
        st.markdown("<h2 style='text-align: center;'>Cost of Ownership</h2>", unsafe_allow_html=True)
        st.pyplot(fig1)

        st.markdown("<h2 style='text-align: center;'>Emissions</h2>", unsafe_allow_html=True)
        st.pyplot(fig2)
    
    with col2:
        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown(' ')
        
        vehicle_age_link = 'https://www.bts.gov/content/average-age-automobiles-and-trucks-operation-united-states'
        st.metric(label = "Lifetime Savings: ", 
                  value = "$" + str(lifetime_savings),
                  label_visibility = "visible",
                  help = f"Assuming 14 years of ownership--the [average age of cars passenger cars in operation]({vehicle_age_link}) in the United States")
        
        st.markdown(' ')
        st.markdown(' ')

        st.metric(label = "Breakeven After: ",
                  value = str(int(intersection_point_cost[0]/12)) + ' Years',
                  label_visibility = "visible",
                  help = "The year during which the lifetime cost of ownership of the EV is less than the ICEV")
    
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


### Web App UI ###

st.title('EV vs Gas Vechicles: Cost and Emissions Visualization Tool')



with st.expander("How to Read This Chart"):
    #st.write('''Insert annotated version of charts here''')
    st.image('How-To Mock Up.jpg')

with st.sidebar:
    EV_dropdown = st.selectbox(
        label='Select Electric Vehicle:',
        options=vehicles_df[vehicles_df['fuelType'] == 'Electricity']['model'],
        format_func=add_make_to_model
        )
    
    ICEV_dropdown = st.selectbox(
        label = 'Select Gas Vehicle:',
        options = vehicles_df[vehicles_df['fuelType'] == 'Regular']['model'],
        format_func = add_make_to_model
        )
    
    gas_price_slider = st.slider(
        label='Gas Price ($/gallon):', 
        min_value=2.00, 
        max_value=5.00, 
        value=3.20,
        help='Default set to the average price of gas in Utah, $3.20/gallon',
        label_visibility="visible")
        #on_change=)
        
    electricity_slider = st.slider(
        label='Electricity Price ($/kWh):',
        min_value=0.00, 
        max_value=0.30, 
        value=0.12,
        help='Default set to the average cost of electricity in Utah, $0.12/kWh',
        label_visibility="visible")

    tax_credit_link = 'https://homes.rewiringamerica.org/federal-incentives/30d-new-ev-tax-incentive'
    tax_credit_checkbox = st.checkbox(
        label='Apply Full Federal Tax Credit',
        help='Check this box to include the full, $7,500 Federal Tax Credit for EVs.  For more information, [click here](' + tax_credit_link + ').',
        )

    miles_input_box = st.number_input(
        label='Miles Driven per Year:',
        min_value=0, 
        max_value=20000, 
        value=11000, 
        step=500)

    grid_emissions_radio_buttons = st.radio(
        label='Grid Emissions Options:', 
        options=[1,2,3],
        captions=["PacifiCorp's Forecasts", "Based on 2023 Actuals", "Hypothetical All-Coal Grid"],
        format_func=radio_button_output
        )

plot_cars(model_1=EV_dropdown, model_2=ICEV_dropdown, gas_price=gas_price_slider, kwh_price=electricity_slider, grid_emissions_option=grid_emissions_radio_buttons, miles_per_year=miles_input_box, apply_tax_credit=tax_credit_checkbox)

st.markdown(' ')
st.markdown(' ')
st.write('Created By Adrian Martino')
