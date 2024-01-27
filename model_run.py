from datetime import datetime
from pathlib import Path
import os
import copy
# from numpy import random
# from main import MetroModel
# from mesa.batchrunner import BatchRunner
# from mesa.datacollection import DataCollector
#
# import pandas as pd
#
# # 10 replications for each scenario
# replications = 1
#
# # run time 1x 1 hours; 1 tick 1 minute
# run_length = 720  # 2 hours to get all the metro's up and running
# # and 4 hour for the escape so 6 hours = 6x60=360
# # 360* 2=720 because time step = 0.5 so 720 time steps
#
# # The seed is a randomly chosen integral between 100 and 100000
# seed = 12345  # random.randint(100, 100000)
#
# # Import scenarios.csv and create array
# # dfscenarios = pd.read_csv('../data/scenarios.csv', sep=',')
# # scenarios = dfscenarios.to_numpy()
#
# # Running through every scenario, multiple replications and given amount of steps
# # for row in scenarios:
# #    # bdpA, bdpB, bdpC, bdpD = row[0:] # Take breakdown percentages according to scenarios
# #    runs = pd.DataFrame()    # Clear runs dataframe from scenario run before
# # for replication in range(replications):
# #     sim_model = MetroModel(crim_pos='De Tochten', crim_strat='furthest', crim_bounded_rat=False, pol_strat='largest',
# #                            pol_guarding_metro=True,pol_guarding_station=False)
# #     runs = pd.DataFrame()  # Clear runs dataframe from scenario run before
# #     for j in range(run_length):
# #         if sim_model.capture==True:
# #             print ('Criminal caught!')
# #             break
# #         if sim_model.escape==True:
# #             print ('Criminal escaped!')
# #             break
# #
# #         else:
# #             sim_model.step()
# #
# #         df = sim_model.datacollector.get_agent_vars_dataframe() # Collect all the data from the model
# #         df = df[["unique_id","pos","in_metro"]] # Add generation_time and travel_distance columns to the df
# #         # if sim_model.capture==False:
# #         #     df['capture'] = False
# #         # elif sim_model.capture==True:
# #         #     df['capture'] = True
# #         # df['capture']='test'
# #         df = (df.dropna()
# #              .reset_index()
# #               .groupby("AgentID") # Sort all data per vehicle
# #              .agg({"Step": "max", # Aggregate all the data by taking only the useful outputs
# #                    # "AgentID": "max",
# #                    "unique_id": "max",
# #                    # "pos": "unique",
# #                    "in_metro":"unique",
# #                    # "capture":"unique",
# #        #                 }).assign(time_taken=lambda x: (x.Step - x.generation_time) / 60, # Calculate the travel time
# #        #                run_number=i + 1) # Mark row with run number
# #                  }) )
# #
# #     # Add each run to the same dataframe.
# #         runs = pd.concat([runs, df], 0)
# #     # Filtering only the trucks that travelled the entire road
# #        #runs = runs.loc[runs['travel_distance'] >= 227000]
# #     # Exporting the filtered dataframe to a csv file with a name corresponding to its scenario
# #     #     print(runs)
# #         runs.to_csv('experiment/scenario{}.csv'.format(replication))
# #
#
# # Save at every time step
# data_types = {
#     'Scenario': float,
#     'Crim init pos': str,
#     'Crim strat': str,
#     'crim bounded rationality': bool,
#     'police strategy': str,
#     'police guarding metro': bool,
#     'police guarding station': bool,
#     'Police units (#)': float,
#     'criminal capture % platform (metro)': float,
#     'criminal capture % station exits': float,
#     'Init call delay (min)': float
# }
# dfscenarios = pd.read_csv('data\Scenarios.csv', sep=',')
#
# for row in dfscenarios.itertuples(index=False):
#     scenario, crim_pos, crim_strat, crim_bounded_rat, pol_strat, pol_guarding_metro, \
#     pol_guarding_station, units, crim_Mguard_percent, crim_Sguard_percent, init_call_delay = row[0:]
#
#     br_params = {'crim_pos': [str(crim_pos)],
#                  'crim_strat': [str(crim_strat)],
#                  'crim_bounded_rat': [bool(crim_bounded_rat)],
#                  'pol_strat': [str(pol_strat)],
#                  'pol_guarding_metro': [bool(pol_guarding_metro)],
#                  'pol_guarding_station': [bool(pol_guarding_station)],
#                  'units': [int(units)],  # police units
#                  'crim_Mguard_percent': [int(crim_Mguard_percent)],
#                  'crim_Sguard_percent': [int(crim_Sguard_percent)],
#                  'init_call_delay': [int(init_call_delay)]}  # minutes
#
#     for iteration in range(6):  # Set the desired number of iterations here
#         br = BatchRunner(MetroModel,
#                          br_params,
#                          iterations=1,  # Run one iteration at a time
#                          max_steps=run_length,
#                          model_reporters={"Model Data Collector": lambda m: m.datacollector},
#                          )
#
#         br.run_all()
#         br_df = br.get_model_vars_dataframe()
#         br_step_data = pd.DataFrame()
#         br_run_data = pd.DataFrame()
#
#         for i in range(len(br_df["Model Data Collector"])):
#             if isinstance(br_df["Model Data Collector"][i], DataCollector):
#                 i_run_data = br_df["Model Data Collector"][i].get_model_vars_dataframe()
#                 br_step_data = br_step_data.append(i_run_data, ignore_index=True)
#                 br_run_data = br_run_data.append(br_step_data.iloc[-1], ignore_index=True)
#
#         # Convert boolean columns back to their original data type (if needed)
#         # boolean_cols = ['capture', 'escape', 'game_over']
#         # br_run_data[boolean_cols] = br_run_data[boolean_cols].astype(bool)
#
#         file_name = f"experiment/Base_scenario_{scenario}_iteration_{iteration + 1}.csv"
#
#         # Save the DataFrame to CSV with the modified file name
#         br_run_data.to_csv(file_name, index=False)

from numpy import random
from main import MetroModel
from mesa.datacollection import DataCollector

import pandas as pd

# 10 replications for each scenario
replications = 1

# run time 1x 1 hours; 1 tick 1 minute
run_length = 720  # 2 hours to get all the metro's up and running
# and 4 hour for the escape so 6 hours = 6x60=360
# 360* 2=720 because time step = 0.5 so 720 time steps

# The seed is a randomly chosen integral between 100 and 100000
seed = 12345  # random.randint(100, 100000)

# Save at every time step
data_types = {
    'scenario': float,
    'crim_pos': str,
    'crim_strat': str,
    'crim_bounded_rat': bool,
    'crim_bound_rat_time':float,
    'crim_max_diverge': float,
    'pol_strat': str,
    'pol_guarding': str,
    'police_undercover':bool,
    'Police_entrance': str,
    'criminal_detection_police':float,
    'crim_loose_goal': bool,
    'units': float,
    'P_multiple_at_station': float,
    'crim_Mguard_percent': float,
    'crim_Sguard_percent': float,
    'init_call_delay': float,
    'info_update_freq':float,
}

# for row in dfscenarios.itertuples(index=False):
#     scenario, crim_pos, crim_strat, crim_bounded_rat, crim_bound_rat_time,crim_max_diverge,pol_strat, pol_guarding, \
#     police_undercover,Police_entrance, criminal_detection_police, crim_loose_goal,units,\
#     P_multiple_at_station, crim_Mguard_percent, crim_Sguard_percent, init_call_delay,info_update_freq = row[0:]
#     collected_data = []
#
#     for iteration in range(3):
#         model = MetroModel(crim_pos=crim_pos,
#                            crim_strat = crim_strat,
#                            crim_bounded_rat = crim_bounded_rat,
#                            crim_bound_rat_time = crim_bound_rat_time,
#                            crim_max_diverge =crim_max_diverge,
#                            pol_strat = pol_strat,
#                            pol_guarding = pol_guarding,
#                            police_undercover = police_undercover,
#                            Police_entrance = Police_entrance,
#                            criminal_detection_police = criminal_detection_police,
#                            crim_loose_goal=crim_loose_goal,
#                            units = units,
#                            P_multiple_at_station = P_multiple_at_station,
#                            crim_Mguard_percent = crim_Mguard_percent,
#                            crim_Sguard_percent = crim_Sguard_percent,
#                            init_call_delay = init_call_delay,
#                            info_update_freq = info_update_freq)
#
#         # data_collector = DataCollector(model_reporters={"Model Data Collector": lambda m: m.datacollector})
#
#         # Create a DataCollector to collect model-level data
#         data_collector = DataCollector(model_reporters={
#             "time": lambda m: m.clocktime,
#             "capture": lambda m: m.capture,
#             "escape": lambda m: m.escape,
#             "game_over": lambda m: m.game_over,
#             'criminal_goal': lambda m: m.criminal_goal,
#             'game_over_pos': lambda m: m.game_over_pos,
#             'criminal_passed_police': lambda m: m.criminal_passed_police,
#             'police_goal': lambda m: m.police_goal,
#             'police_start_pos': lambda m: m.police_start_pos,
#             'police_gone_undercover': lambda m: m.police_gone_undercover,
#             'police_changed_goal':lambda m:m.
#
#
#
#             ,
#             'criminal_diverged_from_path': lambda m:m.criminal_diverged_from_path,
#             'criminal_tried_exits': lambda m: m.crim_tried_exits,
#             'criminal_seen_police': lambda m: m.criminal_seen_police,
#         })
#
#         for step in range(run_length):
#             if not (model.capture == True and model.escape == False) or (model.capture == False and model.escape == True):
#                 model.step()
#
#         if (model.capture == True and model.escape == False) or (model.capture == False and model.escape == True):
#             data_collector.collect(model)
#
#         # Extract collected data from the DataCollector
#         model_data = data_collector.get_model_vars_dataframe()
#         collected_data.append(model_data)
#         print('Scenario:',scenario, 'Iteration:', iteration)
#
#     scenario_data = pd.concat(collected_data)
#     #
#     file_name = f"experiment/Base_scenario_{scenario}.csv"
#     scenario_data.to_csv(file_name, index=False)

# Create an empty list to store collected data for all scenarios

#load uncertainties
df_uncertainties = pd.read_csv('data\exp_uncertainties_crim_int.csv', sep=',')


current_date = datetime.now().strftime('%d%m%Y')

# Create the filename with today's date
# filename = f'data/experiment_scenarios_{current_date}.csv'
filename='data/base_29112023.csv'
# filename='data/experiment_BASEscenario_27112023.csv'
scenario_date=filename[5::]

dfscenarios = pd.read_csv(filename, sep=',')
dfscenarios =dfscenarios.iloc[:1]

collected_data = []

if len(df_uncertainties)<50:
    crim_start_pos={'centre':['Beurs', 'Schiedam Centrum','Beurs', 'Schiedam Centrum'],\
                    'one_line':['Poortugaal', 'Slotlaan','Poortugaal', 'Slotlaan'],\
                    'end':['De Akkers', 'Binnenhof','De Akkers', 'Binnenhof']}
else:
    crim_start_pos={'centre':['Beurs', 'Schiedam Centrum'],'one_line':['Poortugaal', 'Slotlaan'],'end':['De Akkers', 'Binnenhof']}

for row in dfscenarios.itertuples(index=False):
    scenario, crim_pos, crim_strat, crim_bounded_rat, crim_bound_rat_time, crim_loose_goal, crim_max_diverge, \
    pol_strat, pol_guarding, Police_entrance, units, P_multiple_at_station = row
    # Create a DataCollector to collect model-level data
    data_collector = DataCollector(model_reporters={
                "start_time": lambda m: m.start_time,
                "time": lambda m: m.clocktime,
                "capture": lambda m: m.capture,
                "escape": lambda m: m.escape,
                "game_over": lambda m: m.game_over,
                'criminal_goal': lambda m: m.criminal_goal,
                'game_over_pos': lambda m: m.game_over_pos,
                'criminal_passed_police': lambda m: m.criminal_passed_police,
                'police_goal': lambda m: m.police_goal,
                'police_start_pos': lambda m: m.police_start_pos,
                'police_gone_undercover': lambda m: m.police_gone_undercover,
                'police_changed_goal': lambda m: m.police_changed_goal,
                'criminal_diverged_from_path': lambda m: m.criminal_diverged_from_path,
                'criminal_tried_exits': lambda m: m.crim_tried_exits,
                'criminal_seen_police': lambda m: m.criminal_seen_police,
                'crim_Sguard_percent': lambda m: m.crim_Sguard_percent,
                'crim_Mguard_percent': lambda m: m.crim_Mguard_percent,
                'init_call_delay': lambda m: m.init_call_delay
    })

    for iteration, item in enumerate(df_uncertainties['crim_Mguard_percent']):
        crim_Mguard_percent = df_uncertainties['crim_Mguard_percent'].iloc[iteration]
        crim_Sguard_percent = df_uncertainties['crim_Sguard_percent'].iloc[iteration]
        init_call_delay = df_uncertainties['init_call_delay'].iloc[iteration]
        # info_update_freq = df_uncertainties['info_update_freq'].iloc[iteration]
        criminal_detection_police = df_uncertainties['criminal_detection_police'].iloc[iteration]


        for pos in crim_start_pos[crim_pos]:
            start_pos=pos
            model = MetroModel(crim_pos=start_pos,
                               crim_strat=crim_strat,
                               crim_bounded_rat=crim_bounded_rat,
                               crim_bound_rat_time=crim_bound_rat_time,  # interactie
                               crim_max_diverge=crim_max_diverge,
                               pol_strat=pol_strat,
                               pol_guarding=pol_guarding,
                               # police_undercover=police_undercover, #interactie
                               Police_entrance=Police_entrance,
                               P_multiple_at_station=P_multiple_at_station,
                               crim_loose_goal=crim_loose_goal, #interactie
                               units=units,
                               init_call_delay=init_call_delay,  # uncertainty
                               criminal_detection_police=criminal_detection_police, # uncertainty interaction
                               # info_update_freq=info_update_freq,      #uncertainty interaction
                               crim_Mguard_percent=crim_Mguard_percent,  # uncertainty
                               crim_Sguard_percent=crim_Sguard_percent)  # uncertainty


            for step in range(run_length):
                if not (model.capture == True and model.escape == False) or (model.capture == False and model.escape == True):
                    model.step()

            if (model.capture == True and model.escape == False) or (model.capture == False and model.escape == True):
                data_collector.collect(model)
            # if (model.capture == True and model.escape == False) or (model.capture == False and model.escape == True):
            #     # Create a deep copy of the DataCollector object before appending it to the list
            #     data_collector_copy = copy.deepcopy(data_collector)
            #     collected_data.append(data_collector_copy)

        print('Scenario:', scenario, 'Iteration:', iteration)
        print_name = f"experiment/print_output_{current_date}.txt"
        with open(print_name, "a") as f:
            print(('Scenario:', scenario, 'Iteration:', iteration), file=f)

    # Extract collected data from the DataCollector for this scenario
    model_data = data_collector.get_model_vars_dataframe()
    # print(model_data['police_start_pos'])
    collected_data.append(model_data)
        # file_name = f"experiment/AAASc{iteration}_{scenario_date}.csv"

            # if os.path.exists(file_name):
            #     model_data.to_csv(file_name, mode="a")
            # else:
            #     model_data.to_csv(file_name)

        # print(22222,collected_data[0])

# Concatenate collected data for each scenario into one DataFrame and save it
for i, scenario_data in enumerate(collected_data):
    file_name = f"experiment/Sc{dfscenarios['scenario'].iloc[i]}_{scenario_date}.csv"
    scenario_data.to_csv(file_name, index=False)




#     file_name = f"experiment/AAASc{dfscenarios['scenario'].iloc[i]}_{scenario_date}.csv"
#     model_data.to_csv(file_name, index=False)

# collected_data = []
#
# for row in dfscenarios.itertuples(index=False):
#     scenario, crim_pos, crim_strat, crim_bounded_rat, crim_bound_rat_time, crim_max_diverge, \
#     pol_strat, pol_guarding, police_undercover, Police_entrance, criminal_detection_police, \
#     crim_loose_goal, units, P_multiple_at_station, crim_Mguard_percent, crim_Sguard_percent, \
#     init_call_delay, info_update_freq = row[0:]
#
#     # Create a DataCollector to collect model-level data
#     data_collector = DataCollector(model_reporters={
#         "start_time": lambda m: m.start_time,
#         "end_time": lambda m: m.scheduler.time,
#         "capture": lambda m: m.capture,
#         "escape": lambda m: m.escape,
#         "game_over": lambda m: m.game_over,
#         'criminal_goal': lambda m: m.criminal_goal,
#         'game_over_pos': lambda m: m.game_over_pos,
#         'criminal_passed_police': lambda m: m.criminal_passed_police,
#         'police_goal': lambda m: m.police_goal,
#         'police_start_pos': lambda m: m.police_start_pos,
#         'police_gone_undercover': lambda m: m.police_gone_undercover,
#         'police_changed_goal': lambda m: m.police_changed_goal,
#         'criminal_diverged_from_path': lambda m: m.criminal_diverged_from_path,
#         'criminal_tried_exits': lambda m: m.crim_tried_exits,
#         'criminal_seen_police': lambda m: m.criminal_seen_police,
#
#     })
#
#     for iteration in range(30):
#         model = MetroModel(crim_pos=crim_pos,
#                            crim_strat=crim_strat,
#                            crim_bounded_rat=crim_bounded_rat,
#                            crim_bound_rat_time=crim_bound_rat_time,
#                            crim_max_diverge=crim_max_diverge,
#                            pol_strat=pol_strat,
#                            pol_guarding=pol_guarding,
#                            police_undercover=police_undercover,
#                            Police_entrance=Police_entrance,
#                            criminal_detection_police=criminal_detection_police,
#                            crim_loose_goal=crim_loose_goal,
#                            units=units,
#                            P_multiple_at_station=P_multiple_at_station,
#                            crim_Mguard_percent=crim_Mguard_percent,
#                            crim_Sguard_percent=crim_Sguard_percent,
#                            init_call_delay=init_call_delay,
#                            info_update_freq=info_update_freq)
#
#         for step in range(run_length):
#             if not (model.capture == True and model.escape == False) or (model.capture == False and model.escape == True):
#                 model.step()
#
#         if (model.capture == True and model.escape == False) or (model.capture == False and model.escape == True):
#             data_collector.collect(model)
#
#         # Extract collected data from the DataCollector
#         model_data = data_collector.get_model_vars_dataframe()
#         collected_data.append(model_data)
#         print('Scenario:', scenario, 'Iteration:', iteration)
#
#         scenario_data = pd.concat(collected_data)
#         file_name = f"experiment/Sc{scenario}.csv"
#         scenario_data.to_csv(file_name, index=False)
#
# # Concatenate collected data for all scenarios into one DataFrame
# final_data = pd.concat(collected_data)

# Save the final_data DataFrame to a single CSV file
# final_data.to_csv('experiment/output.csv', index=False)