from datetime import datetime
from multiprocessing import Process, Pool


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
    # 'criminal_detection_police':float,
    'crim_loose_goal': bool,
    'units': float,
    'P_multiple_at_station': float,
    # 'crim_Mguard_percent': float,
    # 'crim_Sguard_percent': float,
    'init_call_delay': float,
    # 'info_update_freq':float,
}

#load uncertainties
df_uncertainties = pd.read_csv('data\exp_uncertainties_no_int.csv', sep=',')


current_date = datetime.now().strftime('%d%m%Y')

# Create the filename with today's date
# filename = f'data/experiment_scenarios_{current_date}.csv'
filename='data/base_28112023.csv'
# filename='data/experiment_BASEscenario_27112023.csv'
scenario_date=filename[5::]

dfscenarios = pd.read_csv(filename, sep=',')
# dfscenarios =dfscenarios.iloc[]

collected_data = []

if len(df_uncertainties)<50:
    crim_start_pos={'centre':['Beurs', 'Schiedam Centrum','Beurs', 'Schiedam Centrum'],\
                    'one_line':['Poortugaal', 'Slotlaan','Poortugaal', 'Slotlaan'],\
                    'end':['De Akkers', 'Binnenhof','De Akkers', 'Binnenhof']}
else:
    crim_start_pos={'centre':['Beurs', 'Schiedam Centrum'],'one_line':['Poortugaal', 'Slotlaan'],'end':['De Akkers', 'Binnenhof']}


def f(row):

    # NO INTERACTION
    scenario,crim_pos,crim_strat,crim_bounded_rat,crim_max_diverge,pol_strat,\
    pol_guarding,Police_entrance,units,P_multiple_at_station = row

    # CRIM INTERACTION
    # scenario,crim_pos,crim_strat,crim_bounded_rat,crim_bound_rat_time,crim_loose_goal,crim_max_diverge,\
    # pol_strat,pol_guarding,Police_entrance,units,P_multiple_at_station= row

    #POL INTERACTION
    # scenario,crim_pos,crim_strat,crim_bounded_rat,police_undercover,\
    # crim_max_diverge,pol_strat,pol_guarding,Police_entrance,units,P_multiple_at_station= row

    #FULL INTERACTION
    # scenario,crim_pos,crim_strat,crim_bounded_rat,crim_bound_rat_time,crim_loose_goal,police_undercover,\
    # crim_max_diverge,pol_strat,pol_guarding,Police_entrance,units,P_multiple_at_station = row

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
                'init_call_delay': lambda m: m.init_call_delay,
                'info_update_freq': lambda m: m.info_update_freq,
                'criminal_detection_police': lambda m: m.criminal_detection_police,
    })

    for iteration, item in enumerate(df_uncertainties['crim_Mguard_percent']):
        crim_Mguard_percent = df_uncertainties['crim_Mguard_percent'].iloc[iteration]
        crim_Sguard_percent = df_uncertainties['crim_Sguard_percent'].iloc[iteration]
        init_call_delay = df_uncertainties['init_call_delay'].iloc[iteration]
        # criminal_detection_police = df_uncertainties['criminal_detection_police'].iloc[iteration]
        # info_update_freq=df_uncertainties['info_update_freq'].iloc[iteration]


        for pos in crim_start_pos[crim_pos]:
            start_pos = pos
            model = MetroModel(crim_pos=start_pos,
                               crim_strat=crim_strat,
                               crim_bounded_rat=crim_bounded_rat,
                               # crim_bound_rat_time=crim_bound_rat_time, #interactie
                               crim_max_diverge=crim_max_diverge,
                               pol_strat=pol_strat,
                               pol_guarding=pol_guarding,
                               # police_undercover=police_undercover, #interactie
                               Police_entrance=Police_entrance,
                               P_multiple_at_station=P_multiple_at_station,
                               # crim_loose_goal=crim_loose_goal, #interactie
                               units=units,
                               init_call_delay=init_call_delay,  # uncertainty
                               # criminal_detection_police=criminal_detection_police, # uncertainty interaction
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


    # Concatenate collected data for each scenario into one DataFrame and save it
    for i, scenario_data in enumerate(collected_data):
        file_name = f"experiment/Sc{scenario}_{scenario_date}.csv"
        scenario_data.to_csv(file_name, index=False)

if __name__ == '__main__':

    pool = Pool(processes=10)
    for row in dfscenarios.itertuples(index=False):
        args = [row[0:]]
        result = pool.apply_async(f, args=(args))

    pool.close()
    pool.join()


