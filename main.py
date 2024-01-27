import networkx as nx
import matplotlib.pyplot as plt
# import random
# import time
import math
import sys
from collections import deque
import json
import os
# import pandas as pd
# from mesa import Agent
# from Police import *
from Criminal import *
from Metro import *
# from GameRun import Game
# from mesa.visualization.ModularVisualization import ModularServer
# from mesa import Agent
# from enum import Enum
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
# from mesa.space import ContinuousSpace
# from collections import defaultdict
#
# from networkx import dijkstra_path


# ------------------------------------------------------------------------
"""In this code: 
                MAKE NETWORK
                INIT GRAPH
                METRO SCHEDULE
                STEP"""
# -----------------------------------------------------------------------------------------



# read the CSV file into a pandas dataframe
# dfNodes = pd.read_csv('Nodes_complete_police.csv')
dfNodes = pd.read_csv('data/Nodes_complete_police.csv')
dfTimeTable = pd.read_csv('data/MetroTimetTable.csv')

# dfNodes = pd.read_csv('data/simpleNodes.csv')




# run_amount = int(input("How many runs do you want per defined scenario?"))
# while isinstance(run_amount, int) is not True or run_amount < 1:
#     run_amount = int(input("Invalid input, use integer above 0"))

#seed = 1234567

# -------------------------------------------------------------------------
"""MAKE NETWORK"""
# -----------------------------------------------------------------------------------------
class MetroModel(Model):

    step_time=0.5

    def __init__(self,crim_pos,crim_strat, pol_strat,pol_guarding, crim_bounded_rat,units, init_call_delay,P_multiple_at_station, \
                 crim_Sguard_percent,crim_Mguard_percent,Police_entrance,crim_max_diverge, \
                 crim_bound_rat_time=100, crim_loose_goal=False, criminal_detection_police=1000, \
                 police_undercover=False, info_update_freq=1000, \
                 step_num=0, G=nx.graph, location=None, exits = None,\
                 mode_node=None, node_color=None, node_labels=None, source_node=None, \
                 function=None, coordinates=None, mode_edge=None, length=None, edge_labels=None,\
                 edge_color_list=None, police_agent=None, criminal_agent=None,criminal_positions=None,\
                 criminal_information=None,source_sink_node=None, departure_time=None,
                 criminal_id=0, police_id=100, metro_id=999,metro_starting_schedule={},run_hours=0,metro_at_station={},\
                 graph=False,capture=False, escape=False,running=True,clocktime=None, game_over=False, criminal_seen_police=0,\
                 criminal_goal=None,  game_over_pos=None, police_gone_undercover=False,criminal_passed_police=0,criminal_diverged_from_path=0,\
                 police_goal={},police_start_pos={}, police_changed_goal={}, crim_tried_exits=0,
                 ):
        self.scheduler = BaseScheduler(self)
        self.time_step = 0.5
        self.current_time = 0
        self.step_num = step_num
        self.G=G
        self.location = location
        self.exits = exits
        self.mode_node = mode_node
        self.node_color = node_color
        self.node_labels = node_labels
        self.source_node = source_node
        self.function = function
        self.coordinates = coordinates
        self.mode_edge = mode_edge
        self.info_update_freq=info_update_freq
        self.length = length
        self.police_undercover=police_undercover
        self.criminal_detection_police=criminal_detection_police
        self.criminal_diverged_from_path=criminal_diverged_from_path
        self.edge_labels = edge_labels
        self.edge_color_list = edge_color_list
        self.police_agent = police_agent
        self.Police_entrance=Police_entrance
        self.criminal_agent = criminal_agent
        self.criminal_positions = criminal_positions
        self.crim_max_diverge=crim_max_diverge
        self.crim_tried_exits=crim_tried_exits
        self.capture = capture
        self.escape = escape
        self.criminal_information = criminal_information
        self.source_sink_node = source_sink_node
        self.departure_time = departure_time
        self.criminal_id = criminal_id
        self.police_id = police_id
        self.police_gone_undercover=police_gone_undercover
        self.metro_id = metro_id
        self.metro_starting_schedule = metro_starting_schedule
        self.run_hours = run_hours
        self.metro_at_station = metro_at_station
        self.graph = graph
        self.units = units
        self.init_call_delay = init_call_delay #in minutes
        self.P_multiple_at_station=P_multiple_at_station
        self.crim_pos = crim_pos
        self.crim_strat = crim_strat
        self.pol_strat = pol_strat
        self.pol_guarding= pol_guarding
        self.crim_bounded_rat = crim_bounded_rat
        self.crim_bound_rat_time=crim_bound_rat_time
        self.police_changed_goal=police_changed_goal
        self.running = running
        self.clocktime = clocktime
        self.crim_Mguard_percent = crim_Mguard_percent
        self.crim_Sguard_percent = crim_Sguard_percent
        self.criminal_seen_police=criminal_seen_police
        self.game_over=game_over
        self. criminal_goal = criminal_goal
        self.game_over_pos = game_over_pos
        self.criminal_passed_police = criminal_passed_police
        self.police_goal = police_goal
        self.police_start_pos = police_start_pos
        self.crim_loose_goal=crim_loose_goal

        self.police_goal={}
        self.police_start_pos={}
        self.police_changed_goal={}
        self.police_gone_undercover = False
        self.criminal_goal = None
        self.game_over_pos = None
        self.criminal_passed_police = 0
        self.criminal_diverged_from_path = 0
        self.crim_tried_exits = 0
        self.criminal_seen_police = 0

        # since all metro's drive every 10 minutes the starting time will be varied between 2 hours and 2 hours and 9 minutes,
        # to get an even spread of the scenarios across metro departures.
        #*2 bc each minute is two steps (bc each step is 30 sec)
        self.varying_start = random.choice(range(0, 9)) *2

        # criminal is initated after warm up time
        # warm_up time = 2hours=120min=240steps
        self.start_time= 240+self.varying_start

        # init_call_delay is the time it takes for the meldkamer to get a call that the crim escaped in metro
        # initiate_police-time is warm up time + init_call_delay (mulitplied by 2 because every step is 1/2 min)
        self.initiate_police_time = (self.start_time + (self.init_call_delay* 2))


        self.datacollector = DataCollector(model_reporters={
                "start_time": lambda m: m.start_time,
                "time": lambda m: m.clocktime,
                "capture": lambda m: m.capture,
                "escape": lambda m: m.escape,
                "game_over": lambda m: m.game_over,
                'criminal_goal' : lambda m: m.criminal_goal,
                'game_over_pos' : lambda m: m.game_over_pos,
                'criminal_passed_police' : lambda m: m.criminal_passed_police,
                'police_goal' : lambda m: m.police_goal,
                'police_start_pos' : lambda m: m.police_start_pos,
                'police_gone_undercover':lambda m:m.police_gone_undercover,
                'police_changed_goal': lambda m: m.police_changed_goal,
                'criminal_diverged_from_path': lambda m:m.criminal_diverged_from_path,
                'criminal_tried_exits': lambda m:m.crim_tried_exits,
                'criminal_seen_police':lambda m:m.criminal_seen_police,
                'crim_Sguard_percent': lambda m: m.crim_Sguard_percent,
                'crim_Mguard_percent': lambda m: m.crim_Mguard_percent,
                'init_call_delay': lambda m: m.init_call_delay,
                'info_update_freq': lambda m: m.info_update_freq,
                'criminal_detection_police': lambda m: m.criminal_detection_police


            }
            # ,
            # agent_reporters={
            #     "unique_id": lambda a: getattr(a, "unique_id", None)
            #     # "pos": lambda a: getattr(a, "pos", None),
            #     # "in_metro": lambda a: getattr(a, "in_metro", None)
            # }
            )


        # police_and_criminal_agents = [agent for agent in self.scheduler.agents if
        #                               isinstance(agent, (Police, Criminal))]
        # self.datacollector.collect(police_and_criminal_agents)

        self.datacollector.collect(self)
        self.run()

    def print2file(self, text):
        current_date = datetime.now().strftime('%d%m%Y')
        print_name = f"experiment/print_output_{current_date}.txt"
        with open(print_name, "a") as f:
            print(text, file=f)

    def run(self):
        self.G = nx.Graph()

        json_file_path = f"metro_network.json"

        # Check if the JSON file exists
        if os.path.exists(json_file_path):
            # Load the network from the JSON file
            with open(json_file_path, "r") as f:
                network_data = json.load(f)

            # Create an empty graph
            self.G = nx.Graph()

            # Add nodes and their attributes from loaded data
            for node, attributes in network_data["nodes"]:
                self.G.add_node(node, **attributes)

            # Add edges and their attributes from loaded data
            for u, v, data in network_data["edges"]:
                self.G.add_edge(u, v, **data)
        else:

            # make network with noeds
            for idx, row in dfNodes.iterrows():
                if row['type'] == 'nodelink' or row['type'] == 'node':
                    if row['metroline']=='A':
                        node_color="#9fc492"
                    elif row['metroline']=='B':
                        node_color="#f9f7bb"
                    elif row['metroline']=='C':
                        node_color="#f9b3bd"
                    elif row['metroline']=='D':
                        node_color="#c2f2f0"
                    elif row['metroline']=='E':
                        node_color="#b8c2e2"
                    if self.G.has_node(row['source']):
                        self.G.nodes[row['source']]['metroline'].append(row['metroline'])
                        self.G.nodes[row['source']]['source_sink_node'].append(row['sourcesinknode'])
                    else:
                        self.G.add_node(row['source'], source=row['source'], Location=row['Location'], label=row['Label'],
                                        metroline=[row['metroline']],train_station=row['train_station'],\
                                        mode_node=row['mode'], Exits=row['Exits'], function=row['function'],
                                        source_sink_node=[row['sourcesinknode']], \
                                        coordinates=(row['lat'], row['lon']),closest_metro=row['closest_metro'])

            #set positions
            self.coordinates = nx.get_node_attributes(self.G, 'coordinates')


            # function to calculate distance between two points on a sphere
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371  # radius of the Earth in kilometers
                dLat = math.radians(lat2 - lat1)
                dLon = math.radians(lon2 - lon1)
                a = (math.sin(dLat/2) * math.sin(dLat/2) +
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                     math.sin(dLon/2) * math.sin(dLon/2))
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                d = R * c
                return d

            #set edges between nodes

            for idx, row in dfNodes.iterrows():
                if (row['type'] == 'nodelink' or row['type'] == 'link'):
                    lat1=dfNodes['lat'].iloc[idx]
                    lon1 = dfNodes['lon'].iloc[idx]

                    #get target aka: station where edge has to go to
                    target = dfNodes['target'].iloc[idx]

                    #get the multiple locations in the csv where this station is mentioned
                    locations_target_in_df=([(dfNodes[col][dfNodes[col].eq(target)].index[i], dfNodes.columns.get_loc(col)) for col in dfNodes.columns for i in range(len(dfNodes[col][dfNodes[col].eq(target)].index))])
                    #get the index of the first time where the station is mentioned in the source column
                    # #this is also where its lat an lon are specified
                    index_target_in_df=locations_target_in_df[0][0]

                    #set lat and lon for the desitation of the edge
                    lat2 = dfNodes['lat'].iloc[index_target_in_df]
                    lon2 = dfNodes['lon'].iloc[index_target_in_df]
                    target=dfNodes['target'].iloc[idx]
                    length = haversine(lat1, lon1, lat2, lon2)
                    self.G.add_edge(row['source'], row['target'],mode_edge=row['mode'],
                               length="{:.2f}".format(length), metroline=row['metroline'])


            #get node attributes
            self.location = nx.get_node_attributes(self.G, 'Location')
            self.exits = nx.get_node_attributes(self.G, 'Exits')
            self.mode_node = nx.get_node_attributes(self.G, 'mode_node')
            self.node_color = nx.get_node_attributes(self.G, 'color')
            self.node_labels = nx.get_node_attributes(self.G, 'label')
            self.source_node = nx.get_node_attributes(self.G, 'source')
            self.function = nx.get_node_attributes(self.G, 'function')
            self.coordinates = nx.get_node_attributes(self.G, 'coordinates')
            self.source_sink_node = nx.get_node_attributes(self.G, 'source_sink_node')
            self.metroline = nx.get_node_attributes(self.G, 'metroline')
            self.train_station=nx.get_node_attributes(self.G, 'train_station')
            self.closest_metro=nx.get_node_attributes(self.G, 'closest_metro')
            # self.node_labels = {n: (d['Location'], d['exits']) for n, d in self.G.nodes(data=True)}
            print(10, self.coordinates)
            print(11, self.node_labels)

            # get edge attributes
            self.mode_edge = nx.get_edge_attributes(self.G, 'mode_edge')
            self.length = nx.get_edge_attributes(self.G, 'length')
            self.metroline = nx.get_edge_attributes(self.G, 'metroline')
            self.edge_labels = {(u, v): '' for u, v in self.G.edges()}
            # self.edge_labels = {(u, v): d['length'] for u, v, d, in self.G.edges(data=True)}
            print(12, self.edge_labels)
            #add weights to all the lengths, and convert it from str to float
            for u, v, data in self.G.edges(data=True):
                data['weight'] = float(data['length'])

            #set edge_color
            edge_colors = {'metro': '#000066', 'road': '#e6f2ff'}
            self.edge_color_list = []
            for u, v in self.G.edges():
                mode = self.G[u][v]['mode_edge']
                self.edge_color_list.append(edge_colors[mode])

            node_data = [(node, attributes) for node, attributes in self.G.nodes(data=True)]
            edge_data = [(u, v, data) for u, v, data in self.G.edges(data=True)]

            network_data = {"nodes": node_data, "edges": edge_data}

            # Save the network data to the JSON file
            with open(json_file_path, "w") as f:
                json.dump(network_data, f)


            # create a figure and axes to draw the graph
            # fig, ax = plt.subplots(figsize=(10, 10))
            #
            # # draw the initial graph
            # nx.draw(self.G, pos=self.coordinates, labels=self.node_labels, edge_color=self.edge_color_list, with_labels=True, node_size=100, font_size=14)
            # nx.draw_networkx_edge_labels(self.G, pos=self.coordinates, edge_labels=self.edge_labels, label_pos=0.5)#edge_labels=self.edge_labels,


            # Show the plot (optional)
            plt.show()

        # -------------------------------------------------------------------------
        """METRO SCHEDULE"""
        # -----------------------------------------------------------------------------------------

        #i create a metro agent and immediatly delete in order for the base
        # #scheduler to first always move the metro
        self.metro_agent = Metro('D',0,'Rotterdam Centraal', self.metro_id, self, G=self.G)
        self.scheduler.add(self.metro_agent)
        self.metro_id=self.metro_id+1

        # plt.show()

        #this code creates a dictionary with all the nodes that are a source
        self.metro_starting_schedule={}
        for e,node in enumerate(dfTimeTable['Station']):
            if dfTimeTable['sourcesinknode'].iloc[e]=='source':
                # the departuretimes of the metro is stored as key so that time step can loop through it
                # and see if timestep == departure time
                departure=dfTimeTable['departure'].iloc[e]
                #it create an at_station time at initialize time so that when the metro is made, it
                # is at the station 30 secons previous to its departure. now the criminal
                # has time to enter the metro at the first station, if they way
                at_station=departure-0.5
                if at_station in self.metro_starting_schedule:
                    self.metro_starting_schedule[at_station].append([node, dfTimeTable['metroline'].iloc[e], e])
                elif at_station not in self.metro_starting_schedule:
                    self.metro_starting_schedule[at_station] = \
                    [[node, dfTimeTable['metroline'].iloc[e], e]]


        #in this next forloop I continue the time table so that a new metro of each line it initiated every
        #10 minutes (which it does in Rotterdam). So for example if line B departs on the 7th minute
        # of every hour, in this loop i will make it depart at minute 7,17,27,37,47,57.

        times=list(self.metro_starting_schedule.keys())
        for i in range(1, 6): #6 bc there are 6x '10 minutes' in an hour
            for time in times:
                departing_metros=len(self.metro_starting_schedule[time])
                for metro in range(0,departing_metros):
                   new_time = time + (i * 10)
                   if new_time in self.metro_starting_schedule:
                       self.metro_starting_schedule[new_time].append(self.metro_starting_schedule[time][metro])
                       self.metro_starting_schedule[new_time].append(self.metro_starting_schedule[time][metro])
                   elif new_time not in self.metro_starting_schedule:
                       self.metro_starting_schedule[new_time] = \
                           self.metro_starting_schedule[new_time]=[self.metro_starting_schedule[time][metro]]

    # ------------------------------------------------------------------------
    """STEP FUNCTION """
    # -----------------------------------------------------------------------------------------


    def step(self):
        self.scheduler.step()

        if self.police_agent:
            if self.police_agent.under_cover==True:
                self.police_undercover=True


        if self.criminal_agent:
            if self.criminal_agent.capture == True:
                self.capture = True
                self.escape = False
                self.game_over=True
            if self.criminal_agent.escape == True:
                self.escape = True
                self.capture = False
                self.game_over = True
            self.datacollector.collect(self)
            self.update_agent_reporters()

        if self.game_over== False:
            #make sure that the simulation recognizes 120 time steps as 60 minutes as 1 hour
            #this, in order for the metro schedule to continue to work
            if self.scheduler.time<120:
                self.current_time =self.scheduler.time * self.time_step
                self.run_hours=0
            elif self.scheduler.time==120 or self.scheduler.time<240:
                self.current_time =(self.scheduler.time-120) * self.time_step
                self.run_hours=1
            elif self.scheduler.time==240 or self.scheduler.time<360:
                self.current_time =(self.scheduler.time-240) * self.time_step
                self.run_hours = 2
            elif self.scheduler.time==360 or self.scheduler.time<480:
                self.current_time = (self.scheduler.time - 360) * self.time_step
                self.run_hours = 3
            elif self.scheduler.time==480 or self.scheduler.time<600:
                self.current_time =(self.scheduler.time-480) * self.time_step
                self.run_hours = 4
            elif self.scheduler.time==600 or self.scheduler.time<720:
                self.current_time =(self.scheduler.time-600) * self.time_step
                self.run_hours = 5
            self.clocktime=str(self.run_hours)+'.'+str(self.current_time)
            if self.run_hours>=2:
                self.print2file(("Current time= ", self.clocktime))



            #initate criminal
            #after 2 hours of warm up time the metro's are all spread out among the stations.
            #if time is equal to start time the criminal can be initiated

            if self.scheduler.time==self.start_time:
                # create the criminal agent at a random position
                self.criminal_agent = Criminal(self.criminal_id,
                                               self,
                                               G=self.G,
                                               pos=self.crim_pos,
                                               criminal_strategy=self.crim_strat,
                                               bounded_rationality=self.crim_bounded_rat,
                                               Mguard_percent=self.crim_Mguard_percent,
                                               Sguard_percent=self.crim_Sguard_percent,
                                               bound_rat_time=self.crim_bound_rat_time,
                                               max_diverge=self.crim_max_diverge,
                                               loose_goal=self.crim_loose_goal)
                # draw the criminal agent
                self.scheduler.add(self.criminal_agent)
                self.criminal_id = self.criminal_id + 1
                # nx.draw_networkx_nodes(self.G, pos=self.coordinates, nodelist=[self.criminal_agent.pos], node_size=200,
                #                        node_color='red')

            #initate police agent
            #there is an inital delay in the time it takes for the meldkamer to get a call that the criminal escaped in the metro
            #this function initiates police after the delay time
            #the delay time is multiplied by 2 as steps are in 1/2 minutes, and the delay time is in minutes


            if self.scheduler.time==self.initiate_police_time:
                for i in range(0,self.units): #initate as many police units as there are resources availble = pre-determined
                    self.police_agent = Police(self.police_id,
                                               self,
                                               G=self.G,
                                               police_strategy=self.pol_strat,
                                               guarding= self.pol_guarding,
                                               P_multiple_at_station=self.P_multiple_at_station,
                                               info_update_freq=self.info_update_freq,
                                               entrance=self.Police_entrance)
                    self.scheduler.add(self.police_agent)
                    self.police_id = self.police_id + 1
                    # police_node = nx.draw_networkx_nodes(self.G, pos=self.coordinates,
                    #                                      nodelist=[self.police_agent.pos], node_size=200,
                    #                                      node_color='yellow')


            #iniiate a metro agent
            if self.current_time in self.metro_starting_schedule:
                departing_metros = len(self.metro_starting_schedule[self.current_time ])
                for metro in range(0, departing_metros):
                    start_point=self.metro_starting_schedule[self.current_time][metro][0]
                    line=self.metro_starting_schedule[self.current_time][metro][1]
                    index = self.metro_starting_schedule[self.current_time][metro][2]
                    self.metro_agent=Metro(line=line,index=index,start_point=start_point,unique_id=self.metro_id, \
                                           model=self, G=self.G)
                    self.scheduler.add(self.metro_agent)
                    self.metro_id = self.metro_id + 1

            #keep track of which metros are at stations and thus vailable to enter for criminal
            if self.current_time >= min(self.metro_starting_schedule) or self.run_hours > 0:
                metro_positions = []
                self.metro_at_station = {}
                for agent in self.scheduler.agents:
                    if isinstance(agent, Metro):
                        if agent.at_station == True:
                            self.metro_at_station[agent.pos] = [agent.line, agent.unique_id]
            if self.scheduler.time>=self.start_time:
                self.print2file(("metro's at station=", self.metro_at_station))


            #plot graph
            if self.graph==True:
                fig, ax = plt.subplots(figsize=(10,10))
                nx.draw(self.G, pos=self.coordinates, labels=self.node_labels, edge_color=self.edge_color_list, with_labels=True, node_size=100, font_size=14)
                nx.draw_networkx_edge_labels(self.G, pos=self.coordinates, edge_labels=edge_labels, label_pos=0.5)

                # create new position for criminal based upon neighbouring nodes and draw

                nx.draw_networkx_nodes(self.G, pos=self.coordinates, nodelist=[self.criminal_agent.pos], node_size=200,
                                       node_color='red')
                #If model run time is bigger than the earliest
                # departure time, there must be a metro agent. so, only if time> min(departure time)
                # can a metro agent be drawn, therefore this if statement

                if self.current_time>= min(self.metro_starting_schedule):
                    metro_positions =[]
                    self.metro_at_station={}
                    for agent in self.scheduler.agents:

                        if isinstance( agent,Metro):
                            if agent.at_station==False:
                                edge_color = ['none' if not (u == agent.pos and v == agent.next_node) else 'green' for u, v in
                                          self.G.edges()]
                                nx.draw_networkx_edges(self.G, pos=self.coordinates, edgelist=self.G.edges(), width=3.0,
                                                   edge_color=edge_color)


                            if agent.at_station==True:
                                self.metro_at_station[agent.pos]=[agent.line,agent.unique_id]

                        metro_positions.append(agent.pos)

                    nx.draw_networkx_nodes(self.G, pos=self.coordinates, nodelist=metro_positions, node_size=100,
                                           node_color='green')



                #create new position for police based upon neighbouring nodes and draw
                # police_location=self.police_agent.move(self.criminal_positions, self.criminal_information)
                # police_node = nx.draw_networkx_nodes(self.G, pos=self.coordinates, nodelist=[police_location], node_size=200, node_color='yellow')


                # show the plot
                # if self.current_time in [10,20,30,40,50,60]:
                #     plt.title(self.current_time)
                #     plt.show()

            # for agent in self.scheduler._agents.values():
            #     for item in self.datacollector.agent_reporters:
            #         print(9999999999,getattr(agent, item))
            #     for var, reporter in self.datacollector.agent_reporters.items():
            #         print(var,reporter)
            #         print(44444444,getattr(agent, var))
            # if criminal_location == police_location:
            #     print('Police caught the criminal')
            #     self.capture=True


    def update_agent_reporters(self):
        for agent in self.scheduler.agents:
            if isinstance(agent, Police):
                self.police_goal[agent.unique_id]=agent.goal
                self.police_start_pos[agent.unique_id]=agent.start_pos
                self.police_changed_goal[agent.unique_id]=agent.changed_goal
                self.police_gone_undercover=agent.under_cover
            elif isinstance(agent, Criminal):
                self.criminal_goal=agent.goal
                self.game_over_pos=agent.game_over_pos
                self.criminal_passed_police=agent.passed_police
                self.criminal_diverged_from_path=agent.total_diverged_from_path
                self.crim_tried_exits=agent.tried_exits
                self.criminal_seen_police=agent.seen_police

        # criminal goal
        # police goal
        # - police starting position
        # - capture / ecape
        # - time
        # capture / escape\
        # - times criminal passed police w / ocaptur\
        # - capture position