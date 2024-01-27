import random
import time
from mesa import Agent
from datetime import datetime
from mesa.time import BaseScheduler
import networkx as nx
import pandas as pd
from mesa.datacollection import DataCollector


# ------------------------------------------------------------------------
"""In this code: 
                CLASS: SOURCE (AGENT) 
                CLASS: METRO (AGENT)
                SET_PATH 
                REMOVE METRO AGENT"""
# -----------------------------------------------------------------------------------------


# ------------------------------------------------------------------------
"""CLASS: METRO (AGENT)"""
# -----------------------------------------------------------------------------------------
dfTimeTable = pd.read_csv('data/MetroTimetTable.csv')
class Metro(Agent):
    def __init__(self, line,index, start_point, unique_id, model, G, speed=1333.33,\
                 pos=(0,0), at_sink=False, next_node=None, node_index=None, \
                 departure_time=None, next_sink=False,arrival_time=None, at_station=False,
                 station_busyness=0):
        super().__init__(unique_id, model)
        self.G = G
        self.type = "Metro"
        self.pos = pos
        self.speed = speed  # = 80 km/h = 1333.333 m/min
        self.time = time
        self.generated_at_step = (model.scheduler.steps)*0.5
        self.at_sink=at_sink
        self.line=line
        self.index=index
        self.start_point=start_point
        self.pos=start_point
        self.unique_id=unique_id
        self.next_node = next_node
        self.node_index = node_index
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.at_station=at_station
        self.next_sink=next_sink
        self.station_busyness=station_busyness
        self.mean_busyness = dfTimeTable['Drukte'].mean()
        self.set_path()

    def print2file(self, text):
        current_date = datetime.now().strftime('%d%m%Y')
        print_name = f"experiment/print_output_{current_date}.txt"
        with open(print_name, "a") as f:
            print(text, file=f)

        # ------------------------------------------------------------------------
        """SET_PATH"""
    # -----------------------------------------------------------------------------------------

    def set_path(self):
        self.at_sink=False
        self.pos=self.start_point
        self.at_station = True
        self.station_busyness=random.uniform((dfTimeTable['Drukte'].iloc[self.index])-1,(dfTimeTable['Drukte'].iloc[self.index])+1)
        self.next_node = dfTimeTable['Station'].iloc[self.index+1]
        self.departure_time = self.model.current_time+0.5
        self.arrival_time = +0  # -0.5 because mesa only uses half a units nd not minutes


    # ------------------------------------------------------------------------
    """STEP"""
    # -----------------------------------------------------------------------------------------

    def move(self):
        #i created an metro agent to let the base scheduler know the metro is the first one to move.
        # its unique_id = 999 so when an agent with that ID comes in it should be removed straight away
        if self.unique_id == 999:
            self.sink()

        #remove metro vehicle as it is at end stop
        if self.next_sink==True:
            self.sink()

        #set the metro to go to sink at next stop
        if dfTimeTable['sourcesinknode'].iloc[self.index] == 'sink':
            self.next_sink = True

        #make train arrive at station 30 secondds before it should depart from this station
        if self.model.current_time == self.arrival_time:
            self.pos=self.next_node
            self.index = self.index + 1
            self.at_station = True
            self.station_busyness=random.uniform((dfTimeTable['Drukte'].iloc[self.index])-1,(dfTimeTable['Drukte'].iloc[self.index])+1)
            #sketchy way to fix that the metro index does not exceed allowed indx number
            if self.index != 243:
                self.next_node = dfTimeTable['Station'].iloc[self.index + 1]
            elif self.index == 243:
                self.next_node = 'Den Haag Centraal'

        #let the metro leave at the correct time as gotten from the time table online
        elif self.model.current_time == self.departure_time:
            self.pos=self.pos
            self.at_station=False
            #set new departure time
            #sketchy way to fix that the metro index does not exceed allowed indx number
            if self.index != 243:
                self.departure_time = self.model.current_time + dfTimeTable['trip_duration'].iloc[self.index + 1]
            elif self.index == 243:
                self.departure_time=self.model.current_time + 2
            #set arrival time at next station to be 30 seconds before departure time
            self.arrival_time = int(self.departure_time) - 0.5  # -0.5 because mesa only uses half a units nd not minutes
        else:
            self.at_station = False
            self.pos=self.pos

        #transfer the arrival and departure times to the next hour
        if self.arrival_time >= 60:
            self.arrival_time=self.arrival_time-60
        if self.departure_time >= 60:
            self.departure_time=self.departure_time-60



        # ------------------------------------------------------------------------
        """REMOVE METRO AGENT"""
        # ----------------------------------------------------------------------------------------

    def sink(self):
        self.model.scheduler.remove(self)



    def step(self):
        if self.model.game_over==False:
            self.move()


