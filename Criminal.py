import random
import time
from mesa import Agent
import networkx as nx
from collections import deque
from Police import Police
from datetime import datetime
# ------------------------------------------------------------------------
"""In this code: 
                START/ INIT 
                STRATEGIES
                SET PATH 
                MOVES 
                    (1)enter a metro
                        NO PANIC 
                        PANIC
                        
                    (2) wait on station for correct metro
                    (3)get off metro and leave station
                    (4) stay in train because going in right direction
                    (5) get off metro and wait at station for next
                    (6) sit in train because it is driving.
                CHECK BUSYNESS
                STEP"""
# -----------------------------------------------------------------------------------------


class Criminal(Agent):
    diverged_stations=[]
    def __init__(self, unique_id, model, G, pos, criminal_strategy, bounded_rationality,\
                 Mguard_percent, Sguard_percent, bound_rat_time, max_diverge,loose_goal, positions={}, round = 0,
                 goal=None, path=[], distance=0, speed = 1333.333, time=None,
                 in_metro=False,metro_id=None,G_policeless=None,metro_agent=None, criminal_positions={}, previous_station = None,\
                 upcoming_station=None, capture=False, escape=False,turn_around=False, \
                 passed_police=0, too_empty=False,very_busy=False, game_over_pos=None,seen_police=0,\
                 potential_metro_agent=None,previous_busyness=None, diverged_from_path=0,total_diverged_from_path=0, wait_steps=0,\
                 exit_chosen=None,tried_exits=0, interactie=False,sinksources=[]):

        super().__init__(unique_id, model)
        self.G = G
        self.type='Criminal'
        self.pos = pos
        self.criminal_strategy=criminal_strategy
        self.goal = goal  # intended goal where agents wants to go
        self.path = path
        self.distance = distance
        self.speed = speed  # = 80 km/h = 1333.333 m/min
        self.time = time
        self.criminal_positions=criminal_positions
        self.in_metro=in_metro
        self.metro_id=metro_id
        self.metro_agent=metro_agent
        self.upcoming_station=upcoming_station
        self.previous_station=previous_station
        self.capture=capture
        self.escape=escape
        self.turn_around=turn_around
        self.Mguard_percent=Mguard_percent
        self.Sguard_percent=Sguard_percent
        self.passed_police=passed_police
        self.seen_police=seen_police
        self.G_policeless=G_policeless
        self.bounded_rationality=bounded_rationality
        self.bound_rat_time=bound_rat_time
        self.very_busy=very_busy
        self.too_empty=too_empty
        self.game_over_pos=game_over_pos
        self.max_diverge=max_diverge
        self.potential_metro_agent=potential_metro_agent
        self.previous_busyness=previous_busyness
        self.diverged_from_path=diverged_from_path
        self.total_diverged_from_path=total_diverged_from_path
        self.wait_steps=wait_steps
        self.exit_chosen=exit_chosen
        self.loose_goal=loose_goal
        self.tried_exits=tried_exits
        self.G_policeless = self.G.copy()
        self.police_nodes = []
        self.interactie=interactie
        self.sinksources=sinksources
        self.sinksources=['De Akkers', "Binnenhof","Hoek van Holland Strand", "Den Haag Centraal", 'Nesselande', 'De Terp', 'Slinge']
        for node in self.G.nodes('function'):
            if node[1] == 'police_bureau':
                self.police_nodes.append(node[0])

        self.G_policeless.remove_nodes_from(self.police_nodes)
        self.start()

    def print2file(self, text):
        current_date = datetime.now().strftime('%d%m%Y')
        print_name = f"experiment/print_output_{current_date}.txt"
        with open(print_name, "a") as f:
            print(text, file=f)

# ------------------------------------------------------------------------
    """START/ INIT"""
# -----------------------------------------------------------------------------------------
    def start (self):
        # criminal_nodes=[]
        # for node in self.G.nodes('function'):
        #     if node[1] == 'metro_station':
        #         criminal_nodes.append(node[0])
        # pos = random.choice(criminal_nodes)
        # self.pos=pos
        self.criminal_positions={}
        self.get_strategy()

# -------------------------------------------------------------------------
    """STRATEGIES"""
# -----------------------------------------------------------------------------------------
    def get_strategy(self):
        # criminal_strategies= ['random','furthest','to_train']
        # self.criminal_strategy=random.choice(criminal_strategies)
        # self.criminal_strategy='furthest'
        if self.criminal_strategy== 'furthest':
            self.furthest_strategy()
        elif self.criminal_strategy== 'random':
            self.random_strategy()
        elif self.criminal_strategy== 'to_train':
            self.to_train_strategy()

    def random_strategy(self):
        self.goal = random.choice(list(self.G_policeless.nodes()))
        self.print2file(('The strategy of the criminal is',self.criminal_strategy,\
                    '. The goal of this strategy is to get to node', self.goal))
        self.set_path()

    def furthest_strategy(self):
        shortest_path_lengths = nx.shortest_path_length(self.G_policeless, self.pos)
        sorted_nodes = sorted(shortest_path_lengths, key=shortest_path_lengths.get, reverse=True)
        # Select the furthest 5 nodes
        furthest_nodes = sorted_nodes[:5]
        self.goal=random.choice(furthest_nodes)
        self.print2file(('The strategy of the criminal is', self.criminal_strategy, \
              '. The goal of this strategy is to get to node', self.goal))
        self.set_path()

    def to_train_strategy(self):
        metro_with_train_stat = []
        #select only the metro stations which also are a train station so the crim can transfer to train network
        for node in self.G_policeless.nodes():
            if self.G_policeless.nodes[node]['train_station'] == 'yes': #select only metro which has 'yes' for having train.
                metro_with_train_stat.append(node)

        #counts how many station are in between the station with train and current criminal pos
        station_count = [nx.shortest_path_length(self.G_policeless, self.pos, station) - 1 for station in
                         metro_with_train_stat] #-1 bc I don't want to include self.pos in count

        #metro_with_metro_staton list is updated using zip to remove the stations which have <5 distance between
        #station and self.pos in the station_count list (zip takes list pos from station_count and removes this in
        #position also in the metro_with_train_station list)
        metro_with_train_stat = [station for station, count in zip(metro_with_train_stat, station_count) if count >= 5]
        #items <5 distance removed from station_count
        station_count = [count for count in station_count if count >= 5]

        # if there are no stations with distance > 5 stations, game over, bc game does not suffice assumption crim needs to travel >5 stops
        if len(metro_with_train_stat)==0:
            self.capture= True
            self.print2file(('CRIM INFO: no station which is further than 5 stops. Game over.'))

        else:
            self.goal = random.choice(metro_with_train_stat)
            self.goal='Rotterdam centraal'
            self.print2file(('The strategy of the criminal is', self.criminal_strategy, \
                 '. The goal of this strategy is to get to node', self.goal))
            self.set_path()

# -------------------------------------------------------------------------
        """SET PATH"""
# ---------------------------------------------------------------------------------------

    def set_path(self):
        #if statement making sure this is their first time chosing a path, because they have to make an initial path
        #regardless whether they are in panic

        if self.too_empty==False and self.very_busy==False:
            self.path = nx.shortest_path(self.G_policeless, self.pos, self.goal)
            #make sure criminal travels at least 5 stops
            if len(self.path)<5:
                if self.criminal_strategy=='random':
                    self.random_strategy()
                elif self.criminal_strategy=='else':
                    self.random_strategy()
            self.previous_station = self.pos
            self.print2file(('init self.path for criminal=', self.path))

        #if they are in panic and have not moved accordingly to original path aka gotten off/ on too early/late
        elif self.too_empty == True or self.very_busy == False:
            #using self.upcoming_station becuase when this is done the criminal has just gotten on a metro. Meaning,
            #it is driving and therefore upcoming station is used to calculate path to goal (because self.pos is still previous station)
            self.path = nx.shortest_path(self.G_policeless, self.upcoming_station, self.goal)
            self.path.insert(0,self.upcoming_station)
            self.diverged_from_path=self.diverged_from_path+1
            self.total_diverged_from_path=self.total_diverged_from_path+1
            self.print2file(('new self.path for criminal=', self.path))

        elif self.too_empty == False or self.very_busy == True  or self.turn_around==True:
            #using self.pos because when self.very_busy== True the criminal will have gotten off the metro
            #too early and therefore has a self.pos which can be used to determine a new path to goal
            self.path = nx.shortest_path(self.G_policeless, self.pos, self.goal)
            self.diverged_from_path = self.diverged_from_path + 1
            self.total_diverged_from_path = self.total_diverged_from_path + 1
            self.turn_around = False
            self.print2file(('new self.path for criminal=', self.path))

        else:
            self.print2file(('HELP! check at set.path() criminal what is going wrong'))




# -------------------------------------------------------------------------
    """MOVES"""
# -----------------------------------------------------------------------------------------
    def move(self):
        if self.model.scheduler.time >= self.model.initiate_police_time:
            guarded_stations, guarded_metros = Police.get_all_guarded(cls=Police)
        else:
            guarded_stations={}
            guarded_metros=[]

  # INTERACTIE
        #in paniek raken. If crim has seen police 3x, he starts panicing
        if self.seen_police >self.bound_rat_time and self.interactie==True:
            self.bounded_rationality=True


        #the criminal can make 6 moves: (1)enter a metro, (2) wait on station for correct metro, (3)get off metro and leave station
        # (4) stay in train because going in right direction,  (5) get off metro and wait at station for next
        # and (6) sit in train because it is driving.


    #NOTE
    # MOVE 1: enter a metro
        #if criminal is at station but not in a train
        if self.in_metro==False:
            #criminal is at a platform. only if police is also at metro platform he can be caught so station guarding isnt checked
            if self.pos in guarded_metros:
                self.capture = random.uniform(0, 100) <= self.Mguard_percent
                if self.capture == True:
                    self.print2file(('CRIM INFO: police caught criminal'))
                    self.print2file((('-----------------')*10,self.capture))
                    self.escape = False
                    self.game_over_pos = self.pos
                else:
                    self.print2file(('CRIM INFO: criminal passed police'))
                    self.passed_police = self.passed_police + 1
                    self.capture = False
 # INTERACTIE
                    detection_police = random.uniform(0, 100) <= (self.model.criminal_detection_police)
                    if self.interactie==False:
                        detection_police=False
                    if self.model.police_undercover==False and detection_police==True:
                        self.seen_police=self.seen_police+1



            ###NO PANIC##
            if self.pos in self.model.metro_at_station:
                potential_metro_id=self.model.metro_at_station[self.pos][1]
                #call the agent so we can get more of its attributes, such as 'next node'
                self.potential_metro_agent = next((a for a in self.model.scheduler.agents if a.unique_id == potential_metro_id), None)
                potential_next_node=self.potential_metro_agent.next_node
  # INTERACTIE
                detection_police = random.uniform(0, 100) <= (self.model.criminal_detection_police)
                if self.interactie == False:
                    detection_police = False

                # move 1: if the metro that is coming has the same next node, then criminal 'enter' the metro
                # print('123', potential_next_node, self.path)
                if potential_next_node==self.path[1]:
                    #if the metro which is at the same station as the criminal is, AND if the metro has the same next node, criminal can enter metro
                    self.metro_id=potential_metro_id
                    self.in_metro=True
                    self.upcoming_station = potential_next_node
                    self.pos = self.pos
                    self.print2file(('CRIM INFO: Entering metro to next station:', self.upcoming_station))
                    self.previous_station=self.pos
                    self.metro_agent = next((a for a in self.model.scheduler.agents if a.unique_id == self.metro_id), None)


                ###PANIC##
                ##if crminal is in panic, they may still enter the metro
                #this is elif bc if the criminal has entered metro in previous if statement, it should not try this if statement
                #or get on metro if you see the police
                elif self.bounded_rationality==True and self.pos not in self.sinksources or\
                        self.pos in guarded_metros and detection_police==True  and self.pos not in self.sinksources:
                    self.check_busyiness(self.potential_metro_agent)
                    if self.too_empty==True:
                        #if in panic and the station is below threshold for allowed emptiness, still enter metro
                        # it can be that the criminal lands in a cyclus where station after station is very_busy or too_empty. If this
                        # is the case, he will keep diverging from path, so he will never arrive. However, if a criminal is in panic
                        # he will also want to get home, so after three times diverging path, he will disregard the very_busy and too_empty
                        # and follow its path
                        if self.diverged_from_path >= self.max_diverge:
                            self.print2file(('CRIM INFO: 012,diverged from path =', self.diverged_from_path,'so following original path path'))
                            self.diverged_from_path = 0
                        elif self.potential_metro_agent not in Criminal.diverged_stations:
                        #he can only make a panic-action once per station, otherwise criminal can not get to goal
                            self.metro_id = potential_metro_id
                            self.in_metro = True
                            self.upcoming_station = potential_next_node
                            self.pos = self.pos
                            self.print2file(('AAACRIM INFO: Due to panic entering metro to next station:', self.upcoming_station))
                            self.previous_station = self.pos
                            self.metro_agent = next((a for a in self.model.scheduler.agents if a.unique_id == self.metro_id),
                                                    None)
                            Criminal.diverged_stations.append(self.pos) #append metro station to diverged_stations list
                            self.set_path()
# INTERACTIE
                        if self.pos in guarded_metros and detection_police==True:
                            self.passed_police=self.passed_police+1
                            detection_police = random.uniform(0, 100) <= (self.model.criminal_detection_police)
                            if self.interactie == False:
                                detection_police = False
                            if self.model.police_undercover == False and detection_police == True:
                                self.seen_police = self.seen_police + 1


    #NOTE
    # MOVE 2:  wait on station for correct/next metro
            #move 2: wait till next step when hopefully there is a metro in the right direction
            else: #if crimiminal is not in metro and no metro in correct line, stay on station and wait for next

                if self.pos in guarded_metros:
                    self.capture = random.uniform(0, 100) <= (self.Sguard_percent)
                    if self.capture == False:
                        self.escape = True
                        self.print2file(('CRIM INFO: Police did not see mee, i escaped'))
                        self.game_over_pos = self.pos
                        self.passed_police=self.passed_police+1
    # INTERACTIE
                        detection_police = random.uniform(0, 100) <= (self.model.criminal_detection_police)
                        if self.interactie == False:
                            detection_police = False
                        if self.model.police_undercover == False and detection_police == True:
                            self.seen_police = self.seen_police + 1
                    elif self.capture == True:
                        self.escape = False
                        self.print2file(('CRIM INFO: I got caught by surveying metro platform'))
                        self.game_over_pos = self.pos
                else:
                    self.pos=self.pos




    #NOTE
    # MOVE 3: get off metro and leave station
        #criminal is in metro
        elif self.in_metro==True:
            self.check_busyiness(self.metro_agent)
            #if metro has arrived at next station
            if self.metro_agent.at_station==True and self.metro_agent.pos == self.upcoming_station:
                # self.pos = self.path.pop(0)
                self.pos=self.metro_agent.pos
                self.path.remove(self.metro_agent.pos)


                # if criminal is at station and police is also surveying metro platform, capture is possible
                if self.pos in guarded_metros:
                    self.capture = random.uniform(0, 100) <= self.Mguard_percent
                    if self.capture == True:
                        self.print2file(('CRIM INFO: police caught criminal'))
                        self.print2file((('-----------------') * 3, self.capture))
                        self.escape=False
                        self.game_over_pos=self.pos
                    else:
                        self.print2file(('CRIM INFO: criminal passed police'))
                        self.passed_police = self.passed_police + 1
  # INTERACTIE
                        detection_police = random.uniform(0, 100) <= (self.model.criminal_detection_police)
                        if self.interactie == False:
                            detection_police = False
                        if self.model.police_undercover == False and detection_police == True:
                            self.seen_police = self.seen_police + 1
                        self.capture=False

                #move 3: get off an leave because criminal made it to goal
                if len(self.path)==1:
                    # pick an exit
                    ###PANIC##
                    if self.bounded_rationality == True:
                        exits_value = self.G_policeless.nodes[self.pos]['Exits']
                        if self.bounded_rationality == True or exits_value <= 2:
                            self.exit_chosen = random.choice(['a', 'b'])
                        ###NO PANIC##
                        elif self.bounded_rationality == False and exits_value > 2:
                            side_entrances = int(exits_value - 2)  # subtract 2 bc the 2 first entrances are main and we are picking a side entrance here
                            self.exit_chosen = random.choice([chr(ord('c') + i) for i in range(side_entrances)])



                    self.print2file(('CRIM INFO: I made it to end station'))
                    #checking if police is surveying station
                    if self.pos in guarded_stations and self.exit_chosen in guarded_stations[self.pos]:
                        if self.model.police_undercover == False:
                            exits_value = self.G_policeless.nodes[self.pos]['Exits']
  # INTERACTIE
                            detection_police = random.uniform(0, 100) <= (self.model.criminal_detection_police)
                            if self.interactie == False:
                                detection_police = False
                            if detection_police == True:
                                self.turn_around = True
                                self.passed_police = self.passed_police + 1
                                self.seen_police=self.seen_police+1
                                if self.loose_goal==True:
                                    self.goal = random.choice(list(self.G_policeless.neighbors(self.goal)))
                                    self.set_path()
                                elif self.loose_goal== False:
                                    self.tried_exits=1
                                    while self.tried_exits < exits_value and detection_police==True and self.exit_chosen in guarded_stations[self.pos]:
                                        self.exit_chosen = random.choice([chr(ord('c') + i) for i in range(int(exits_value))])
                                        detection_police = random.uniform(0, 100) <= (self.model.criminal_detection_police) * (self.Sguard_percent * len(
                                                               guarded_stations[self.pos]) / exits_value)
                                        self.tried_exits = self.tried_exits+1
                                    if self.exit_chosen in guarded_stations[self.pos]:
                                        self.capture = False
                                        self.escape = True
                                        self.print2file(('CRIM INFO: I found an exit which wasnt guarded, I escaped'))
                                        self.game_over_pos = self.pos

                                    elif self.tried_exits==exits_value or detection_police==False:
                                        self.capture = random.uniform(0, 100) <= (self.Sguard_percent)
                                        if self.capture == False:
                                            self.escape = True
                                            self.print2file(('CRIM INFO: I tried all the exits, I escaped'))
                                            self.game_over_pos = self.pos
                                        elif self.capture == True:
                                            self.escape = False
                                            self.print2file(('CRIM INFO: I tried all the exits and got caught'))
                                            self.game_over_pos = self.pos

                            if detection_police==False:
                                exits_value = self.G_policeless.nodes[self.pos]['Exits']
                                self.capture = random.uniform(0, 100) <= (self.Sguard_percent)
                                if self.capture == False:
                                    self.escape=True
                                    self.print2file(('CRIM INFO: No police at station, I escaped'))
                                    self.game_over_pos=self.pos
                                elif self.capture == True:
                                    self.escape=False
                                    self.print2file(('CRIM INFO: I got caught by surveying station exits'))
                                    self.game_over_pos=self.pos

                        if self.model.police_undercover==True:
                            self.capture = random.uniform(0, 100) <= (self.Sguard_percent)
                            if self.capture == False:
                                self.escape = True
                                self.print2file(('CRIM INFO: Police did not see mee, i escaped'))
                                self.game_over_pos = self.pos
                            elif self.capture == True:
                                self.escape = False
                                self.print2file(('CRIM INFO: I got caught by surveying station exits'))
                                self.game_over_pos = self.pos


                    #checking if police is at surveying the metro
                    if self.pos in guarded_metros:
                        self.capture = random.uniform(0, 100) <= self.Mguard_percent
                        if self.capture==False:
                            self.escape=True
                            self.print2file(('CRIM INFO: No police at station, I escaped'))
                            self.game_over_pos = self.pos
                        else:
                            self.escape=False
                            self.capture=True
                            self.print2file(('CRIM INFO: I got caught by surveying metro exits'))
                            self.game_over_pos = self.pos

                    # if no guard at station and correct exit and no guard at metro, criminal escaped
                    if (self.pos not in guarded_stations and self.pos not in guarded_metros) or \
                        (self.pos in guarded_stations and self.exit_chosen not in guarded_stations[self.pos]):
                        self.capture=False
                        self.escape=True
                        self.print2file(('CRIM INFO: I escaped at final distination, no surveillance!'))
                        self.game_over_pos = self.pos


    #NOTE
    # MOVE 4: stay in train because going in right direction or stay due to panic
    # MOVE 5: get off metro and wait at station for next
                #move 4: stay in train because it is going in the right direction
                #move 5: if metro next stop is not preferred then the criminal will get off metro and wait at stop

                elif self.metro_agent.next_node == self.path[1]:
        #NOTE
        # Panic - move 5 - get off metro
                    ###PANIC##
                    if self.bounded_rationality==True and self.very_busy==True and self.diverged_from_path < self.max_diverge\
                            and self.pos not in Criminal.diverged_stations:
                    #criminal is in panic and current station is busy, so he will get out regardless
                    #unless he has diverged from path three times, then follow original path
                        self.in_metro = False
                        Criminal.diverged_stations.append(self.pos)
                        self.print2file(('CRIM INFO: Due to panic get off metro and wait for next'))
                        self.set_path()
                    #if criminal is not in panic he will continue self.path and stay on train
        #NOTE
        # No panic - move 4 - stay in metro
                    ###NO PANIC##
                    else:
                         if self.diverged_from_path > self.max_diverge:
                            self.print2file(('CRIM INFO: diverged from path =', self.diverged_from_path, 'so following original path path'))
                         self.in_metro=True
                         self.previous_station=self.pos
                         self.upcoming_station = self.metro_agent.next_node
                         self.pos= self.pos
                         self.print2file(('CRIM INFO: staying in metro at:' , self.metro_agent.pos,' to next station:', self.upcoming_station))

                elif self.metro_agent.next_node != self.path[1]:
                #if next metro stop is not desired, get off
        #NOTE
        # Panic - move 4 - stay on metro
                    ###PANIC##
                    if self.bounded_rationality==True and self.too_empty==True and self.diverged_from_path<self.max_diverge\
                            and self.pos not in Criminal.diverged_stations:
                        #unless criminal is in panic and current station is too empty, stay on
                        self.in_metro = True
                        self.previous_station = self.pos
                        self.upcoming_station = self.metro_agent.next_node
                        Criminal.diverged_stations.append(self.pos)
                        self.print2file(('CRIM INFO: due to panic staying in metro at:', self.metro_agent.pos, ' to next station:',
                              self.upcoming_station))
                        self.set_path()

        #NOTE![](experiment/experiment analysis/Figures/noint_mean_cap_pivot.png)
        # No panic - move 5 - get off metro
                    ###NO PANIC##
                    else:
                        if self.diverged_from_path>=self.max_diverge:
                            self.diverged_from_path=0
                        self.in_metro = False
                        self.print2file(('CRIM INFO: get off metro and wait till other line'))




    #NOTE
    # MOVE 6: sit in train because it is driving
            #move 6: Sit in train while train in driving. Then its position does not move
            elif self.metro_agent.at_station==False:
                self.pos='driving'
                self.print2file(('CRIM INFO: in metro (still driving) to next station:', self.upcoming_station))
        self.criminal_positions[self.model.current_time]=[self.pos]
        self.print2file(('CRIM INFO: criminal pos', self.pos))

 # -------------------------------------------------------------------------
        """CHECK BUSYNESS"""
  #-----------------------------------------------------------------------------------------
    def check_busyiness(self,agent):
        threshold_empty=agent.mean_busyness*0.75
        threshold_busy=agent.mean_busyness*1.25
        self.very_busy=False
        self.too_empty=False

        if agent.station_busyness >= threshold_busy:
            self.very_busy=True
        elif agent.station_busyness<=threshold_empty:
            self.too_empty=True
        self.print2file(('CRIM INFO: busyness mean:',agent.mean_busyness,\
              'threshold empty',threshold_empty,'threshold busy',threshold_busy,
              'station busyness', agent.station_busyness))

        # -------------------------------------------------------------------------
        """STEP FUNCTION"""
    # -----------------------------------------------------------------------------------------
    def step(self):
        if self.model.game_over==False and self.wait_steps==0:
            self.move()
        elif self.wait_steps>0:
            self.wait_steps=self.wait_steps-1

    def diverged(cls):
        return cls.diverged_stations





