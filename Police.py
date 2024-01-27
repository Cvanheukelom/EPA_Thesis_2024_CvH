import random
import time
from mesa import Agent
import networkx as nx
from datetime import datetime
import itertools
import networkx as nx
from scipy.spatial.distance import euclidean


# ------------------------------------------------------------------------
"""In this code: 
                START/ INIT 
                GET CALL
                STRATEGIES 
                SET POSITION
                MOVES 
                STEP
                GUARDED"""

# -----------------------------------------------------------------------------------------

class Police (Agent):
    guarded_stations={}
    guarded_metros=[]
    police_goals={}

    def __init__(self,unique_id, model, G,  police_strategy,guarding,P_multiple_at_station,\
                 info_update_freq, entrance,entrance_covered=None, criminal_passed_police=0,\
                 pos=(0,0), goal=None, path=[], distance=0,changed_goal=0,under_cover=False,\
                 speed = 833.33, driving_time=None, criminal_information=None, criminal_agent=None, \
                 arrival_time=None,start_pos=None, done=False, new_goal=False, arrived=False, top_closest=[], furthest_nodes=[],\
                 nodes_set_far=None,interactie=False):
        super().__init__(unique_id, model)
        self.G=G
        self.type='Police'
        self.pos=pos #current position
        self.police_strategy=police_strategy
        self.P_multiple_at_station=P_multiple_at_station
        self.goal=goal #intended goal where agents wants to go
        self.path=path
        self.distance=distance
        self.top_closest=top_closest
        self.furthest_nodes=furthest_nodes
        self.speed=speed #= 50 km/h = 833,33 m/min
        self.driving_time=driving_time
        self.arrival_time=arrival_time
        self.changed_goal=changed_goal
        self.criminal_information=criminal_information
        self.guarding=guarding
        self.criminal_agent=criminal_agent
        self.start_pos=start_pos
        self.entrance=entrance
        self.entrance_covered=entrance_covered
        self.done=done
        self.nodes_set_far=nodes_set_far
        self.under_cover=under_cover
        self.criminal_passed_police=criminal_passed_police
        self.arrived=arrived
        self.new_goal=new_goal
        self.info_update_freq=info_update_freq
        self.interactie=interactie
        self.G_policeless = self.G.copy()
        # nodes_to_remove = ['Politie Stadhuis', 'Politie Rotterdam maashaven', 'Politie Marconiplein', 'Politie Noord', \
        #                    'Politie Zuidplein', 'Politie Veilingweg', 'Politie Krimpen aan den Ijssel',
        #                    'Politie Ijsselmonde']
        # self.G_policeless.remove_nodes_from(nodes_to_remove)

        self.criminal_agent = next((a for a in self.model.scheduler.agents if a.unique_id == (self.model.criminal_id-1)), None)

        self.police_nodes = []
        for node in self.G.nodes('function'):
            if node[1] == 'police_bureau':
                self.police_nodes.append(node[0])

        self.G_policeless.remove_nodes_from(self.police_nodes)

        Police.guarded_stations = {}
        Police.guarded_metros = []
        Police.police_goals = {}
        self.get_strategy()

    def print2file(self, text):
        current_date = datetime.now().strftime('%d%m%Y')
        print_name = f"experiment/print_output_{current_date}.txt"
        with open(print_name, "a") as f:
            print(text, file=f)

# -------------------------------------------------------------------------
    """STRATEGIES"""
# ----------------------------------------------------------------------------------------
    def get_strategy(self):

        police_strategies= ['random','largest', 'furthest']
        if self.unique_id==100: #which means it is the first police unit:
            self.police_strategy= 'plaats_delict'
            self.police_strategy = self.plaats_delict()


        else:
            if self.police_strategy=='random':
                self.random_strategy()
            elif self.police_strategy == 'largest':
                self.largest_strategy()
            elif self.police_strategy == 'furthest':
                self.furthest_strategy()
            elif self.police_strategy == 'surround':
                self.surround_strategy()



    def random_strategy(self):
        self.goal=random.choice(list(self.G_policeless))
        # pick a goal
        self.goal = random.choice(list(self.G_policeless))
        # pick another goal if another unit is already going there AND P_multiple_at_station has not been satisfied yet
        #OR
        # if this goal is already chosen AND P_multiple_at_station is satisfied AND all the exits of that goal are already taken
        while (self.goal in Police.police_goals and self.unique_id< (100+self.P_multiple_at_station)) or \
            (self.goal in Police.police_goals and self.unique_id> (100+self.P_multiple_at_station) and \
             Police.police_goals[self.goal] >= self.G_policeless.nodes[self.goal]['Exits']):
            self.goal = random.choice(list(self.G_policeless))
        # append goal to dictionary and keep track of how many units have been sent to this goal
        if self.goal not in Police.police_goals:
            Police.police_goals[self.goal] =1
        elif self.goal in Police.police_goals:
            Police.police_goals[self.goal]+=1

        self.new_goal = False
        self.arrived = False

        # based on strategy, the meldkamer will choose a goal where to go to, based on this police units will be initialized
        self.set_pos(self.goal)


        self.path = nx.shortest_path(self.G, self.pos, self.goal)
        self.distance=1000*(nx.shortest_path_length(self.G, source=self.pos,
                                                            target=self.goal, weight='weight')) #to get meters
        ##help waarom is dit verschillend?
        # distance1=1000*(nx.shortest_path_length(self.G, source=self.pos,
        #                                                     target=self.path[0], weight='weight')) #to get meters
        # distance2 = 1000 * (nx.shortest_path_length(self.G, source=self.path[0],
        #                                             target=self.path[1], weight='weight'))  # to get meters
        # distance3 = 1000 * (nx.shortest_path_length(self.G, source=self.path[1],
        #                                             target=self.path[2], weight='weight'))  # to get meters
        # print(self.path,self.pos,self.path[1])
        # print('AAA! distance:',self.distance)
        # print('distance pos-[0]=',distance1, self.pos,self.path[0])
        # print('distance pos[1]',distance2,self.path[0],self.path[1])
        # print('distance pos-[2]', distance3,self.path[1],self.path[2])
        # print('total', distance1+distance2+distance3)

        #setting an arrival time, so the police doesn't go over an edge within one timestep.
        self.driving_time = self.distance / self.speed #metres/metres/min=min
        #using self.scheduler.time so that it doesnt go from 60 min to the next hour with 0 min. (makes it easier)
        #and multiplying self.driving_time by 2 because every step is 1/2min and driving_time is in minutes
        self.arrival_time = self.model.scheduler.time + (self.driving_time*2)
        self.print2file(('POL INFO: strategy:',self.police_strategy, 'Start pos:', self.pos,\
                    ', Goal:', self.goal, ', Travel time in minutes:', self.driving_time, ' Arrival time:', self.arrival_time))

    def plaats_delict(self):
        self.goal = self.criminal_agent.path[0] #go to init crime scene (which is what police does)
        #append goal to list so that next police unit wqon't go there
        Police.police_goals[self.goal]=1

        #based on strategy, the meldkamer will choose a goal where to go to, based on this police units will be initialized
        self.set_pos(self.goal)

        self.path = nx.shortest_path(self.G, self.pos, self.goal)
        self.distance = 1000 * (nx.shortest_path_length(self.G, source=self.pos,
                                                            target=self.goal, weight='weight'))  # to get meters

        # setting an arrival time, so the police doesn't go over an edge within one timestep.
        self.driving_time = self.distance / self.speed  # metres/metres/min=min
        # using self.scheduler.time so that it doesnt go from 60 min to the next hour with 0 min. (makes it easier)
        # and multiplying self.driving_time by 2 because every step is 1/2min and driving_time is in minutes
        self.arrival_time = self.model.scheduler.time + (self.driving_time * 2)

        self.print2file(('POL INFO: strategy:', self.police_strategy, 'Start pos:', self.pos, \
                  ', Goal:', self.goal, ', Travel time in minutes:', self.driving_time, ' Arrival time:', self.arrival_time))

    def largest_strategy(self):
        if self.new_goal==False:
            neighbor_counts = sorted([(len(list(self.G.neighbors(n))), n) for n in self.G.nodes() if
                                          self.G.nodes[n].get('function') == 'metro_station'], reverse=True)
            largest_nodes = []
            for node in neighbor_counts:
                if node[0]>2:
                    largest_nodes.append(node[1])
                #if multiple nodes are equally large, then pick the one which is closest
            sorted_nodes = sorted(largest_nodes)
            # Get the top 4 closest nodes
            self.top_closest = sorted_nodes[:min(self.model.units, 6)]

            # pick a goal
            self.goal = random.choice(self.top_closest)
            first_time=True

        elif self.new_goal==True:
            distances = {}  # Dictionary to store distances
            for node in self.top_closest:
                if nx.has_path(self.G_policeless, node, self.criminal_information):
                    shortest_path_length = nx.shortest_path_length(self.G_policeless, source=node, target=self.criminal_information)
                    distances[node] = shortest_path_length
                else:
                    distances[node] = float('inf')  # Set a large value for nodes not connected to the target
            # Sort furthest_nodes based on distances
            self.top_closest = sorted(self.top_closest, key=lambda node: distances[node])
            #pick a goal
            self.goal=self.top_closest[0]
            first_time=False


        # pick another goal if another unit is already going there AND P_multiple_at_station has not been satisfied yet
        # OR
        # if this goal is already chosen AND P_multiple_at_station is satisfied AND all the exits of that goal are already taken
        while (self.goal in Police.police_goals and self.unique_id < (100 + self.P_multiple_at_station) and first_time == True and self.model.units < 6) \
            or (self.goal in Police.police_goals and len(Police.police_goals) < len(self.top_closest)):
           self.goal = random.choice(list(self.top_closest))

       # append goal to dictionary and keep track of how many units have been sent to this goal
        if self.goal not in Police.police_goals:
            Police.police_goals[self.goal] = 1
        elif self.goal in Police.police_goals:
            Police.police_goals[self.goal] += 1

        # based on strategy, the meldkamer will choose a goal where to go to, based on this police units will be initialized
           # get a starting location for the police, unless this is its second detination then it already has a position
        if self.new_goal == False:
            self.set_pos(self.goal)
        elif self.new_goal == True:
            self.pos = self.pos
            self.new_goal = False
            self.arrived = False

        self.path = nx.shortest_path(self.G, self.pos, self.goal)
        self.distance = 1000 * (nx.shortest_path_length(self.G, source=self.pos,
                                                        target=self.goal, weight='weight'))  # to get meters

        # setting an arrival time, so the police doesn't go over an edge within one timestep.
        self.driving_time = self.distance / self.speed  # metres/metres/min=min
        # using self.scheduler.time so that it doesnt go from 60 min to the next hour with 0 min. (makes it easier)
        # and multiplying self.driving_time by 2 because every step is 1/2min and driving_time is in minutes
        self.arrival_time = self.model.scheduler.time + (self.driving_time * 2)

        self.print2file(('POL INFO: strategy:', self.police_strategy, 'Start pos:', self.pos,\
              ', Goal:', self.goal, ', Travel time in minutes:', self.driving_time, ' Arrival time:', self.arrival_time))

    def furthest_strategy(self):
        if self.new_goal==False:
            shortest_path_lengths = nx.shortest_path_length(self.G_policeless, self.criminal_agent.path[0])
            sorted_nodes = sorted(shortest_path_lengths, key=shortest_path_lengths.get, reverse=True)
            # Select the furthest 5 nodes
            # self.furthest_nodes = sorted_nodes[:min(self.model.units, 6)]

            self.furthest_nodes = []
            exit_sum = 0
            for node in sorted_nodes:
                if exit_sum < self.model.units:
                    exit_sum += self.G_policeless.nodes[node]['Exits']
                    self.furthest_nodes.append(node)
                elif exit_sum >= self.model.units:
                    break

            # pick a goal
            self.goal = random.choice(self.furthest_nodes)
            first_time=True


        elif self.new_goal==True:
            distances = {}  # Dictionary to store distances
            for node in self.furthest_nodes:
                if nx.has_path(self.G_policeless, node, self.criminal_information):
                    shortest_path_length = nx.shortest_path_length(self.G_policeless, source=node, target=self.criminal_information)
                    distances[node] = shortest_path_length
                else:
                    distances[node] = float('inf')  # Set a large value for nodes not connected to the target
            # Sort furthest_nodes based on distances
            self.furthest_nodes = sorted(self.furthest_nodes, key=lambda node: distances[node])
            #pick a goal
            self.goal=self.furthest_nodes[0]
            first_time = False
            self.print2file(('POL INFO: going to new location. was at', self.pos,' and now going to', self.goal,self.unique_id))


        # pick another goal if another unit is already going there AND P_multiple_at_station has not been satisfied yet
        # OR
        # if this goal is already chosen AND P_multiple_at_station is satisfied AND all the exits of that goal are already taken
        while (self.goal in Police.police_goals and self.unique_id < (100 + self.P_multiple_at_station) and first_time==True and self.model.units<6) \
                or (self.goal in Police.police_goals and len(Police.police_goals)<len(self.furthest_nodes)):
            self.goal = random.choice(list(self.furthest_nodes))

        # append goal to dictionary and keep track of how many units have been sent to this goal
        if self.goal not in Police.police_goals:
            Police.police_goals[self.goal] = 1
        elif self.goal in Police.police_goals:
            Police.police_goals[self.goal] += 1

        #get a starting location for the police, unless this is its second detination then it already has a position
        if self.new_goal==False:
            self.set_pos(self.goal)
        elif self.new_goal==True:
            self.pos=self.pos
            self.new_goal = False
            self.arrived = False

        self.path = nx.shortest_path(self.G, self.pos, self.goal)
        self.distance = 1000 * (nx.shortest_path_length(self.G, source=self.pos,
                                                        target=self.goal, weight='weight'))  # to get meters

        # setting an arrival time, so the police doesn't go over an edge within one timestep.
        self.driving_time = self.distance / self.speed  # metres/metres/min=min
        # using self.scheduler.time so that it doesnt go from 60 min to the next hour with 0 min. (makes it easier)
        # and multiplying self.driving_time by 2 because every step is 1/2min and driving_time is in minutes
        self.arrival_time = self.model.scheduler.time + (self.driving_time * 2)

        self.print2file(('POL INFO: strategy:', self.police_strategy,  'Start pos:', self.pos,\
              ', Goal:', self.goal, ', Travel time in minutes:', self.driving_time, ' Arrival time:', self.arrival_time))

    def surround_strategy(self):
        if self.new_goal == False:
            path_lengths = {}
            for police_node in self.police_nodes:
                path_lengths[police_node] = nx.shortest_path_length(self.G, source=police_node, \
                                                                    target=self.criminal_agent.path[0], weight='weight')
                path_lengths=dict(sorted(path_lengths.items(), key=lambda item: item[1]))
                # print(123,path_lengths)

                self.nodes_set_far = dict(itertools.islice(path_lengths.items(), 4))


            #pick starting point
            pos=random.choice(list(self.nodes_set_far))
            self.pos = pos
            self.start_pos = pos

            # # pick a goal
            closest_metro=self.G.nodes[self.pos]['closest_metro']
            path=nx.shortest_path(self.G_policeless,source=closest_metro, target=self.criminal_agent.path[0])
            #set goal to be halfway betwen police and criminal.
            self.goal = path[len(path)//2]
            first_time = True

        elif self.new_goal == True:
            start_node = self.criminal_information
            num_steps = 3
            nodes_set_close = set([start_node])
            self.nodes_set_far = set([start_node])
            for step in range(num_steps):
                # Create a set to store the neighbors at the current step
                current_step_neighbors = set()
                for node in self.nodes_set_far:
                    current_step_neighbors.update(self.G_policeless.neighbors(node))

                # Update the set for the next iteration
                if step <= num_steps - 2:
                    self.nodes_set_far = current_step_neighbors
                if step <= num_steps - 5:
                    nodes_set_close = current_step_neighbors

            self.nodes_set_far = list(self.nodes_set_far - nodes_set_close)

            start = self.criminal_agent.path[0]
            update = self.criminal_information
            path = self.path = nx.shortest_path(self.G, self.pos, self.goal)

            # Find the indices of nodes A and X in the path
            index_start = path.index(start) if start in path else None
            index_update = path.index(update) if update in path else None

            # Check if A and X are both in the path
            if index_start is not None and index_update is not None:
                # Create a new list without nodes in the path
                self.nodes_set_far = [node for node in self.nodes_set_far if node not in path]

            self.goal = random.choice(self.nodes_set_far)
            first_time = False
            self.print2file( ('POL INFO: going to new location. was at', self.pos, ' and now going to', self.goal, self.unique_id))

        # pick another goal if another unit is already going there AND P_multiple_at_station has not been satisfied yet
        # OR
        # if this goal is already chosen AND P_multiple_at_station is satisfied AND all the exits of that goal are already taken
        while (self.goal in Police.police_goals and self.unique_id < (
                100 + self.P_multiple_at_station) and first_time == True and self.model.units < 6) \
                or (self.goal in Police.police_goals and len(Police.police_goals) < len(self.top_closest)):
            self.goal = random.choice(list(self.nodes_set_far))

        # append goal to dictionary and keep track of how many units have been sent to this goal
        if self.goal not in Police.police_goals:
            Police.police_goals[self.goal] = 1
        elif self.goal in Police.police_goals:
            Police.police_goals[self.goal] += 1

        # based on strategy, the meldkamer will choose a goal where to go to, based on this police units will be initialized
        # get a starting location for the police, unless this is its second detination then it already has a position
        if self.new_goal == True:
            self.pos = self.pos
            self.new_goal = False
            self.arrived = False

        self.path = nx.shortest_path(self.G, self.pos, self.goal)
        self.distance = 1000 * (nx.shortest_path_length(self.G, source=self.pos,
                                                        target=self.goal, weight='weight'))  # to get meters

        # setting an arrival time, so the police doesn't go over an edge within one timestep.
        self.driving_time = self.distance / self.speed  # metres/metres/min=min
        # using self.scheduler.time so that it doesnt go from 60 min to the next hour with 0 min. (makes it easier)
        # and multiplying self.driving_time by 2 because every step is 1/2min and driving_time is in minutes
        self.arrival_time = self.model.scheduler.time + (self.driving_time * 2)

        self.print2file(('POL INFO: strategy:', self.police_strategy, 'Start pos:', self.pos, \
                         ', Goal:', self.goal, ', Travel time in minutes:', self.driving_time, ' Arrival time:',
                         self.arrival_time))
# -------------------------------------------------------------------------
        """SET POSITION"""
 # -----------------------------------------------------------------------------------------
    def set_pos(self,goal):
        # Compute shortest path lengths from each police node to the crime scene
        path_lengths = {}
        for police_node in self.police_nodes:
            path_lengths[police_node] = nx.shortest_path_length(self.G, source=police_node, \
                                                                target=goal, weight='weight')
        # print(path_lengths)
        # Choose the police node with the shortest path to the crime scene
        pos = min(path_lengths, key=path_lengths.get)
        self.pos = pos
        self.start_pos=pos




    # -------------------------------------------------------------------------
    """MOVES"""
# -----------------------------------------------------------------------------------------
    # in the move function agent will evaluate if they have reached their goal, and then decide upon
    # what to do next. Based on what to do next, they will refer to new function
    #intitally police will stay at desired end location and keep the look out
    def move(self):
        #generate criminal location updates on every info_update_freq (input) minute, with delay init_call_delayt(input)
# INTERACTIE
        self.criminal_information=None
        if self.model.current_time % self.info_update_freq == 0 and self.interactie==True:
            timestamp= int(self.model.current_time)-int(self.model.init_call_delay)
            # Start from the current timestamp and go back in time
            self.criminal_passed_police=self.model.criminal_agent.passed_police
            while timestamp >= 0:
                if self.model.criminal_agent.criminal_positions[timestamp][0] != 'driving':
                    self.criminal_information = self.model.criminal_agent.criminal_positions[timestamp][0]
                    break  # Exit the loop when a non-'driving' entry is found
                timestamp -= 0.5

    # INTERACTIE
        if self.criminal_passed_police >3 and self.interactie==True:
            self.under_cover=True

        time_till_arrival = (int(self.arrival_time) - int(self.model.scheduler.time)) / 2
        if time_till_arrival > 0:
            self.print2file(('POL INFO: underway to', self.goal,', ETA in:', (int(self.arrival_time)-int(self.model.scheduler.time))/2,\
                  'Unique ID:', self.unique_id))
            self.pos=self.pos

        elif time_till_arrival==0:
            self.pos=self.goal
            self.arrived=True
            self.print2file(('POL INFO:  made it to goal:', self.pos,self.goal, self.unique_id))

            if self.guarding == 'station_exit':
                self.print2file(('POL INFO: no new info, guarding station exit'))


                #if police is guarding main entrances or there are only main entrances:
                exits_value = self.G_policeless.nodes[self.pos]['Exits']
                if self.entrance=='main' or exits_value<=2:
                    self.entrance_covered = random.choice(['a','b'])
                # or guard side exits if there are any
                elif self.entrance !='main' and exits_value>2:
                    side_entrances =int(exits_value-2) #subtract 2 bc the 2 first entrances are main and we are picking a side entrance here
                    self.entrance_covered = random.choice([chr(ord('c') + i) for i in range(side_entrances)])

                if self.pos in Police.guarded_stations:
                    while Police.guarded_stations[self.pos].count(self.entrance_covered)>=1 and len(Police.guarded_stations[self.pos])<exits_value:
                        self.entrance_covered = random.choice([chr(ord('a') + i) for i in range(int(exits_value))])


                # add position and exit to dictionary
                if self.pos in Police.guarded_stations:
                    Police.guarded_stations[self.pos].append(self.entrance_covered)
                elif self.pos not in Police.guarded_stations:
                    Police.guarded_stations[self.pos]=[(self.entrance_covered)]

            elif self.guarding=='metro_platform':
                self.pos = self.pos
                Police.guarded_metros.append(self.pos)

# INTERACTIE
        if self.criminal_information != None and self.arrived==True and self.unique_id>100 and self.interactie==True:
            #checking whether the criminal is moving in the direction of the police's self.goal. if the distsnce between the police goal
            #and criminal start position is larger than the distance between criminal's recent position and police goal, then
            #criminal is moving towards police's goal and police will stay there.
            #if the difference is bigger the criminal is moving away from police, and police should move.
            distance_crimstart_goal = nx.shortest_path_length(self.G_policeless, source=self.criminal_agent.path[0], target=self.goal)
            distance_crimrecent_goal = nx.shortest_path_length(self.G_policeless, source=self.criminal_information, target=self.goal)
            if distance_crimstart_goal > distance_crimrecent_goal:
                pass
            elif distance_crimstart_goal < distance_crimrecent_goal:
                #remove the position from the list of metros and exits that are being guarded
                if self.guarding == 'station_exit':
                    if self.pos in Police.guarded_stations:
                        if len(Police.guarded_stations[self.pos])==1:
                            Police.guarded_stations.pop(self.pos, None)
                        elif len(Police.guarded_stations[self.pos]) >1:
                            Police.guarded_stations[self.pos].remove(self.entrance_covered)
                elif self.guarding == 'metro_platform':
                    Police.guarded_metros.remove(self.pos)
                self.changed_goal=self.changed_goal+1
                self.new_goal=True
                self.get_strategy()

    # -------------------------------------------------------------------------
        """STEP FUNCTION"""
    # -----------------------------------------------------------------------------------------

    def step(self):
        if self.model.game_over == False:
            self.move()
        if self.model.scheduler.time == self.model.initiate_police_time:
            self.done=True



# -------------------------------------------------------------------------
    """GUARDED"""
# -----------------------------------------------------------------------------------------
    def get_all_guarded(cls):
        return cls.guarded_stations, cls.guarded_metros

    def destination_goals(cls):
        return cls.police_goals
