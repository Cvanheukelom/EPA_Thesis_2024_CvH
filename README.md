# Fugitive interception simulation game

The fugitive interception simulation game is a model which simulates the interaction between the offender and police in such a situation. The simulation can be altered to 4 experiments; one where there is no interaction with information between the agents, one scenario where only the offender can interact with information, another scenario where only the police can interact with information and the final scenario where both players can interact with information. The model is initiated in the main.py file, which triggers the metro, criminal and police agent. This simulation is part of the EPA Master thesis course of the Engineering and Policy Analysis Mcs program at the Delft University of Technology. 


## To run the model

Launch the simulation game using multiprocessor and without visualization:
```
    $ model_run_parallel.py
```

Launch the simulation game with visualization:
```
    $ model_run.py
```

Currently the files are set to run the base scenario of the no- interaction experiment (1). If another experiments should run, interaction variable for the police and criminal agents need to be changed accordingly.

## Files

### Python files:
* [main.py](main.py): Python script which initiates the agents in the simulation game.
* [Metro.py](Metro.py): Python script to define metro agent.
* [Criminal.py](Criminal.py): Python script to define criminal agent.
* [Police.py](Police.py): Python script to define police agent.

### Supporting files:
* [metro_network.json](metro_network.json): Json file with metro network and police nodes to limit computational time. If network is altered, it needs to be re-created once.

### Data folder general:
* [Metro time table](data/MetroTimetTable.csv): Excel file containing metro schedule upon which metro agents are triggered
* [Network](data/Nodes_complete_police.csv): Excel file containing all nodes for the network with its attributes.

### Data folder exploratory run:
* [Scenario set-up](data/experiment_scenarios_27112023.csv): Excel file containing input variables per scenario.
* [Uncertainties](data/exp_uncertainties_no_int_exploring.csv): Excel file containing uncertainties for every scenario.
* [Output](experiment/no_int_exploring_27112023): Folder containing output file and log files


### Data folder experiment 1 - no interaction:
* [Base scenario](data/base_28112023.csv): Excel file containing input variables for base scenario.
* [Scenario set-up](data/experiment_scenarios_28112023.csv): Excel file containing input variables per scenario.
* [Uncertainties](data/exp_uncertainties_no_int.csv): Excel file containing uncertainties for every scenario.
* [Output](experiment/no_interaction_28112023): Folder containing output file and log files


### Data folder experiment 2 - criminal interaction:
* [Base scenario](data/base_29112023.csv): Excel file containing input variables for base scenario.
* [Scenario set-up](data/experiment_scenarios_29112023.csv): Excel file containing input variables per scenario.
* [Uncertainties](data/exp_uncertainties_crim_int.csv): Excel file containing uncertainties for every scenario.
* [Output](experiment/crim_int_29112023): Folder containing output file and log files


### Data folder experiment 3 - police interaction:
* [Base scenario](data/base_01122023.csv): Excel file containing input variables for base scenario.
* [Scenario set-up](data/experiment_scenarios_01122023.csv): Excel file containing input variables per scenario.
* [Uncertainties](data/exp_uncertainties_pol_int.csv): Excel file containing uncertainties for every scenario.
* [Output](experiment/pol_int_01122023): Folder containing output file and log files


### Data folder experiment 4 - full interaction:
* [Base scenario](data/base_03122023.csv): Excel file containing input variables for base scenario.
* [Scenario set-up](data/experiment_scenarios_03122023.csv): Excel file containing input variables per scenario.
* [Uncertainties](data/exp_uncertainties_full_int.csv): Excel file containing uncertainties for every scenario.
* [Output](experiment/full_int_03122023): Folder containing output file and log files



### Notebook folder:
* [Analysis exploratory run](experiment/experiment%20analysis/exp_no_int_exploring_27112023.ipynb):  Data analysis exploratory run
* [Analysis experiment 1](experiment/experiment%20analysis/exp_no_int_28112023.ipynb): Data analysis experiment 1
* [Analysis experiment 2](experiment/experiment%20analysis/exp_crim_int_29112023.ipynb): Data analysis experiment 2
* [Analysis experiment 3](experiment/experiment%20analysis/exp_pol_int_01122023.ipynb): Data analysis experiment 3
* [Analysis experiment 4](experiment/experiment%20analysis/exp_full_int_03122023.ipynb): Data analysis experiment 4
