# Human-Agent Interaction in Competitive Search and Rescue Games

This repository houses the implementation of the experiments carried out in the 'names to come onces published' paper

## Navigating the repository
* `ExperimentEmpty` contains all code necessary to run the human versus human condition.
* `ExperimentHuman` contains all code necessary for the human versus algorithms conditions.


## Running each experiment
The code is based on a combined implementation of gazebo and ros2, thus to run each experiment this environemt has to be guaranteed.

To run the code these packages need to be installed additionally to ros2:
* `Open-Cv`
* `PyQt`


Once the code has been downloaded, situate yourself in the given experiment folder you want to run.
* First source your ros2 codebase `source /opt/ros/{your-ros2-DIST}/setup.bash`
* Next you need to build the codebase `colcon build`
* Source your setup file `source install/local_setup.bash`
* Execute the simulated environment `ros2 launch LaunchSup.py`

This starts the Nodes and simulated environmnet in gazebo, the last thing to do is to start the interfaces for the human participant.

In a new terminal for EACH interface proceed with the following steps:
* source the setup file and ros2 codebase again!
* move to the UI folder `cd UI`
* execute `python3 searcher.py` or `python3 hider.py`


## Citation
For any citation purposes: 
>citaiton to come
