Tutorials on Tree Search Algorithms
============
This is a short guide on how to use the search algorithms in tree_search.py for lectures and tutorials.

Set up
============
For using the code, you have to install the tools as described in the README in the root folder. For using the code in
Pycharm, create a new project in the folder `./GSMP/motion_automata`, to avoid path issues.

# Documentation
For running the search algorithms, you can currently specify three different configurations which mainly influence the
visualization of the search and are defined in `config.py`:

### Default: 

* This configuration should only be used in the jupyter notebook tutorial (`cr_uninformed_search_tutorial` and
`cr_informed_search_tutorial`). It first runs the algorithm and stores the state of
each motion primitive, i.e., explored/frontier/currently exploring etc., for each step. Then it plots these steps using
ipywidgets, so the student can step through the search in the jupyter notebook. 

* The color code for plotting is taken from the AIMA tutorial, but can be changed in `config.py`. 

* By setting PLOT_LEGEND to True, a legend with all states of
motion primitives (see `MotionPrimitiveStatus` in `helper_tree_search.py`) is plotted. 

* To reduce the number of steps, you can turn off that motion primitives with collisions are plotted one after the other by setting PLOT_COLLISION_STEPS
to False.

### StudentScript: 

This configuration uses the same parameters as the Default configuration, but it can be run in Pycharm,
e.g. when executing `main.py`. It plots the different steps while executing the search algorithm. It should help
students to easily go through the code.
