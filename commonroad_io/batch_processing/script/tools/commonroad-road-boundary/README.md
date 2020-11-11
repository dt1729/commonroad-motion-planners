Automatic road boundary construction for CommonRoad scenarios
=============================================================
Implemented in the course of the Bachelor Thesis: __Automated Approximation of Lane Boundaries of Arbitrary Road Networks with Simple Geometric Shapes__ by Alexander Zhu (alexander.zhu@tum.de).

Installation
============

1. Activate your Anaconda Environment, e.g., that you have also used for the CommonRoad-Python-Tools.
2. Download and install the [Python Wrapper for the Triangle library](http://dzhelil.info/triangle/installing.html).
3. Install the [Python Wrapper for the General Polygon Clipper library](http://www.cs.man.ac.uk/~toby/alan/software/):
    ` pip install polygon3`
4. If you are using the PyCharm IDE, add the path to the repository to your Python-Interpreter.

Introduction
============
The road boundaries are generated in __construction.py__. It has the function:

	construct(scenario_file, build_order, draw_order=[], plot_order=['plot'], boundary_margin=20)

See section *Road Boundary Construction* for explanation of its parameters.

__main.py__ includes example functions and measurements.
When using Pycharm, you can call the functions in main.py or manually try out the construction method as following:  
	```Edit the run configuration for main.py and enable "Show command line afterwards"```
	
Road Boundary Construction
==========================
Here, the use of the function __construct__ in construction.py is described:
	
	construct(scenario_file, build_order, draw_order=[], plot_order=['plot'], boundary_margin=20)

First, a CommonRoad scenario .xml file has to be selected (without including the .xml suffix).

Then, the boundary construction is determined by three string lists, which are called orders.
They determine what kind of boundary is built and drawn and how the outcome is displayed.
The sequence of the individual commands corresponds to the order in which they are executed.

An example build_order: ```['simple_triangles', 'quads']```
Here, first the road is built with simple triangles, then the quadtree boundary is constructed.

It is possible to supply parameters to a command. This is done by enclosing the command in a 2-tuple, with the command name as the first element, and the parameters as a dictionary in second.

Example: ```['simple_triangles', ('quads', {'max_depth': 10})]```
Again, the road is built with simple triangles, and then the quadtree is constructed, with the parameter max_depth set to a value of 10.

In the following, the possible commands for each order are listed.

Build Order
-----------
Determines what construction methods are executed.
### Commands
- #### __simple_triangles__  
	Build road representation, where each lanelet is split into triangles. Required for all boundary construction methods.
- #### section_triangles
	Road representation, where each lane section (section of adjacent lanelets) is split into triangles. Used to __mitigate holes__ between adjacent lanelets, that appear if the lanelet borders are not consistent. Should be used __in addition to simple_triangles__, if used alone there might appear gaps between lane sections.
  
	*example of use*: 
    	```build_order = ['simple_triangles', 'section_triangles', 'quads']```
- #### __quads__  
	Boundary representation constructed with a quadtree. 
	##### Parameters: 
	+ max_depth: Maximum recursion depth, higher values improve precision of the boundary, but generate more objects and exponentially increase the runtime. Default: 10
	+ speedup: Flag that enables a speedup method that decreases runtime. Default: False (is automatically set to True if 'rectangles' has been constructed)
	+ build_antiquads: Flag that determines whether rectangles that are not added to the boundary are added to the 'antiquads' shapegroup instead. Default: False
	
- #### rectangles  
	Rectangles along the lane section polylines, are used for enabling the quadtree speedup method.
	##### Parameters:
	+ width: Width of an individual rectangle. Default: 0.2

- #### __shell__  
	Boundary representation with "road-oriented" rectangles, constructed with the lane section polylines, similar to the 'rectangles', but they are guaranteed to not collide with the road.
	##### Parameters:
    - width: Width of an individual rectangle. Default: 0.2
    - margin: Distance between the shell rectangles and the road.Default: 0.02
    - max_depth: Maximum recursion depth of the rectangle subdivision process, which is used to guarantee that the boundary doesn't intersect with the road. Default: 5
	
- #### __triangulation__  
	Boundary representation with a Delaunay Triangulation by the [*Triangle library*](https://www.cs.cmu.edu/~quake/triangle.html).
    ##### Parameters:
	- call_options: String Flags provided to the Triangle library, see https://www.cs.cmu.edu/~quake/triangle.switch.html for possible values. Default: '-Q -S'
	- input_holes: Flag that determines if the triangles in the road area are filtered by the Triangle library initially. Default: False
	- filter_radius: Radius of the middle points/circles that are used for triangle filtering. Increase value if artifact triangles appear between lanelets. Default: 0 
	
- #### critical_area  
	Road representation that extends sligthly beyond the road itself. Used for Monte Carlo measurement of the area that is close to the road.
	##### Parameters:
	+ width: Distance by which the critical_area extends beyond the road. Default: 2
	
- #### runtime  
	Tool for runtime measurement with [timeit](https://docs.python.org/3/library/timeit.html)
	##### Parameters:
	- build: Build_order for which the runtime is tested. Default: 'pass'
	- setup: Step between each run. Default: 'pass'
	- number: Number of repetitions. Default: 100
	##### Examples:
	- ```build_order = ['simple_triangles', ('runtime', {'build': ['triangulation'], 'number': 50})]```  
		Test Triangulation 50 times.
	- ```build_order = ['simple_triangles', ('runtime', {'build': ['rectangles', 'quads'], 'number': 50, 'setup': 'lambda : reset_shapegroups()'})]```  
		Test Rectangle and Quads construction 50 times, in between each run reset the shapegroups.

Draw Order
----------
Performs draw_object function for a shapegroup (from fvks.visualization.draw_dispatch)
### Commands and default color
- quads: #a6cee3
- simple_triangles: #fb9a99
- section_triangles: #fdbf6f
- rectangles: #b2df8a
- antiquads: #a6cee3
- shell: #33a02c
- triangulation: #ff7f00
- critical_area: #cab2d6, zorder = 10

### Other Defaults
- edgecolor = #000000
- zorder = 20

Plot Order
----------
Displays a plot with matplotlib, can also save it to a .pdf figure.
### Commands
- #### plot  
	Shows the plot. 
	##### Parameters:
	+ corners: The plot is limited to these coordinates. Default: coordinates of the entire 	scenario
	example: 'corners': [0,10,-5,10] Only shows the area between the coordinates x=[0,10] and y=[-5,10]
	+ show_axis: The plot shows the axis and labels. Default: True
- #### save
	Saves the plot as a .pdf figure that can be used in LaTeX papers.
	##### Parameters:
	+ filename: Name of the .pdf file. Default: 'figure'
	+ scale: Scale of the saved figure. Default: 0.9 (LaTeX default figure size)
	+ corners: The plot is limited to these coordinates. Default: coordinates of the entire 	scenario
	+ show_axis: The plot shows the axis and labels. Default: False 

Examples
========
See main.py for other examples.
- ```construct('scenarios/GER_Ffb_2', ['simple_triangles', 'quads'], plot_order=[])```  
	Constructs a simple quadtree boundary for the scenario.
- ```construct('scenarios/GER_Ffb_2', ['simple_triangles', 'quads'], ['quads'])```  
	Constructs a simple quadtree boundary and draws and plots it.
- ```construct('scenarios/GER_Ffb_2', ['simple_triangles', 'section_triangles', ('shell' {'width': 2})], ['shell'])```  
	Constructs a shell boundary with a width of 2. Also uses section triangles to possibly remove gaps in the road. The shell is then drawn and plotted.
- ```construct('scenarios/GER_Ffb_2', ['simple_triangles', 'triangulation'], ['triangulation'], [('save', {'filename': 'figure1', 'corners': [0,10,-5,10]})])```  
	Constructs a triangulation boundary, then saves the plot in the file 'figure1.pdf'. The plot only shows the area between the coordinates x=[0,10] and y=[-5,10].
	
Scenarios
=========
In the scenario folder, there are some CommonRoad scenarios that can be used for testing. Preferably, use the scenarios in the subfolder measurable, as the others might have some flaws. 

Known Issues
============
- Scenario boundary: The scenario boundary is a simple axis aligned box. This might not fit well for some scenarios. Use the shell boundary if that is the case.
- Scenario lanelet adjacencies: The included scenarios feature an outdated adjacency definition, where merging lanelets are not marked as adjacent. This can lead to minor artifacts (which are only problematic in the Triangulation approach). The new adjacency definition should mark such lanelets as adjacent, which should remove all artifacts.
- Triangulation: The triangulation can potentially lead to thin triangle artifacts within the road. If that is the case, tweak the triangulation parameters or update the scenario lanelet adjacencies and use section triangles. 
