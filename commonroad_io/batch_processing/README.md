# Graph Search-Based Motion Planner with Motion Primitives

This is a programming exercise for the lecture **Introduction to Artificial Intelligence** and **Gems of Informatics II** delivered at the  Department of Informatics, TUM. Please clone this repository or download it using the button at the upper-right corner. The repository has the following folder structure:
``` code-block:: text
commonroad-search/
	├GSMP/
		├motion_automata/
			├automata/
			├figures/
			├motion_primitives/
			└vehicle_model/
		└tools/
			├commonroad-collision-checker/
			├commonroad-road-boundary/
			├docker/
			└motion_primitives_generator/
	├notebooks/	
		├batch_processing/
		└tutorials/
	├pdfs/	
	├scenarios/
		├exercise/
		└tutorial/
	└solutions/ 
```
The codes are written in Python 3.7 and tested on Ubuntu 18.04. 

## Ways to Install

You can either install the softwares on your own machine, use a virtual machine image or run a docker image.

1. Own machine (Works on Ubuntu): Please follow the installation guide below if you are using your own machine. 
2. Virtual machine (`Recommended` for Ubuntu, Mac OS & Windows): Alternatively, you can use the virtual machine image provide by us, in which all the necessary modules are installed already. You can down the virtual machine image via [this](https://syncandshare.lrz.de/dlpw/fi2BN8NUepqiQzfG3LzWYf4J/Virtual_Machine.zip) link and run it in Virtual Box. The downloading password and default login password are both `commonroad`. 
3. Docker (Works on Ubuntu, Mac OS with OS X El Capitan 10.11 or later, Windows 10 64-bit: Pro, Enterprise, or Education, Build 15063 or later): Also, you can run a docker image provided by Tom Dörr. After installing docker, you can run the image using command

```sh
docker run -it -p 9000:8888 --mount src="$(pwd)",target=/commonroad-search,type=bind tomdoerr/commonroad-search
```

&ensp;&ensp;&ensp;&ensp;&ensp;and open the Jupyter Notebook by visiting `localhost:9000` in your web browser.

After you have set up your environment, please further proceed with `pdfs/0_Guide_for_Exercise.pdf`. 

## Installation guide

`Skip this section if you intend to use the provided virtual machine or docker image.`

We recommend using [Anaconda](https://www.anaconda.com/) to manage your environment so that even if you mess up something, you can always have a safe and clean restart. A guide for managing the environments can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Also, the usage of [PyCharm](https://www.jetbrains.com/pycharm/) is highly recommended (free version available for students).

After installing Anaconda, create a new environment by command:
``` sh
 $ conda create -n commonroad-py37 python=3.7
```

Here the name of the environment is called **commonroad-py37**. You may also change this name as you wish. In such case, don't forget to change it in the following commands as well.

`Always activate` this environment before you do anything related:

```sh
  $ conda activate commonroad-py37
  or
  $ source activate commonroad-py37
```
Install `Jupyter Notebook` and supplementary modules:
```sh
  $ conda install jupyter
  $ conda install ipykernel
  $ pip install ipython-autotime
  $ pip install PyYAML
  $ conda install ipywidgets
  $ conda install sphinx
  $ conda install scipy
  $ jupyter nbextension install --py widgetsnbextension --user
  $ jupyter nbextension enable widgetsnbextension --user --py
```

This exercise has three main dependencies that need to be installed.

### 1. CommonRoad-io

As documented in CommonRoad-io [Documentation](https://commonroad.in.tum.de/static/docs/commonroad-io/index.html), type in the following command to install the package:

```sh
  $ pip install commonroad-io
```


### 2. CommonRoad-Collision-Checker
Go to folder `GSMP/tools/commonroad-collision-checker/` and follow the instruction in README.rst. (You may navigate to it from this page for a better rendering of the .rst file)

A tutorial of CommonRoad Collision Checker can be found [here](https://commonroad.in.tum.de/tutorials/).

### 3. CommonRoad-Road-Boundary
Go to folder `GSMP/tools/commonroad-road-boundary/` and follow the instruction in README.md. (You may navigate to it from this page for a better rendering of the .md file) In case you face an error, refer to troubleshooting section. 

## Tutorials

Navigate your terminal to `commonroad-search/` folder, and start Jupyter Notebook with:
```shell
  $ jupyter notebook
```

In the prompt up page, navigate to `notebooks/tutorials/` and follow the tutorials `tutorial_commonroad-search.ipynb`, `cr_uninformed_search_tutorial.ipynb`, and `cr_informed_search_tutorial.ipynb`.  Remember to refer to `pdfs/0_Guide_for_Exercise.pdf` for additional explanation. The executed Jupyter notebooks for tutorials can also be found [here](https://commonroad.in.tum.de/tutorials/).

## Implement your own search algorithm

Open `GSMP/motion_automata/automata/MotionPlanner.py`. Write your own heuristic functions and/or search algorithm in the following functions:

```python
	def calc_heuristic_cost()
	def search_alg()
```

There are already two search algorithms, namely `A*` and `Greedy Best First Search`, implemented as examples for you. You are free to refer to them for some inspiration.
The `A*` is not working perfecly and should only serve as code reference how a heuristic can be developed.

## Troubleshooting

### 1. Boundary library not working properly

If there are errors stating not finding the boundary library (e. g. module `construction` not found) while going through the second tutorial, try manually copying all the contents under folder `GSMP/tools/commonroad-road-boundary/`  into `/path/to/your/anaconda3/envs/lib/python3.7/site-packages/commonroad-road-boundary/`. 
`Make sure to copy all files within the folder manually, not just copying the folder it self. Also, remember to add this path to your IDE's (e. g. PyCharm) interpretor path.`