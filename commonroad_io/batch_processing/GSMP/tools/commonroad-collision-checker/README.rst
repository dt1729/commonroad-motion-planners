============================
CommonRoad-Collision-Checker
============================

System Requirements
-------------------

The software is written in Python 3.6 and tested on MacOs and Linux. The usage of the Anaconda_ Python distribution is strongly recommended. The requirements for the C++ collision checker library are a C++11 compiler and CMake. The requirements for the Python wrapper are C++11 compiler, CMake, and Python 3.6 with development headers. If you are a Mac user, we recommend you to use Homebrew_, allowing you to install required dependencies such as Eigen.

.. _Anaconda: http://www.anaconda.com/download/#download
.. _Homebrew: https://brew.sh


Dependencies
------------

Following third party libraries are needed to be installed for the CommonRoad-Collision-Checker:

* `FCL -- The Flexible Collision Library <https://github.com/flexible-collision-library/fcl>`_
* `libccd <https://github.com/danfis/libccd>`_
* `Eigen3 <https://eigen.tuxfamily.org/dox/>`_

Following packages are necessary, if you want to use the CommonRoad-Collision-Checker in Python with `commonroad-io <https://pypi.org/project/commonroad-io/>`_:

* `pybind11 <https://github.com/pybind/pybind11>`_
* `commonroad-io <https://pypi.org/project/commonroad-io/>`_
* `Triangle <https://pypi.org/project/triangle/>`_
* `matplotlib <https://pypi.org/project/matplotlib/>`_
* `Shapely <https://pypi.org/project/Shapely/>`_
* `numpy <https://pypi.org/project/numpy/>`_

Following packages are necessary, if you want to execute the available tutorials:

* `Jupyter <https://pypi.org/project/jupyter/>`_

=====================================
Installation of Third Party Libraries
=====================================

#. Install `libccd <https://github.com/danfis/libccd>`_:

   Clone the repository from `https://github.com/danfis/libccd <https://github.com/danfis/libccd>`_ (master branch) and execute the following commands.

	.. code-block:: bash

            $ cd libccd
            $ mkdir build && cd build
            $ cmake -G "Unix Makefiles" -DENABLE_DOUBLE_PRECISION=ON -DBUILD_SHARED_LIBS=ON ..
            $ make
            $ sudo make install

   If you face some issue when installing FCL (in the following step), try setting DCMAKE_POSITION_INDEPENDENT_CODE on:

    .. code-block:: bash

            $ cd libccd
            $ mkdir build && cd build
            $ cmake -G "Unix Makefiles" -DENABLE_DOUBLE_PRECISION=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
            $ make
            $ sudo make install

#. Install `FCL -- The Flexible Collision Library <https://github.com/flexible-collision-library/fcl>`_:

   Clone the repository from `https://github.com/flexible-collision-library/fcl <https://github.com/flexible-collision-library/fcl>`_ (master branch) and execute the following commands.

	.. code-block:: bash

            $ cd fcl

            linux: $ sudo apt-get install libboost-dev libboost-thread-dev libboost-test-dev libboost-filesystem-dev libeigen3-dev
            macOS: $ brew install eigen

	.. code-block:: bash

            $ mkdir build && cd build
            $ cmake ..
            $ make # use make -j 4 to speed up the process
            $ sudo make install

============
Installation
============

Full Installation with Anaconda
-------------------------------

It is assumed that you have installed Anaconda_ and that your Anaconda environment is called **commonroad-py37**.
(Commonroad-search is written in python3.7)

#. Open your console in the root folder of the CommonRoad-Collision-Checker.

#. Activate your environment with

	.. code-block:: bash

		    $ conda activate commonroad-py37

#. Compile the CommonRoad-Collision-Checker library by running
        **Note that you have to replace**
         - *"/path/to/your/anaconda3/envs/commonroad-py37"* with the path to your Anaconda environment;
         - *"3.7"*  with the Python version of your Anaconda environment.
         - please provide **absolute** path.

        .. code-block:: bash

            $ mkdir build
            $ cd build
            $ cmake -DADD_PYTHON_BINDINGS=TRUE -DPATH_TO_PYTHON_ENVIRONMENT="/path/to/your/anaconda3/envs/commonroad-py37" -DPYTHON_VERSION="3.7" -DCMAKE_BUILD_TYPE=Release ..

        The next line refers only to users of Mac OS X 10+:

        .. code-block:: bash

            $ sed -i '' 's!-lccd!/usr/local/lib/libccd.2.0.dylib!' python_binding/CMakeFiles/pycrcc.dir/link.txt

        .. code-block:: bash

            $ make




#. (Optional) Install the CommonRoad-Collision-Checker with

    .. code-block:: bash

            $ cd ..
            $ python setup.py install

    **OR** add the root folder of the CommonRoad-Collision-Checker to your Python-Interpreter.


Documentation
-------------

The documentation of the C++ API can be found under doc/cpp/html/index.html.

The documentation of the Python wrapper can be found under doc/python/html/index.html.
