from setuptools import setup, find_packages

setup(
    name='commonroad-collision-checker',  
    version='2019.1',
    description='Collision checker for CommonRoad scenarios.',
    url='https://commonroad.in.tum.de/',
    author='Technical University of Munich',  
    author_email='commonroad-i06@in.tum.de',
    license='BSD',
    packages=['.', 'commonroad_cc'],
    include_package_data=True,
	install_requires=[
        'commonroad-io',
		'numpy>=1.13',
		'shapely>=1.6.4',
		'matplotlib>=2.2.2',
        'jupyter',
        'triangle',
	],
	extras_require={
		'doc':	['sphinx>=1.3.6',
				 'graphviz>=0.3',
				 'sphinx-autodoc-typehints>=1.3.0',
                 'sphinx_rtd_theme>=0.4.1',
                 'sphinx-gallery>=0.2.0',
                 'ipython>=6.5.0'],
	},
    data_files=[('.',['LICENSE'])],
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
	    "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",	
    ],
    zip_safe=False,
)
