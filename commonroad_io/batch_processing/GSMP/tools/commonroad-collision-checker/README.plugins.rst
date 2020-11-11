You could optionally modify the library to use a plugin for fast triangulation of polygons which uses the CGAL library (https://www.cgal.org/). 
To do so, CGAL needs to be installed and in line 12 of the file cpp/collision/src/plugins/triangulation/triangulate.cc 0 needs to be replaced with 1.
To enable this functionality one also needs to modify line 6 of the cpp/collision/include/collision/application_settings.h file, replacing 0 with 1.
Please see the license notice in notes.txt.
