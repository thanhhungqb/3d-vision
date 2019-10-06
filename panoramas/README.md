# Information
This folder contains source code and data/images for panorama exercise.

The source code use some old version of libraries, so make sure environment include correct version.
*requirements.txt* includes many libraries and correct to install.

Source code for functions of Chapter 3 are from http://programmingcomputervision.com.
Reuse source with small change on sift.py, command line of *sift* call changed.


# Note
Enviroment install (Ubuntu 18.04)

        apt install python-tk

*sift* program must be in **$PATH**, following steps:
- download from https://www.cs.ubc.ca/~lowe/keypoints/
- extract to a folder call $PP
- linux command to export **$PATH**

        export PATH=$PP:$PATH
