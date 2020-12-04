# Author:
Lih-Narn Wang
Master of Engineering in Robotics
University of Maryland, College Park
Email: ytcdavid@terpmail.umd.edu

# Introduction
This is a homework from the course CMSC733 in the University of Maryland.
This is an implementation of the paper "A Flexible New Technique For Camera Calibration" by Zhengyou Zhang.
The intrinsic matrix is very important to the computer vision.
It's the very first step of everything in this field, so it's worth to take time to understand it. 
It not only considers the focal length but also the radial distortion parameters.
In addition, it also uses non-linear optimization to minimize the geometric error.

# Running Instruction:
Please put the "Wrapper.py" file at the same directory as the "utils" folder.
The "Wrapper.py" file has 5 arguments.
The first two are:
1. --ImagePath, the default value is 'Calibration_Imgs/'. It's the base path of images.
2. --SavePath, the default value is 'Output/undistort_Imgs/'. It's the base path of saving images.

Please make sure you have those folders at the default directory or manually input the directory through command line.

The other three are:
They are related to different approaches, and will all give us the same result.
So, I suggest you using the default value.
3. --Normalize, default=False
4. --NormalizeMode, default=1
5. --CovertR, default=True

# Sample Running Command
$ cd "directory of Wrapper.py"
$ python Wrapper.py --ImagePath="Your Image Path" --SavePath="Your Save Path"
