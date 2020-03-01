# Introduction
The NaiveAutoPano.py is a version that warped and merged the images to the last imgae. It works perfectly if your images are changed in small angle(< 90 degree).  The RobustAutopano.py is a version that has the ability to find the order of the input images and choose the best one to warp and merge. Works pretty slow, but can handle diffcult tasks, such as irrelevant image or wide angle images.

# Running Instructions
1. Clone the file.
2. $ cd Code/
3. $ python *filename* --InputPath="*inputImage_path*" --InputSet="*inputImage_folder*"  
Example: $ python NaiveAutoPano.py --InputPath="~/Autopano/Data/Train/" --InputSet="Set1/"

The results and detail implementations are in the report. Please download it if you're interested.
