# Visual Odometry 
In this project, I implemented a visual odometry that calculated the displacements from a video of a car driving around.  
I utilized the epipolar geometry to recover camera poses from frame to frame, then used triangulation to refined the poses.  Finally, I plot the displacements on the graph and compared it with the result from the OpenCV built in function(Ground Truth.

# Steps:
1. Get the feature points using SIFT algorithm.
2. Match the points between two frame.
3. Use RANSAC to reject outliers.
4. Estimate the Fundamental Matrix and Essential Matrix.
5. Recover poses from the Essential Matrix.
6. Use the triangulation to refine the poses.
7. Repeat each steps.

# Future works:
The errors are mainly came from RANSAC, which might screw up the estimation of the Fundamental matrix. If I change the RANSAC part to the OpenCV built in function, the results will be perfectly same. I believe there are some alogorithms can further refine the inliers and give me a better estimation of the Fundamental matrix. 
