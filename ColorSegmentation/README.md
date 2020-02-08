# Color Segmentation: Buoy Detection
Using algorithm: EM, GMM

Output video: https://youtu.be/qs9g0fOCxBE

in this project, my goal is to detect three buoys with different color(yellow, green, orange). I will briefly explain what steps I did to detect the buoy.

1. I croped the images of three buoys to generate training data.
2. I implemented the EM algorithm. Feel free to download the em.py file and run it to see the effect of the EM algorithm.
3. I calculated the cumulative histograms of three buoys in the training data to decide what dimension of the Gaussian function I'll use.
3. I train the GMM (Gaussian Mixture Model) with the training data.
4. I used two setps prediction to refine the result. Basically, I used a looser threshold of prediction to filter out most of the image. Then use a tighter threshold to target the buoy with the right color distribution.
5. Draw contours.

It's a good practice of EM and GMM.