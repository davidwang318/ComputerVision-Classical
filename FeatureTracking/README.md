# Feature Tracking using Inverse Lucas-Kanade algorithm

Using algorithm: Inverse Lucas-Kanade, Gauss-Newton algorithm

Output video: https://youtu.be/Ni1K11pzmyE

The main idea of lucas-kanade algorithm is to match a subregion of the picture to the template image. To match the image, we need to use the affine transform that has 6 DOFs.

The cost function is the square error of the difference of pixel values that matched.

To optimize the cost function, it uses Gauss-Newton algorithm, wich basically is a kind of gradient decent that uses second order taylor-serise to linearlize the cost function.

The main steps of LK algorithm are as follows:
1. Warp the image.
2. Compute the error.
3. Solve the optimization question using Gauss-Newton algorithm.
4. Compute the update: delta(p), which is the parameters in the affine transform.
5. update parameters and iterate again to find the best parameters.

However, in the 3 & 4 steps, we need to compute a Hessian Matrix which costs a lot of computations. So, in this project, I implemented the Inverse Lucas-Kanade algorithm.

To be short, the original version warp the image back to template that keeps changing along with the time. The inverse version still warps back the original image, but also warps the template image to align the image. Because the temaplate image always remains the same, so the Hessian matrix is the same. And because the affin transform is a linear operation, so it could be inversed. So we can still use the affine transform of the template to update the parameters of the affine transform of the image.