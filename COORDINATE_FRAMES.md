

## Understanding the use of coordinate frames in SALVe

In general, we employ 4 coordinate frames throughout the repo: spherical (panoramic), room Cartesian, world-normalized Cartesian, and world-metric Cartesian.
<img width="1273" alt="Screen Shot 2022-09-21 at 1 21 44 AM" src="https://user-images.githubusercontent.com/16724970/191420706-80b44f22-2f57-4570-848c-6a153066bfae.png">


## Conversion from ZInD's coordinate system to SALVe's coordinate system

We start from ZInD's ground truth annotations, provided in a left-handed global coordinate frame. We convert these to a right-handed global coordinate frame:
<img width="1552" src="https://user-images.githubusercontent.com/16724970/191420394-272a7ba2-aac7-4cdf-b757-6da0d5eecfab.png">
This can be interpreted as looking at a home's floorplan from a vantage point underneath the home, whereas we convert to a vantage point in a bird's eye view (above the home).

We'll now show an example for a specific home from ZInD, (Building 0000, Floor 01). We can see that a transformation (R,t) followed by a reflection over the y-axis is equivalent to a transformation by (R^T,-t) followed by no reflection.
<img width="1699" src="https://user-images.githubusercontent.com/16724970/192112179-3eda2ccb-974f-44ef-ac9b-0537d297f39d.png">

In the top-right, we can see a panorama (Pano 34), and see that the door and window are on the left wall, when facing the garage door. We plot a line segment from each panorama's location along its +y axis in its local coordinate frame (e.g. pointing towards the garage door, as center column of panorama).

Consider the red inset circles on the left and right. We can see that the rotation angle must be negated (equivalent to transposing the rotation matrix) in order for the panorama orientation to stay correct (facing towards a window).
<img width="1013" src="https://user-images.githubusercontent.com/16724970/192112180-37c63507-1bd2-48fe-a807-86b98b69237d.png">
