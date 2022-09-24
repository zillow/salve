

## Understanding the use of coordinate frames in SALVe

We start from ZInD's ground truth annotations, provided in a left-handed global coordinate frame. We convert these to a right-handed global coordinate frame:
<img width="1552" src="https://user-images.githubusercontent.com/16724970/191420394-272a7ba2-aac7-4cdf-b757-6da0d5eecfab.png">


In general, we employ 4 coordinate frames throughout the repo: spherical (panoramic), room Cartesian, world-normalized Cartesian, and world-metric Cartesian.
<img width="1273" alt="Screen Shot 2022-09-21 at 1 21 44 AM" src="https://user-images.githubusercontent.com/16724970/191420706-80b44f22-2f57-4570-848c-6a153066bfae.png">

Transformation (R,t) followed by a reflection over the y-axis is equivalent to a transformation by (R^T,-t) followed by no reflection.
<img width="1699" src="https://user-images.githubusercontent.com/16724970/192112179-3eda2ccb-974f-44ef-ac9b-0537d297f39d.png">

<img width="1013" src="https://user-images.githubusercontent.com/16724970/192112180-37c63507-1bd2-48fe-a807-86b98b69237d.png">
