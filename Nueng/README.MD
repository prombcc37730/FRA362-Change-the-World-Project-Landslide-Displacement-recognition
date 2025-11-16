# FRA362 Studio VI : Change the World Project

### Topic : Object Displacement Recognition on Landslide image take by drone
#### Description
    This Project aim to build a system that can recognize an object displacement by using 
    computer vision. We aim to make compare process faster, less than 10 minute, to reduce
    the risk of miss and delay esitimation.

## Process on the system (Not final)
1. Image preparation -> this process make image has most the same detail like alignment by image tranformation using static reference frame, adjust light and focus etc.
2. Feature detection -> this process mask an object we interest on call 'Feature' and find it position on image before we do compare process.
3. Feature comparasion -> this process is compare and find difference Feature position using two image to find it.

