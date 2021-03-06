# SimData
About our simulated world, how to read depth maps,  make segmentation labels match with other datasets, and more 

## Description of the data

Our simulated world datasets can be found on the Mila cluster in `/network/tmp1/ccai/data/munit_dataset/simdata/`. 
There are several datasets available: 
- `Unity1000R_fL` : 1000 unique viewpoints, with flood height ~ 50 cm, resolution : (2560 * 1440)
- `Unity11K_R` : augmented version of the above dataset. For each viewpoint, 11 images were taken, slightly moving the camera up-down-left-right
-  `Unity1000R_fL_lowRes` : `Unity1000R_fL` dataset in lower resolution (1200*900)
- `Unity1000R_fL_lowRes_plus1m` : same as above dataset but with water height ~1.5 m 

The data from the simulator provides for each snapshot of the world: 
- Original image (non-flooded)
- Flooded image 
- Binary mask of the area of the flood
- Depth image
- Semantic segmentation image : for both flooded and non-flooded scenario. Typically non-flooded semantic segmentation maps will have name '0100.png' and flooded ones '0100w.png'
- json file with camera parameters


### Depth images

The depth maps are provided as RGBA images. Depth is encoded in the the following way: 
 - The information from the simulator is (1 - LinearDepth (in [0,1])).   
 `far` corresponds to the furthest distance to the camera included in the depth map. 
        `LinearDepth * far` gives the real metric distance to the camera. 
- depth is first divided in 31 slices encoded in the R channel with values ranging from 0 to 247 
- each slice is divided again in 31 slices, whose values are encoded in the G channel
- each of the G slices is divided into 256 slices, encoded in the B channel
    In total, we have a discretization of depth into `N = 31*31*256 - 1` possible values, each value covering a range of 
    far/N meters.   
    Note that, what we encode here is  `1 - LinearDepth` so that the furthest point is [0,0,0] (which is sky) 
    and the closest point is [255,255,255] 
    The metric distance associated to a pixel whose depth is (R,G,B) is : 
    d = (far/N) * [((247 - R)//8)*256*31 + ((247 - G)//8)*256 + (255 - B)]  
    This is the same as :
    d = far* ( 1 - ((R//8)*256*31 + (G//8)*256 + B)/N )
      

### Segmentation images 
Segmentation masks are provided for the flooded version of the images. The 10 classes were merged from the [Cityscapes](https://www.cityscapes-dataset.com/) dataset labels. 
The following table provides the correspondence between classes and colors: 

 
| Label | Description |  RGBA | Color |  Cityscapes labels
| ----- | ----- | ----- | ----- | ----- |
| Water|Water generated by the simulator    |[0, 0, 255, 255] |![#0000ff](https://placehold.it/15/0000ff/000000?text=+) | None
|Ground|Horizontal ground-level structures (road, roundabouts, parking)  |[55, 55, 55, 255] |![#373737](https://placehold.it/15/373737/000000?text=+) | 0, 1 (Road, Sidewalk)
| Building|Buildings, walls, fences| [0, 255, 255, 255]|![#00ffff](https://placehold.it/15/00ffff/000000?text=+) | 2, 3, 4 
|Traffic items| Poles, traffic signs, traffic lights | [255, 212, 0, 255]|![#ffd400](https://placehold.it/15/ffd400/000000?text=+)  | 5, 6, 7
|Vegetation| Trees, hedges, all kinds of vertical vegetation | [0, 255, 0, 255]|![#00ff00](https://placehold.it/15/00ff00/000000?text=+)| 8 
|Terrain| Grass, all kinds of horizontal vegetation, soil, sand | [255, 97, 0, 255] | ![#ff6100](https://placehold.it/15/ff6100/000000?text=+)| 9
|Sky| Open sky | [8, 19, 49, 255] | ![#081331](https://placehold.it/15/081331/000000?text=+) | 10
|Car| This includes only cars | [255, 0, 0, 255] | ![#ff0000](https://placehold.it/15/ff0000/000000?text=+) | 13
|Trees| Some trees are seen as 2D in Unity and not segmented |  [0, 0, 0, 0]
|Truck| Vehicle with greater dimensions  than car (fixed threshold TBD) | | |  14, 15, 16
|Person| Not in the dataset for now| | | 11, 12

 <!--- Note: figure out the Tree labels, since this may introduce noise in the training --->
Even though some categories are not yet included in the simulated dataset, we choose specific colors to represent them in order to convert segmentation maps obtained with 19-class cityscapes to our simulated dataset labels. 

### JSON files

The json files contain the following information:
- `CameraPosition`: camera *absolute* coordinates  in meters- the origin is not the ground but the origin of the simulated world
- `CameraRotation`: pitch (x) , yaw (y), roll (z) in degrees from 0 to 360 (for pitch the direction of the rotation is from down to up)
- `CameraFar`: how far we compute the depth map 
- `CameraFOV`: vertical field of view in degrees
- `WaterLevel`: absolute level of water in meters
