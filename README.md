# imageAugmentationPipeline
an easily extendable and descriptive image augmentation pipeline for tensorflow 2


## TODO: 
  * handle exceptions when crop is not entirely in image anymore in getRandomPositionZoomCrop
  * add flexible crop operation by inheriting from an abstract cropping class and adding that as new input parameter in ImageListAugmentationPipeline.py
  * add functionality for computing multiple random crops from an image
