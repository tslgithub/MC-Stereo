#!/bin/bash
#!/bin/bash

mode_name=sceneflow
mode_name=kitti12
mode_name=kitti15
mode_name=eth3d
python3 demo.py \
	--left_imgs '/mnt/Data2/depth/depth20250102/left6/*.png' \
  --right_imgs '/mnt/Data2/depth/depth20250102/right6/*.png' \
  --restore_ckpt ./weights/${mode_name}/mc-stereo_${mode_name}.pth \
  --output_directory /mnt/Data2/depth/depth20250102/result/MC-Stereo/${mode_name}
