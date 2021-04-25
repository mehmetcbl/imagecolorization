@echo off


set loopcount=1
set max_depth=20
set avg_patch_size=39
set sum_patch_size=8
set hog_patch_size=3
set sd_patch_size=4
set neigh_patch_size=1
set use_distance_features=True
set use_sd_features=False
set use_sum_features=True
set use_avg_features=True
set use_edge_features=False
set use_haris_corner=True
set use_hog_features=False
set use_fast_feature=True
set use_orb_feature=False
set use_neighborhood_features=False
set save_to_file=True

:loop
jupyter nbconvert --to python oemo.ipynb

@echo off
set count=50
setlocal EnableDelayedExpansion

for /L %%i in (1, 1, %count%) do (
    set "formattedValue=000000%%i"

    python oemo.py --image_name p!formattedValue:~-3! --max_depth %max_depth% --avg_patch_size %avg_patch_size% --sum_patch_size %sum_patch_size% --hog_patch_size %hog_patch_size% --sd_patch_size %sd_patch_size% --neigh_patch_size %neigh_patch_size% --use_distance_features %use_distance_features% --use_sd_features %use_sd_features% --use_sum_features %use_sum_features% --use_avg_features %use_avg_features% --use_edge_features %use_edge_features% --use_haris_corner %use_haris_corner% --use_hog_features %use_hog_features% --use_fast_feature %use_fast_feature% --use_orb_feature %use_orb_feature% --use_neighborhood_features %use_neighborhood_features% --save_to_file %save_to_file%
)
pause