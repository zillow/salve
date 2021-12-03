
## ResNet-18, floor and ceiling, RGB-only
- `model_results_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_06_25/2021_06_26_08_38_09"`
- `model_results_dir = "/mnt/data/johnlam/ZinD_trained_models_2021_06_25/2021_06_26_08_38_09"`

## ResNet-50, floor and ceiling, RGB-only
- `model_results_dir = "/mnt/data/johnlam/ZinD_trained_models_2021_06_25/2021_06_28_07_01_26"`
- `config_fpath = "afp/configs/2021_07_15_resnet50_ceiling_floor_rgbonly_test_set_inference.yaml"`
- `serialization_save_dir = "2021_07_15_serialized_edge_classifications_v2"`

# ResNet-50, floor and ceiling, RGB-only, more tours (GT WDO)
- `model_results_dir = "/mnt/data/johnlam/ZinD_trained_models_2021_07_24/2021_07_26_14_42_49"`
- `config_fpath = "afp/configs/2021_07_24_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "2021_07_28_serialized_edge_classifications"`

# ResNet-50, floor and ceiling, RGB only, 186 tours, but low-res (predicted WDO)
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_21_22_13_20"`
- `config_fpath = "afp/configs/2021_10_22_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_10_22___ResNet50_186tours_serialized_edge_classifications_test2021_11_02"`

# ResNet-50, floor and ceiling, RGB only, 373 tours, but (predicted WDO), low-res
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_25_14_31_23"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_25_14_31_23/2021_10_22_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_10_26__ResNet50_373tours_serialized_edge_classifications_test2021_11_02"`

# ResNet-152, floor and ceiling, RGB only, 435 tours, (predicted WDO), low-res
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_26_16_23_16"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_26_16_23_16/2021_10_26_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02"`
- `serialization_save_dir = "/data/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test109buildings_2021_11_16"`

# ResNet-152, ceiling only, RGB only, 587 tours, (predicted-WDO), low-res
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_04/2021_11_04_10_01_06"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_04/2021_11_04_10_01_06/2021_11_04_resnet152_ceilingonly_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test2021_11_08"`
- `serialization_save_dir = "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test2021_11_12"`
- `serialization_save_dir = "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test109buildings_2021_11_16"`

# ResNet-152, floor only, RGB only, 587 tours, (predicted-WDO), low-res
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_06/2021_11_08_07_40_48"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_06/2021_11_08_07_40_48/2021_11_04_resnet152_flooronly_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test2021_11_09" # tested at 93.9 acc.`
- `serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test2021_11_11" # tested at 94.7 acc.`
- `serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test109buildings_2021_11_16"`

# ResNet-152, layout-only, XYZ tours (predicted WDO), low-res (FAILED AT 10 epochs)
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_11_19_45_07"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_11_19_45_07/2021_11_10_resnet152_layoutonly.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_10__ResNet152layoutonly__877tours_serialized_edge_classifications_test2021_11_15"`

# ResNet-152, layout-only, XYZ tours (predicted WDO), low-res (TRAINED 15/50 epochs so far, will go to completion0
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_13_13_28_10"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_13_13_28_10/2021_11_10_resnet152_layoutonly.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test2021_11_15"`
- `serialization_save_dir = "/data/johnlam/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test109buildings_2021_11_16"`

# ResNet-152, floor and ceiling, RGB only, 587 tours, (predicted WDO), low-res, with equal data now as floor-only or ceiling only
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_09/2021_11_19_21_42_11"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_09/2021_11_19_21_42_11/2021_11_09_resnet152_ceiling_floor_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_09__ResNet152floorceiling__587tours_serialized_edge_classifications_test109buildings_2021_11_23"`

## ResNet-152, GT WDO and GT layout (350 training tours)
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_23/2021_11_23_11_22_47"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_23/2021_11_23_11_22_47/2021_11_23_resnet152_ceiling_floor_rgbonly_GT_WDO_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_23_ResNet152floorceiling_GT_WDO_350tours_serialized_edge_classifications_2021_11_24"`

## ResNet-152, GT WDO and GT layout (817 tours)
- `model_results_dir = johnlam@se1-rmx-gpu-002:/data/johnlam/ZinD_trained_models_2021_11_29/2021_11_29_21_08_01`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_29/2021_11_29_21_08_01/2021_11_29_resnet152_ceiling_floor_rgbonly_GT_WDO_expandeddata_no_photometric_augment.yaml"`
- `serialization_save_dir = /data/johnlam/2021_11_29_ResNet152floorceiling_GT_WDO_817tours_serialized_edge_classifications_2021_12_02"`
