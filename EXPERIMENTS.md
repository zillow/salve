
Summary of trained models / Results

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
- `hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"`


# ResNet-152, floor and ceiling, RGB only, 435 tours, (predicted WDO), low-res
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_26_16_23_16"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_10_22/2021_10_26_16_23_16/2021_10_26_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02"`
- `serialization_save_dir = "/data/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test109buildings_2021_11_16"`
- `serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test2021_11_02"`
- `serialized_preds_json_dir = "/data/johnlam/2021_10_26__ResNet152__435tours_serialized_edge_classifications_test109buildings_2021_11_16"`
- `hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"`


# ResNet-152, ceiling only, RGB only, 587 tours, (predicted-WDO), low-res
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_04/2021_11_04_10_01_06"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_04/2021_11_04_10_01_06/2021_11_04_resnet152_ceilingonly_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test2021_11_08"`
- `serialization_save_dir = "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test2021_11_12"`
- `serialization_save_dir = "/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test109buildings_2021_11_16"`
- `serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test2021_11_12"`
- `/data/johnlam/2021_11_04__ResNet152ceilingonly__587tours_serialized_edge_classifications_test109buildings_2021_11_16`
- `hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"`

# ResNet-152, floor only, RGB only, 587 tours, (predicted-WDO), low-res
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_06/2021_11_08_07_40_48"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_06/2021_11_08_07_40_48/2021_11_04_resnet152_flooronly_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test2021_11_09" # tested at 93.9 acc.`
- `serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test2021_11_11" # tested at 94.7 acc.`
- `serialization_save_dir = "/data/johnlam/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test109buildings_2021_11_16"`
- `serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_11_08__ResNet152flooronly__587tours_serialized_edge_classifications_test2021_11_09"`
- `hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"`

# ResNet-152, layout-only, XYZ tours (predicted WDO), low-res (FAILED AT 10 epochs)
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_11_19_45_07"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_11_19_45_07/2021_11_10_resnet152_layoutonly.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_10__ResNet152layoutonly__877tours_serialized_edge_classifications_test2021_11_15"`
- `hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"`

# ResNet-152, layout-only, XYZ tours (predicted WDO), low-res (TRAINED 15/50 epochs so far, will go to completion0
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_13_13_28_10"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_10/2021_11_13_13_28_10/2021_11_10_resnet152_layoutonly.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test2021_11_15"`
- `serialization_save_dir = "/data/johnlam/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test109buildings_2021_11_16"`
- `serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test2021_11_15"`
- `serialized_preds_json_dir = "/data/johnlam/2021_11_10__ResNet152layoutonlyV2__877tours_serialized_edge_classifications_test109buildings_2021_11_16"`
- `hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"`


# ResNet-152, floor and ceiling, RGB only, 587 tours, (predicted WDO), low-res, with equal data now as floor-only or ceiling only
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_09/2021_11_19_21_42_11"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_09/2021_11_19_21_42_11/2021_11_09_resnet152_ceiling_floor_rgbonly_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_09__ResNet152floorceiling__587tours_serialized_edge_classifications_test109buildings_2021_11_23"`
- `hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"`


## ResNet-152, GT WDO and GT layout (350 training tours)
- `model_results_dir = "/data/johnlam/ZinD_trained_models_2021_11_23/2021_11_23_11_22_47"`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_23/2021_11_23_11_22_47/2021_11_23_resnet152_ceiling_floor_rgbonly_GT_WDO_no_photometric_augment.yaml"`
- `serialization_save_dir = "/data/johnlam/2021_11_23_ResNet152floorceiling_GT_WDO_350tours_serialized_edge_classifications_2021_11_24"`
- `hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_20_SE2_width_thresh0.8"`
- `serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_11_23_ResNet152floorceiling_GT_WDO_350tours_serialized_edge_classifications_2021_11_24"`
    
## ResNet-152, GT WDO and GT layout (817 tours)
- `model_results_dir = johnlam@se1-rmx-gpu-002:/data/johnlam/ZinD_trained_models_2021_11_29/2021_11_29_21_08_01`
- `config_fpath = "/data/johnlam/ZinD_trained_models_2021_11_29/2021_11_29_21_08_01/2021_11_29_resnet152_ceiling_floor_rgbonly_GT_WDO_expandeddata_no_photometric_augment.yaml"`
- `serialization_save_dir = /data/johnlam/2021_11_29_ResNet152floorceiling_GT_WDO_817tours_serialized_edge_classifications_2021_12_02"`
- locally `serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_11_29_ResNet152floorceiling_GT_WDO_817tours_serialized_edge_classifications_2021_12_02"`




## From Older Experiments

    # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_13_binary_model_edge_classifications"
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_13_edge_classifications_fixed_argmax_bug/2021_07_13_edge_classifications_fixed_argmax_bug"
    serialized_preds_json_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_06_25/2021_06_28_07_01_26/2021_07_15_serialized_edge_classifications/2021_07_15_serialized_edge_classifications"

    # training, v2
    # serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_07_28_serialized_edge_classifications"

    raw_dataset_dir = "/Users/johnlam/Downloads/ZInD_release/complete_zind_paper_final_localized_json_6_3_21"
    # raw_dataset_dir = "/Users/johnlam/Downloads/2021_05_28_Will_amazon_raw"
    # vis_edge_classifications(serialized_preds_json_dir, raw_dataset_dir)

    hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_alignment_hypotheses_2021_07_14_v3_w_wdo_idxs"

    hypotheses_save_root = "/Users/johnlam/Downloads/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"


    186 tours, low-res, RGB only floor and ceiling. custom hacky val split
    serialized_preds_json_dir = "/Users/johnlam/Downloads/ZinD_trained_models_2021_10_22/2021_10_21_22_13_20/2021_10_22_serialized_edge_classifications"

    373 training tours, low-res, RGB only floor and ceiling, true ZinD train/val/test split
    serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_26_serialized_edge_classifications"
    serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_26_serialized_edge_classifications_v2_more_rendered"

    serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_22___ResNet50_186tours_serialized_edge_classifications_test2021_11_02"
    serialized_preds_json_dir = "/Users/johnlam/Downloads/2021_10_26__ResNet50_373tours_serialized_edge_classifications_test2021_11_02"




    Depth maps are stored on the following nodes at the following places:
    - locally: "/Users/johnlam/Downloads/ZinD_Bridge_API_HoHoNet_Depth_Maps"
    - DGX: "/mnt/data/johnlam/ZinD_Bridge_API_HoHoNet_Depth_Maps"
    - se1-001: "/home/johnlam/ZinD_Bridge_API_HoHoNet_Depth_Maps",

    Hypotheses are saved on the following nodes at the following places:
    - se1-001:
        w/ inferred WDO and inferred layout:
            "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_madori_rmx_v1_2021_10_20_SE2_width_thresh0.65"
        w/ GT WDO + GT layout:
            "/home/johnlam/ZinD_bridge_api_alignment_hypotheses_GT_WDO_2021_11_20_SE2_width_thresh0.8"

    BEV texture maps are saved to the following locations ("bev_save_root"):
        "/Users/johnlam/Downloads/ZinD_BEV_2021_06_24"
        "/Users/johnlam/Downloads/ZinD_BEV_RGB_only_2021_06_25"
        "/mnt/data/johnlam/ZinD_BEV_RGB_only_2021_06_25"
        "/Users/johnlam/Downloads/ZinD_BEV_RGB_only_2021_07_14_v2"
        "/Users/johnlam/Downloads/ZinD_BEV_RGB_only_2021_07_14_v3"
        "/Users/johnlam/Downloads/ZinD_BEV_RGB_only_2021_08_03_layoutimgs_filledpoly"
        "/mnt/data/johnlam/ZinD_07_11_BEV_RGB_only_2021_08_04_ZinD"
        "/Users/johnlam/Downloads/ZinD_07_11_BEV_RGB_only_2021_08_04_ZinD"
        "/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_16"
        "/mnt/data/johnlam/ZinD_Bridge_API_BEV_2021_10_16"
        "/mnt/data/johnlam/ZinD_Bridge_API_BEV_2021_10_17"
        "/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_20_res500x500"
        "/home/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres"  # BEST
        "/Users/johnlam/Downloads/ZinD_Bridge_API_BEV_2021_10_23_debug"

        from GT WDO and GT layout:
            "/home/johnlam/Renderings_ZinD_bridge_api_GT_WDO_2021_11_20_SE2_width_thresh0.8"

    Layout images are saved at:
        # layout_save_root = "/Users/johnlam/Downloads/ZinD_BEV_RGB_only_2021_08_03_layoutimgs"
        # layout_save_root = "/mnt/data/johnlam/ZinD_07_11_BEV_RGB_only_2021_08_04_layoutimgs"
        # layout_save_root = "/Users/johnlam/Downloads/ZinD_07_11_BEV_RGB_only_2021_08_04_layoutimgs"
        # layout_save_root = "/home/johnlam/ZinD_Bridge_API_BEV_2021_10_20_lowres_layoutimgs_inferredlayout"



"""
Nasty depth map estimation failure cases: (from complete_07_10 version)
    # (building, pano_ids)
    "000": [10],  # pano 10 outside
    "004": [10, 24, 28, 56, 58],  # building 004, pano 10 and pano 24, pano 28,56,58 (outdoors)
    "006": [10],
    "981": [5, 7, 14, 11, 16, 17, 28],  # 11 is a bit inaccurate
"""


## Explanation of config files:

    ResNet-50, low-res
        config_name = "2021_10_22_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml

    ResNet-152
        config_name = "2021_10_26_resnet50_ceiling_floor_rgbonly_no_photometric_augment.yaml

    ResNet-152 with just a single modality.
        config_name = "2021_11_04_resnet152_ceilingonly_rgbonly_no_photometric_augment.yaml
        config_name = "2021_11_04_resnet152_flooronly_rgbonly_no_photometric_augment.yaml

    ResNet-152 trained again with equal amount of data as single modality
        config_name = "2021_11_09_resnet152_ceiling_floor_rgbonly_no_photometric_augment.yaml

    ResNet-152 with Layout Only
        config_name = "2021_11_10_resnet152_layoutonly.yaml

    ResNet-152 w/ GT WDO and GT layout.
        config_name = "2021_11_23_resnet152_ceiling_floor_rgbonly_GT_WDO_no_photometric_augment.yaml

    ResNet-152 w/ GT WDO and GT layout, and more data.
        2021_11_29_resnet152_ceiling_floor_rgbonly_GT_WDO_expandeddata_no_photometric_augment.yaml

