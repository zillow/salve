"""
Converts a HorizonNet inference result to PanoData and PoseGraph2d objects. Also supports rendering the inference
result with oracle pose.
"""

import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2
import gtsfm.utils.io as io_utils
import imageio
import matplotlib.pyplot as plt

import salve.common.floor_reconstruction_report as floor_reconstruction_report
import salve.common.posegraph2d as posegraph2d
import salve.dataset.zind_data as zind_data
import salve.utils.csv_utils as csv_utils
from salve.common.posegraph2d import PoseGraph2d
from salve.dataset.rmx_madori_v1 import PanoStructurePredictionRmxMadoriV1
from salve.dataset.rmx_tg_manh_v1 import PanoStructurePredictionRmxTgManhV1
from salve.dataset.rmx_dwo_rcnn import PanoStructurePredictionRmxDwoRCNN

from salve.dataset.zind_partition import DATASET_SPLITS

# Path to batch of unzipped prediction files, from Yuguang
# RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/Users/johnlam/Downloads/YuguangProdModelPredictions/ZInD_Prediction_Prod_Model/ZInD_pred"
# RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/mnt/data/johnlam/zind2_john"
# RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/home/johnlam/zind2_john"
# RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/Users/johnlam/Downloads/zind2_john"
#RMX_MADORI_V1_PREDICTIONS_DIRPATH = "/srv/scratch/jlambert30/salve/zind2_john"

# Path to CSV w/ info about prod-->ZInD remapping.
# PANO_MAPPING_TSV_FPATH = "/home/ZILLOW.LOCAL/johnlam/Yuguang_ZinD_prod_mapping_exported_panos.csv"
# PANO_MAPPING_TSV_FPATH = "/home/johnlam/Yuguang_ZinD_prod_mapping_exported_panos.csv"
# PANO_MAPPING_TSV_FPATH = "/Users/johnlam/Downloads/Yuguang_ZinD_prod_mapping_exported_panos.csv"
# PANO_MAPPING_TSV_FPATH = "/srv/scratch/jlambert30/salve/Yuguang_ZinD_prod_mapping_exported_panos.csv"
PANO_MAPPING_TSV_FPATH = "/Users/johnlambert/Downloads/salve/Yuguang_ZinD_prod_mapping_exported_panos.csv"


REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAMES = [
    "rmx-madori-v1_predictions",  # Ethan’s new shape DWO joint model
]

IMAGE_HEIGHT_PX = 512
IMAGE_WIDTH_PX = 1024


def export_horizonnet_zind_predictions(
    query_building_id: str, raw_dataset_dir: str, predictions_data_root: str, export_dir: str
) -> bool:
    """Load W/D/O's predicted for each pano of each floor by HorizonNet.

    TODO: rename this function, since no pose graph is loaded here.

    Note: we read in mapping from spreadsheet, mapping from their ZInD index to these guid
        https://drive.google.com/drive/folders/1A7N3TESuwG8JOpx_TtkKCy3AtuTYIowk?usp=sharing

        For example:
            "b912c68c-47da-40e5-a43a-4e1469009f7f":
            ZinD Image: /Users/johnlam/Downloads/complete_07_10_new/1012/panos/floor_01_partial_room_15_pano_19.jpg
            Prod: Image URL https://d2ayvmm1jte7yn.cloudfront.net/vrmodels/e9c3eb49-6cbc-425f-b301-7da0aff161d2/floor_map/b912c68c-47da-40e5-a43a-4e1469009f7f/pano/cf94fcb5a5/straightened.jpg # noqa
            See this corresponds to 1012 (not 109).

    Args:
        query_building_id: string representing ZinD building ID to fetch the per-floor inferred pose graphs for.
            Should be a zfilled-4 digit string, e.g. "0001"
        raw_dataset_dir:
        predictions_data_root:
        export_dir: 

    Returns:
        Boolean indicating export success for specified building.
    """
    pano_mapping_rows = csv_utils.read_csv(PANO_MAPPING_TSV_FPATH, delimiter=",")

    # Note: pano_guid is unique across the entire dataset.
    panoguid_to_panoid = {}
    for pano_metadata in pano_mapping_rows:
        pano_guid = pano_metadata["pano_guid"]
        dgx_fpath = pano_metadata["file"]
        pano_id = zind_data.pano_id_from_fpath(dgx_fpath)
        panoguid_to_panoid[pano_guid] = pano_id

    # TSV contains mapping between Prod building IDs and ZinD building IDs
    tsv_fpath = REPO_ROOT / "ZInD_Re-processing.tsv"
    tsv_rows = csv_utils.read_csv(tsv_fpath, delimiter="\t")
    for row in tsv_rows:
        #building_guid = row["floor_map_guid_new"] # use for Batch 1 from Yuguang
        building_guid = row["floormap_guid_prod"]  # use for Batch 2 from Yuguang
        # e.g. building_guid resembles "0a7a6c6c-77ce-4aa9-9b8c-96e2588ac7e8"
        #import pdb; pdb.set_trace()

        zind_building_id = row["new_home_id"].zfill(4)

        #print("on ", zind_building_id)
        if zind_building_id != query_building_id:
            continue
        #import pdb; pdb.set_trace()
        if building_guid == "":
            print(f"Invalid building_guid, skipping ZinD Building {zind_building_id}...")
            return None

        # print(f"On ZinD Building {zind_building_id}")
        # # if int(zind_building_id) not in [7, 16, 14, 17, 24]:# != 1:
        # #     continue

        pano_guids = [
            Path(dirpath).stem
            for dirpath in glob.glob(f"{predictions_data_root}/{building_guid}/floor_map/{building_guid}/pano/*")
        ]
        if len(pano_guids) == 0:
            # e.g. building '0258' is missing predictions.
            print(f"No Pano GUIDs provided for {building_guid} (ZinD Building {zind_building_id}).")
            return None

        floor_map_json_fpath = f"{predictions_data_root}/{building_guid}/floor_map.json"
        if not Path(floor_map_json_fpath).exists():
            print(f"JSON file missing for {zind_building_id}")
            return None

        floor_map_json = io_utils.read_json_file(floor_map_json_fpath)

        vanishing_angles_dict = {}
        plt.figure(figsize=(20, 10))
        for pano_guid in pano_guids:

            if pano_guid not in panoguid_to_panoid:
                print(f"Missing the panorama for Building {zind_building_id} -> {pano_guid}")
                continue
            i = panoguid_to_panoid[pano_guid]

            # Export vanishing angles.
            vanishing_angles_dict[ int(i) ] = floor_map_json["panos"][pano_guid]["vanishing_angle"]

            img_fpaths = glob.glob(f"{raw_dataset_dir}/{zind_building_id}/panos/floor*_pano_{i}.jpg")
            if not len(img_fpaths) == 1:
                print("\tShould only be one image for this (building id, pano id) tuple.")
                print(f"\tPano {i} was missing")
                plt.close("all")
                continue

            img_fpath = img_fpaths[0]
            img = imageio.imread(img_fpath)

            img_resized = cv2.resize(img, (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
            img_h, img_w, _ = img_resized.shape

            floor_id = get_floor_id_from_img_fpath(img_fpath)

            gt_pose_graph = posegraph2d.get_gt_pose_graph(
                building_id=zind_building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir
            )

            model_name = "rmx-madori-v1_predictions"
            # plot the image in question
            # for model_name in model_names:
            # print(f"\tLoaded {model_name} prediction for Pano {i}")
            # model_prediction_fpath = (
            #     f"{predictions_data_root}/{building_guid}/floor_map/{building_guid}/pano/{pano_guid}/{model_name}.json"
            # )
            #root = "/Users/johnlambert/Downloads/ZInD_HNet_Prod_Predictions_Part3/predictions"
            root = "/Users/johnlambert/Downloads/ZInD_HNet_Prod_Predictions_Part4/predictions"
            model_prediction_fpath = f"{root}/{building_guid}/floor_map/{building_guid}/pano/{pano_guid}/{model_name}.json"

            if not Path(model_prediction_fpath).exists():
                print(
                    "Home too old, no Madori predictions currently available for this building id (Yuguang will re-compute later).",
                    building_guid,
                    zind_building_id,
                )
                # skip this building.
                return None

            prediction_data = io_utils.read_json_file(model_prediction_fpath)
            assert len(prediction_data) == 1

            save_fpath = f"{export_dir}/horizon_net/{query_building_id}/{i}.json"
            Path(save_fpath).parent.mkdir(exist_ok=True, parents=True)
            io_utils.save_json_file(save_fpath, prediction_data[0])

            if model_name == "rmx-madori-v1_predictions":
                pred_obj = PanoStructurePredictionRmxMadoriV1.from_json(prediction_data[0]["predictions"])
                if pred_obj is None:  # malformatted pred for some reason
                    continue

            import numpy as np
            render_on_pano = False # np.random.rand() < 0.02 # True True # 
            if render_on_pano:
                plt.imshow(img_resized)
                pred_obj.render_layout_on_pano(img_h, img_w)
                plt.title(f"Pano {i} from Building {zind_building_id}")
                plt.tight_layout()
                os.makedirs(f"prod_pred_model_visualizations_2022_07_18_bridge/{model_name}_bev", exist_ok=True)
                plt.savefig(
                    f"prod_pred_model_visualizations_2022_07_18_bridge/{model_name}_bev/{zind_building_id}_{i}.jpg",
                    dpi=400,
                )
                # plt.show()
                plt.close("all")
                plt.figure(figsize=(20, 10))

        vanishing_angles_building_fpath = f"{export_dir}/vanishing_angle/{query_building_id}.json"
        io_utils.save_json_file(vanishing_angles_building_fpath, vanishing_angles_dict)
        # Success
        return True

    return False


def get_floor_id_from_img_fpath(img_fpath: str) -> str:
    """Fetch the corresponding embedded floor ID from a panorama file path.

    For example,
    "/Users/johnlam/Downloads/zind_bridgeapi_2021_10_05/0109/panos/floor_01_partial_room_03_pano_13.jpg" -> "floor_01"
    """
    fname = Path(img_fpath).name
    k = fname.find("_partial")
    floor_id = fname[:k]

    return floor_id


if __name__ == "__main__":
    """ """
    #predictions_data_root = "/srv/scratch/jlambert30/salve/zind2_john"
    #predictions_data_root = "/Users/johnlambert/Downloads/zind_hnet_prod_predictions/zind2_john/zind2_john"
    #predictions_data_root = "/Users/johnlambert/Downloads/zind_hnet_prod_predictions/ZInD_pred"

    # New as of 2022/07/16
    #predictions_data_root = "/Users/johnlambert/Downloads/ZInD_HNet_Prod_Predictions_Part3/ZinD/downloads"
    # New as of 2022/07/18
    predictions_data_root = "/Users/johnlambert/Downloads/ZInD_HNet_Prod_Predictions_Part4/predictions"

    #raw_dataset_dir = "/srv/scratch/jlambert30/salve/zind_bridgeapi_2021_10_05"
    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"
    #zind_building_ids = ["0715"]
    zind_building_ids = DATASET_SPLITS["test"] # "val"] #  ] # ""train"] #
    #import pdb; pdb.set_trace()

    failed_building_ids = []

    export_dir = "/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_07_18_"

    #for each home in ZInD:
    for query_building_id in zind_building_ids:

        #query_building_id = "0588"
        # #     continue
        #import pdb; pdb.set_trace()

        success = export_horizonnet_zind_predictions(
            query_building_id=query_building_id,
            raw_dataset_dir=raw_dataset_dir,
            predictions_data_root=predictions_data_root,
            export_dir=export_dir
        )
        if not success:
            print(f"No GT found for Building {query_building_id}")
            failed_building_ids.append(query_building_id)
            print("Missing building IDs: ", failed_building_ids)

"""
# test
Missing building IDs:  ['0990', '0021', '0809', '0966', '1551', '0963', '1169', '0203', '1041', '0957', '0496', '1544', '0668', '0299', '0684', '0534', '1348', '0127', '0494', '0854', '1184', '1566', '0629', '1409', '1485', '1001', '1177', '1388', '0245', '1400', '0100', '0190', '1330', '0038', '1494', '0189', '0964', '1069', '0109', '0431', '0778', '1500', '0880', '1203', '0709', '0308', '0141', '1218', '0336', '0028', '1239', '1130', '1328', '1160', '1248', '0075', '1185', '1207', '1175', '0311', '1401', '0528', '0322', '1167', '0453', '0490', '1490', '0115', '0420', '0354', '0583', '0870', '0297', '0325', '1506', '0152', '0429', '1068', '0406', '1028', '1404', '0744', '0691', '0057', '1027', '0785', '1268', '1050', '0353', '0575', '1398', '0792', '0383', '0090', '1153', '0278', '0097', '0011', '0218', '0181', '0039', '0270', '1075', '1383', '0969', '0819', '1538', '0302', '0010', '0076', '0905', '0564', '0800', '1199', '0165', '1210', '1479', '0444', '0681', '0316', '0579', '0588', '0605', '0742', '1326', '0516', '0157', '1368']
"""

"""
# train
['0919', '0422', '0673', '1308', '1356', '0454', '0895', '0313', '1321', '0300', '0206', '0918', '1080', '0475', '0329', '1573', '0437', '1017', '1450', '1124', '0248', '0843', '0646', '0902', '0154', '1256', '1180', '0901', '0058', '0479', '0682', '1111', '1432', '0121', '0738', '1152', '0170', '1215', '1231', '0044', '1197', '1143', '1174', '1200', '0495', '0400', '0345', '0364', '1086', '0215', '0042', '0674', '0254', '0249', '0672', '1146', '0187', '1477', '0935', '0222', '0928', '1161', '1092', '0016', '1011', '1492', '1151', '0216', '1376', '0526', '0930', '0236', '0940', '0261', '0559', '1156', '0214', '0503', '0724', '0831', '0599', '1108', '0640', '0602', '0736', '0892', '0279', '1435', '1033', '0650', '1362', '1540', '1266', '0538', '0746', '0351', '0017', '1127', '1029', '0151', '0137', '1078', '0604', '1542', '0421', '0641', '0332', '0425', '0156', '0971', '0205', '0860', '0525', '0341', '1345', '0869', '1478', '1046', '1385', '1399', '0907', '1187', '0522', '1358', '0786', '1440', '1109', '0521', '0089', '0272', '0282', '1106', '1142', '0275', '0101', '0068', '1279', '1393', '0073', '0296', '0868', '1521', '0838', '0552', '1148', '0848', '0607', '0149', '0303', '1346', '1517', '1263', '1000', '1561', '1212', '1395', '0719', '0103', '0458', '1491', '0818', '0847', '0284', '0035', '0998', '1425', '0029', '0255', '0810', '1120', '0764', '0274', '0330', '0592', '1463', '1390', '0855', '0411', '1407', '1163', '0123', '0508', '0589', '1209', '0360', '1265', '0442', '1431', '0369', '0349', '0394', '1083', '0985', '0606', '1117', '0665', '1408', '0380', '1138', '0954', '0960', '1105', '0711', '0731', '0750', '1182', '0200', '1250', '0373', '0433', '0543', '0788', '0079', '1292', '0377', '0095', '0546', '0929', '1073', '0225', '1186', '0328', '1015', '0910', '1343', '0390', '0623', '0253', '1179', '0319', '0466', '1296', '0449', '1157', '0233', '1162', '1415', '0573', '0468', '1483', '0088', '0281', '0153', '1455', '0177', '0500', '0207', '1294', '0915', '1194', '0845', '0208', '0968', '1444', '0126', '0927', '1150', '0661', '1006', '0866', '1453', '0512', '1164', '0320', '0445', '0689', '0318', '1406', '0140', '1567', '1201', '0783', '0389', '0837', '0146', '1002', '0464', '0212', '0147', '0789', '1123', '0019', '0049', '0197', '1139', '0984', '0186', '1311', '0551', '1097', '0114', '0694', '1476', '0136', '0609', '0358', '1132', '0343', '1280', '0335', '0570', '0801', '0242', '1297', '0934', '1532', '1381', '0250', '0107', '0619', '0572', '1498', '1021', '0169', '0601', '0061', '1112', '0849', '0052', '0072', '0111', '1422', '0649', '0706', '1560', '0729', '1272', '0994', '1564', '0323', '0743', '1366', '0948', '1234', '0230', '1524', '1056', '0409', '1502', '0404', '0591', '0787', '1410', '0813', '0896', '1275', '1205', '0577', '1487', '1305', '1183', '1004', '1341', '0273', '1449', '0387', '1347', '0553', '0145', '0091', '0612', '0947', '1213', '1247', '1331', '0699', '0315', '0176', '0939', '0708', '0505', '0259', '1493', '1286', '0884', '0105', '1458', '0287', '0202', '1559', '1040', '1061', '0443', '1258', '0922', '0700', '0191', '0628', '0697', '0350', '0440', '1558', '1052', '0497', '0478', '0204', '1374', '0767', '0798', '1318', '0289', '0578', '0096', '1447', '1155', '1504', '0408', '0596', '0938', '1505', '0782', '0171', '1372', '0448', '0996', '0585', '1276', '1349', '1367', '0399', '1535', '1325', '0240', '1070', '0163', '0223', '0617', '0695', '1546', '0441', '0704', '1057', '0762', '0536', '0268', '0307', '0861', '0937', '0053', '0239', '0688', '0988', '0600', '0235', '1114', '0228', '1320', '0727', '0094', '1291', '0415', '1129', '1523', '1445', '0840', '0821', '0698', '1224', '0775', '0131', '0620', '0310', '1439', '0128', '0110', '0376', '1009', '0933', '0077', '0080', '0513', '1443', '0484', '1058', '1038', '1541', '0980', '1481', '0381', '1495', '0013', '0008', '1269', '1024', '0054', '0138', '0740', '0264', '0763', '0842', '1307', '0864', '0710', '0844', '0143', '0210', '0651', '0747', '0636', '0412', '1019', '0701', '0973', '1264', '1315', '1310', '0023', '0166', '0294', '0621', '0262', '0648', '1529', '0046', '1235', '1149', '0823', '0117', '0723', '0517', '1257', '0633', '0669', '1405', '1236', '0162', '0756', '1355', '1543', '1466', '1501', '1547', '0180', '0339', '0857', '0945', '0033', '1570', '1254', '1324', '0532', '0045', '0889', '0975', '0537', '1482', '1402', '1549', '0450', '0150', '0283', '0048', '0112', '1336', '1140', '0796', '0398', '1459', '1074', '0116', '0066', '0060', '1569', '0269', '1522', '0873', '0192', '0820', '0418', '0557', '0909', '0752', '0108', '1240', '0955', '0132', '0005', '0771', '0024', '1230', '0524', '1113', '0317', '0337', '1520', '0022', '0702', '1350', '0309', '0806', '0199', '0758', '0125', '0550', '0265', '0807', '1229', '0768', '0167', '1428', '1064', '1244', '0474', '0816', '1556', '1003', '0846', '1360', '1332', '0447', '1241', '0725', '0558', '0213', '0113', '0032', '0238', '1131', '1221', '1473', '0754', '1277', '1433', '0774', '1226', '1253', '0625', '0368', '1005', '1531', '0987', '0493', '0396', '1392', '1026', '0305', '0106', '0063', '0158', '0031', '0634', '0707', '0876', '0314', '0680', '0211', '1353', '1369', '1527', '1437', '0974', '0290', '0342', '0251', '0745', '1533', '1397', '0306', '0476', '0419', '0733', '0741', '0946', '1259', '0130', '0384', '1178', '0726', '0734', '0976', '0173', '0548', '0327', '0460', '1173', '1391', '0436', '0676', '0078', '0656', '0487', '1034', '0055', '1191', '0507', '0347', '0243', '0026', '1518', '0423', '0050', '0182', '0119', '0824', '0977', '0217', '0814', '0881', '1133', '0571', '0509', '0565', '1045', '0970', '0943', '0685', '0696', '0898', '0555', '0477', '1363', '0530', '0043', '0455', '0580', '0972', '0371', '1267', '0822', '0174', '0765', '1055', '1424', '0716', '0992', '0102', '0568', '0086', '0793', '0386', '0777', '1516', '1421', '0540', '0595', '0083', '0135', '0004', '1299', '0956', '0285', '0978', '1270', '0252', '1309', '0198', '0036', '1454', '0304', '0523', '0586', '0779', '1242', '0402', '0720', '1503', '0766', '0168', '0891', '1499', '1442', '0515', '1319', '0624', '0051', '1351', '1018', '0615', '1361', '0133', '0374', '0594', '1262', '1193', '0815', '1512', '0962', '1571', '0906', '0056', '0830', '0885', '1115', '1255', '1054', '0593', '0263', '1562', '0434', '0841', '1515', '1530', '1081', '0012', '0134', '1166', '0346', '1304', '0797', '0183', '0403', '1135', '0414', '0875', '0333', '1339', '0271', '0658', '1216', '0164', '0334', '1438', '0678', '1396', '0067', '1085', '1118', '0662', '0081', '0638', '1382', '0178', '0092', '0034', '0471', '1110', '1465', '0344', '1426', '0030', '0229', '0965', '0705', '0714', '0545', '0687', '1461', '0257', '1233', '0732', '0267', '0188', '1176', '0065', '0410', '0457', '1416', '1427', '1087', '0903', '0703', '0879', '1071', '0491', '0435', '0582', '0877', '0366', '0175', '1290', '0627', '0378', '1389', '0470', '0015', '0611', '1104', '0194', '0193', '0258', '1283', '1008', '0142', '0890', '0221', '1010', '0912', '0232', '1249', '0958', '0923', '0071', '0690', '1245', '0653', '0413', '0811', '0037', '0730', '0671', '0825', '0312', '0622', '1519', '0085', '0917', '0812', '0834', '0392', '0155', '0361', '0379', '0760', '0626', '1059', '0657', '0486', '1014', '1420', '0416', '1446', '0961', '1020', '0219', '0911', '0666', '0459', '0932', '1354', '0757', '1060', '1313', '1380', '1172', '0828', '0908', '0276', '0926', '1134', '0001', '0247', '0040', '0749', '0677', '1032', '1195', '0554', '0803', '1082', '0014', '1222', '1378', '0160', '1335', '0047', '0002', '0009', '0835', '1337', '1312', '1295', '0735', '0781', '0227', '1322', '1289', '1413', '0196', '0356', '0952', '0632', '1484', '0721', '0301', '1468', '1049', '0574', '0093', '1386', '0122', '0172', '0802', '0007', '1480', '0465', '1228', '1067', '1072', '0452', '0295', '0321', '0883', '0367', '1472', '0401', '0391', '0755', '1039', '0280', '0084', '0129', '1121', '1417', '0717', '0098', '0027', '1170', '0659', '0395', '0451', '1394', '0982', '1273', '0087', '1246', '1243', '0739', '1298', '0871', '0829', '0833', '1329', '0446', '0791', '0853', '1555', '0195', '1217', '0372', '0000', '0936', '0799', '1429', '0772', '0006', '1497', '1462', '0241', '0753', '1237', '0124', '0331', '0560', '0159', '0118', '0082', '1536', '1550', '0365', '0201', '1065', '1100', '0041', '1198', '1116', '1031', '0784', '1373', '1553', '0839', '1552', '0679', '1370', '0652', '0070', '0794', '0925', '0951', '0485', '0483']
"""

"""
# val
['0340', '0613', '0209', '0535', '0899', '0185', '0539', '0139', '0527', '0637', '0148', '0144', '0981', '1434', '1278', '0790', '1122', '0293', '0226', '0769', '0224', '0931', '1467', '1359', '0614', '0020', '0161', '0324', '0352', '1181', '1301', '1147', '0064', '1271', '0804', '0326', '0887', '1274', '0547', '0989', '0062', '0921', '0867', '0428', '0288', '0865', '1225', '1206', '0581', '1190', '0370', '0231', '1159', '0967', '1251', '0751', '0059', '0712', '0780', '0286', '1036', '1456', '0888', '0832', '1282', '1334', '1510', '0808', '0900', '1323', '0393', '0924', '0644', '0950', '0074', '1528', '0069', '1043', '0417', '0266', '0863', '0184', '0246', '1365', '1457', '0461', '0432', '0654', '1128', '0894', '0120', '0942', '0003', '1238', '0618', '1168', '0104', '0220', '0492', '0805', '0424', '0590', '1227', '1525', '0872', '0728', '1084', '0018', '0603', '0099', '1125', '1486', '1302', '1077', '0498', '0277', '0025', '0510', '0761', '1338', '1327', '0664', '0713', '0795', '0997', '0514', '1475', '0770', '0179', '0291']
"""
