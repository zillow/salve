"""Vanishing angle inspection / export."""

import glob
from pathlib import Path
from multiprocessing import Pool
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import salve.common.posegraph2d as posegraph2d
import salve.utils.csv_utils as csv_utils
import salve.utils.io as io_utils
import salve.utils.rotation_utils as rotation_utils
from salve.common.sim2 import Sim2
from salve.dataset.zind_partition import DATASET_SPLITS

import vanishing_point as vanishing_point


def compare_vanishing_angle_csv_data() -> None:
    """ """
    fpath1 = "/Users/johnlambert/Downloads/2022_10_06_vanishing_angles/vanishing_angles_0801.csv"
    fpath2 = "/Users/johnlambert/Downloads/2022_10_06_vanishing_angles/vanishing_angles.csv"

    data1 = csv_utils.read_csv(fpath=fpath1, delimiter = ",")

    data2 = csv_utils.read_csv(fpath=fpath2, delimiter = ",")

    def parse_dict(data):
        return {e["entityid"]: float(e["vanishing_angle"]) for e in data}

    data1 = parse_dict(data1)
    data2 = parse_dict(data2)

    import pdb; pdb.set_trace()

    assert set(data1.keys()) == set(data2.keys())

    for k in data1.keys():
        assert data1[k] == data2[k]


def vis_rotated_pano() -> None:

    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"

    building_ids = ["0564"]
    floor_id = "floor_01"

    # building_id = "0308"
    # floor_id = "floor_02"
    # pano_id = 60
    # Vanishing angle: -34.3
    # Angle should be 90 deg. after the correction.

    #pano_id = 18

    #building_ids = [Path(fpath).stem for fpath in glob.glob("/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_08_12/vanishing_angle/*")]
    for building_id in building_ids:

        vanishing_angle_dir = "/Users/johnlambert/Downloads/2022_10_06_lsd_vanishing_angles"

        pano_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")
        for pano_fpath in pano_fpaths:
            fname_stem = Path(pano_fpath).stem
            pano_id = int(fname_stem.split("_")[-1])
            floor_id = fname_stem.split("_partial")[0]

            # json_fpath = f"{vanishing_angle_dir}/{building_id}/{fname_stem}.json"
            # vanishing_angle_deg = io_utils.read_json_file(json_fpath)["vanishing_angle_deg"]

            fpath = f"/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_08_12/vanishing_angle/{building_id}.json"
            data = io_utils.read_json_file(fpath)
            vanishing_angle_deg = data[str(pano_id)]
            print("Vanishing angle: ", vanishing_angle_deg)

            plot_aligned_pano(raw_dataset_dir, building_id, floor_id, pano_id, vanishing_angle_deg)



def plot_aligned_pano(raw_dataset_dir: str, building_id: str, floor_id: str, pano_id: int, vanishing_angle_deg: float) -> None:
    """ """
    floor_pg = posegraph2d.get_gt_pose_graph(building_id=building_id, floor_id=floor_id, raw_dataset_dir=raw_dataset_dir)


    pano_data = floor_pg.nodes[pano_id]

    coord_frame = "worldnormalized"
    pano_data.plot_room_layout(
        coord_frame = coord_frame,
        show_plot = False,
        scale_meters_per_coordinate = None,
    )

    scale_meters_per_coordinate = 1.0
    pano_position = np.zeros((1, 2))
    if coord_frame in ["worldnormalized", "worldmetric"]:
        pano_position_local = pano_position
        pano_position_world = pano_data.global_Sim2_local.transform_from(pano_position_local)
        pano_position = pano_position_world * scale_meters_per_coordinate

    aligned_Sim2_unaligned = Sim2(R=rotation_utils.rotmat2d(theta_deg=vanishing_angle_deg), t=np.zeros(2), s=1.0)
    unaligned_Sim2_aligned = aligned_Sim2_unaligned.inverse()
    pano_data.global_Sim2_local = pano_data.global_Sim2_local.compose(unaligned_Sim2_aligned)

    print("\tAligned angle: ", pano_data.global_Sim2_local.theta_deg)

    point_ahead = np.array([0, 1]).reshape(1, 2)
    if coord_frame in ["worldnormalized", "worldmetric"]:
        point_ahead = pano_data.global_Sim2_local.transform_from(point_ahead) * scale_meters_per_coordinate

    plt.plot([pano_position[0, 0], point_ahead[0, 0]], [pano_position[0, 1], point_ahead[0, 1]], color="b")
    plt.axis("equal")
    plt.title(f"Building {building_id}, {floor_id}, Pano {pano_id}")
    plt.show()

    #def compute_vp_correction(i2Si1: Sim2, vp_i1: float, vp_i2: float) -> float:


def compute_vanishing_angles_all_buildings():
    """ """
    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"
    save_dir = "/Users/johnlambert/Downloads/2022_10_06_lsd_vanishing_angles"

    #building_ids = ["0564"] # ["0308"]
    building_ids = DATASET_SPLITS["test"]

    args_list = []

    for building_id in building_ids:
        pano_fpaths = glob.glob(f"{raw_dataset_dir}/{building_id}/panos/*.jpg")
        for pano_fpath in pano_fpaths:
            args = (
                pano_fpath, # image_fpath
                save_dir, # output_prefix
                0.7, # q_error
                3 # n_refine_iters
            )
            args_list.append(args)
            # vanishing_point.compute_vanishing_point(args)

    num_processes = 10
    #import pdb; pdb.set_trace()

    with Pool(num_processes) as p:
        p.starmap(vanishing_point.compute_vanishing_point, args_list)





def find_missing_vanishing_angles() -> None:
    """
        Missing pano 10 in Building 0844
        Min:  -44.977729621462146
        Mean:  0.2687952118384579
        Max:  44.98641930733205

        Missing for ['0000', '0001', '0002', '0004', '0005', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030', '0031', '0032', '0033', '0035', '0036', '0037', '0038', '0039', '0040', '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0048', '0049', '0050', '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '0060', '0061', '0062', '0063', '0064', '0065', '0066', '0067', '0069', '0070', '0071', '0072', '0073', '0074', '0075', '0076', '0077', '0078', '0079', '0080', '0081', '0082', '0084', '0085', '0086', '0087', '0088', '0089', '0090', '0091', '0092', '0093', '0094', '0095', '0096', '0097', '0098', '0099', '0100', '0101', '0102', '0103', '0105', '0106', '0107', '0108', '0109', '0110', '0111', '0112', '0113', '0114', '0115', '0117', '0118', '0119', '0120', '0121', '0122', '0123', '0124', '0125', '0126', '0127', '0128', '0129', '0130', '0131', '0132', '0133', '0134', '0135', '0136', '0137', '0138', '0139', '0140', '0141', '0142', '0143', '0144', '0145', '0146', '0147', '0148', '0149', '0150', '0151', '0152', '0153', '0154', '0155', '0156', '0157', '0158', '0159', '0161', '0162', '0163', '0164', '0165', '0166', '0167', '0168', '0169', '0170', '0171', '0172', '0173', '0174', '0175', '0176', '0177', '0178', '0179', '0180', '0181', '0182', '0183', '0184', '0185', '0186', '0187', '0188', '0189', '0190', '0191', '0192', '0193', '0194', '0195', '0196', '0197', '0198', '0199', '0200', '0201', '0202', '0203', '0204', '0205', '0206', '0207', '0208', '0209', '0210', '0211', '0212', '0213', '0214', '0215', '0216', '0217', '0218', '0219', '0220', '0221', '0222', '0223', '0224', '0225', '0226', '0227', '0228', '0229', '0230', '0231', '0232', '0233', '0234', '0237', '0240', '0244', '0253', '0256', '0258', '0260', '0268', '0269', '0270', '0271', '0272', '0273', '0274', '0275', '0276', '0277', '0278', '0279', '0280', '0281', '0282', '0283', '0284', '0285', '0286', '0287', '0288', '0289', '0290', '0291', '0292', '0298', '0301', '0302', '0303', '0305', '0307', '0309', '0310', '0311', '0312', '0313', '0314', '0315', '0316', '0317', '0318', '0319', '0320', '0321', '0322', '0323', '0324', '0325', '0326', '0327', '0328', '0329', '0330', '0331', '0332', '0333', '0335', '0338', '0339', '0347', '0348', '0355', '0357', '0358', '0359', '0360', '0361', '0362', '0363', '0368', '0371', '0375', '0382', '0383', '0384', '0385', '0388', '0397', '0398', '0399', '0400', '0402', '0403', '0405', '0407', '0410', '0415', '0416', '0417', '0418', '0419', '0422', '0426', '0427', '0430', '0437', '0438', '0439', '0444', '0456', '0459', '0460', '0462', '0463', '0467', '0469', '0471', '0472', '0473', '0480', '0481', '0482', '0484', '0488', '0489', '0494', '0499', '0501', '0502', '0504', '0505', '0506', '0511', '0513', '0516', '0518', '0519', '0520', '0527', '0529', '0531', '0533', '0538', '0541', '0542', '0544', '0547', '0549', '0551', '0556', '0558', '0561', '0562', '0563', '0566', '0567', '0569', '0571', '0576', '0583', '0584', '0587', '0592', '0597', '0598', '0602', '0603', '0604', '0607', '0608', '0610', '0613', '0616', '0623', '0630', '0631', '0633', '0635', '0636', '0639', '0642', '0643', '0645', '0647', '0650', '0655', '0656', '0657', '0658', '0660', '0661', '0662', '0663', '0666', '0667', '0670', '0671', '0675', '0678', '0683', '0686', '0690', '0692', '0693', '0695', '0696', '0697', '0698', '0699', '0700', '0701', '0702', '0703', '0704', '0705', '0706', '0707', '0708', '0709', '0710', '0711', '0712', '0715', '0718', '0721', '0722', '0724', '0732', '0734', '0735', '0737', '0739', '0740', '0741', '0742', '0743', '0744', '0745', '0746', '0748', '0753', '0759', '0770', '0773', '0776', '0778', '0780', '0781', '0782', '0783', '0784', '0785', '0786', '0787', '0788', '0789', '0790', '0791', '0793', '0794', '0795', '0796', '0797', '0798', '0799', '0803', '0813', '0817', '0820', '0822', '0826', '0827', '0829', '0830', '0836', '0837', '0850', '0851', '0852', '0856', '0857', '0858', '0859', '0860', '0862', '0874', '0878', '0880', '0882', '0886', '0892', '0893', '0896', '0897', '0898', '0899', '0900', '0901', '0904', '0908', '0913', '0914', '0916', '0920', '0929', '0936', '0937', '0938', '0939', '0940', '0941', '0942', '0944', '0949', '0953', '0954', '0959', '0965', '0970', '0977', '0979', '0980', '0983', '0986', '0990', '0991', '0993', '0995', '0999', '1007', '1012', '1013', '1016', '1020', '1022', '1023', '1025', '1026', '1030', '1035', '1037', '1042', '1044', '1047', '1048', '1051', '1053', '1060', '1062', '1063', '1066', '1076', '1079', '1083', '1088', '1089', '1090', '1091', '1092', '1093', '1094', '1095', '1096', '1098', '1099', '1101', '1102', '1103', '1107', '1113', '1116', '1118', '1119', '1120', '1121', '1126', '1134', '1136', '1137', '1139', '1141', '1143', '1144', '1145', '1153', '1154', '1158', '1165', '1167', '1170', '1171', '1177', '1187', '1188', '1189', '1192', '1196', '1197', '1199', '1202', '1204', '1208', '1211', '1214', '1219', '1220', '1221', '1223', '1225', '1227', '1232', '1234', '1237', '1252', '1255', '1260', '1261', '1272', '1273', '1281', '1284', '1285', '1287', '1288', '1291', '1293', '1300', '1301', '1302', '1303', '1304', '1306', '1309', '1312', '1314', '1316', '1317', '1333', '1340', '1342', '1344', '1348', '1352', '1353', '1355', '1357', '1361', '1363', '1364', '1365', '1371', '1375', '1377', '1379', '1384', '1385', '1387', '1389', '1391', '1400', '1403', '1406', '1411', '1412', '1414', '1418', '1419', '1420', '1423', '1425', '1430', '1432', '1435', '1436', '1441', '1448', '1451', '1452', '1454', '1460', '1464', '1469', '1470', '1471', '1474', '1475', '1486', '1488', '1489', '1492', '1493', '1496', '1506', '1507', '1508', '1509', '1510', '1511', '1513', '1514', '1526', '1534', '1536', '1537', '1539', '1545', '1547', '1548', '1554', '1557', '1560', '1563', '1565', '1567', '1568', '1572', '1574']
    """
    vanishing_angles = []

    raw_dataset_dir = "/Users/johnlambert/Downloads/zind_bridgeapi_2021_10_05"

    for split in ["val", "train", "test"]:

        zind_building_ids = DATASET_SPLITS[split]
        for building_id in zind_building_ids:

            fpath = f"/Users/johnlambert/Downloads/zind_horizon_net_predictions_2022_08_12/vanishing_angle/{building_id}.json"
            if not Path(fpath).exists():
                #print(f"Missing all vanishing angles for building {building_id}")
                continue

            data = io_utils.read_json_file(fpath)

            # Find all panos for this building.
            pano_fpaths = Path(raw_dataset_dir).glob(f"{building_id}/panos/*.jpg")

            for pano_fpath in pano_fpaths:
                fname_stem = Path(pano_fpath).stem
                pano_id = int(fname_stem.split("_")[-1])

                if str(pano_id) not in data:
                    print(f"\tMissing pano {pano_id} in Building {building_id}")
                else:
                    #print("All found")
                    vanishing_angle_deg = data[str(pano_id)]
                    if vanishing_angle_deg is None:
                        print(f"\tMissing pano {pano_id} in Building {building_id}")
                    else:
                        vanishing_angles += [float(vanishing_angle_deg)]

    print("Min: ", np.amin(vanishing_angles))
    print("Mean: ", np.mean(vanishing_angles))
    print("Max: ", np.amax(vanishing_angles))



if __name__ == "__main__":
    #compare_vanishing_angle_csv_data()
    #vis_rotated_pano()
    compute_vanishing_angles_all_buildings()
    #find_missing_vanishing_angles()

