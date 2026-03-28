import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from itertools import product

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate
from utils import get_index_of_A_in_B


class ArgoverseV1DatasetVisual(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 30,
                 margin: float = 50,
                 sample_rate: int = 1) -> None:
        self.root = root
        self.sample_rate = sample_rate
        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')

        self.baseline_result_pickle ="/data2/lhk/InMoE_analyze/vis_output/gen_vis_baseline.submission"
        self.moe_result_pickle = "/data2/lhk/InMoE_analyze/vis_output/gen_vis.submission"

        with open(self.baseline_result_pickle, 'rb') as f:
            self.baseline_result = pd.read_pickle(f)
        with open(self.moe_result_pickle, 'rb') as f:
            self.moe_result = pd.read_pickle(f)
        self.apply_situation = 'visualize InMoE or baseline'#'visualize SIMPL'
        if 'visualize InMoE or baseline' in self.apply_situation:
            all_scenario_ids = set(self.baseline_result.keys()) | set(self.moe_result.keys())
            self.better_scenario_ids = ['3744']#['1158','3288','4241','9728','21979','35750']#['40835','2727','13926','5515']
            #VAL-Uturn['28934', '12514', '9929', '6575', '27287', '28741', '21639', '20567', '35897', '8234', '20270', '33911', '13967', '26702', '14063', '31492', '22904', '30119', '39373', '24063', '26145', '30674', '17994', '22550', '25310', '19590', '13626', '40835', '13765', '8359', '20132', '24', '6875', '26226']
            #intention1['38212', '13005', '17210', '30653', '8487', '34803', '33390', '33705', '29205', '35175', '32127', '6999', '31864', '26717', '16586', '27658', '28790', '19487', '35311', '32162', '6673', '13476', '24332', '25630', '34439', '2056', '3796', '30726', '9162', '36101', '31435', '24392', '28491', '39590', '29920', '6043', '6678', '37497', '18692', '9118', '26225', '21183', '843', '35123', '15133', '16006', '30553', '11377', '20128', '22610', '30657', '7708', '31193', '16128', '5828', '29666', '11768', '22219', '16687', '16345', '28448', '7865', '6263', '30574', '30409', '4610', '18392', '20822', '36153', '35343', '22234', '38270', '28332', '4778', '34984', '24604', '593', '18676', '19480', '18925', '34743', '11736', '11791', '34024', '30420', '28505', '40408', '16992', '41107', '40409', '16696', '13180', '31852', '14901', '26994', '3307', '17016', '4882', '33125', '39861', '16186', '19004', '40780', '31526', '16686', '3550', '9143', '17411', '20997', '36542', '11530', '33360', '8463', '18733', '14644', '25396', '28483', '36513', '22657', '19856', '15477', '7299', '17578', '31955', '18908', '9025', '23006', '24651', '30459', '6512', '31810', '6834', '25456', '16013', '11673', '21275', '545', '18530', '11264', '40010', '22939', '21452', '6615', '37038', '28331', '26258', '38961', '19111', '12520', '5284', '34179', '12727', '6940', '35088', '32020', '5528', '2756', '19321', '1735', '5026', '29548', '25826', '31345', '10692', '1261', '10976', '29029', '3990', '17888', '37598', '4412', '31067', '15339', '34675', '33979', '40953', '35369', '8387', '30649', '15655', '9736', '6973', '11917', '15657', '33131', '2669', '14019', '10672', '29408', '16580', '35124', '2930', '14524', '23337', '5435', '22661', '32240', '37440', '29704', '16508']
            #['30721', '19127', '16319', '3119', '10759', '12068', '17233', '17254', '41016', '28537', '20753', '11089', '8000', '14380', '41103', '4223', '41009', '26088', '23697', '22308', '30922', '32713', '8635', '15359', '39066', '23372', '30999', '2201', '25793', '39300', '38044', '24293', '17931', '9942', '34388', '9673', '24231', '18640', '26172', '16867', '39354', '39931', '20966', '6237', '9650', '32616', '41026', '3191', '881', '20891', '11853', '23825', '11339', '22426', '28999', '7090', '19716', '34695', '10262', '39875', '2525', '499', '10290', '16554', '33113', '35905', '7130', '23295', '24236', '19642', '33398', '1451', '18471', '30774', '39212', '34395', '14777', '35117', '34872', '5355', '33708', '7267', '282', '1600', '4286', '23075', '35479', '13330', '17221', '35052', '2152', '26409', '15672', '8956', '16757', '23061', '10942', '10494', '37724', '2775', '25713', '30670', '22970', '8503', '14171', '8312', '34259', '29786', '25666', '8533', '1382', '32908', '4273', '16314', '21468', '3761', '9590', '16529', '907', '3628', '8931', '30373', '10538', '18798', '6723', '35071', '7162', '12508', '4622', '19640', '10587', '13995', '17639', '39979', '9866', '9097', '30990', '22688', '19329', '22016', '24383', '36288', '900', '21142', '16187', '40238', '9273', '9675', '37261', '7121', '16278', '23062', '5453', '5577', '31224', '15973', '22956', '28812', '12796', '19996', '32545', '672', '29015', '35511', '1485', '15526', '39255', '6010', '13042', '28902', '17987', '37861', '39863', '28449', '19685', '24979', '29881', '13011', '14578', '34705', '23204', '2077', '21120', '7385', '21377', '2275', '40444', '10207', '13724', '30081', '33850', '12625', '29022', '34170', '12004', '6407', '33711', '37072', '12458', '19758'] 
            #intention2['30721', '19127', '16319', '3119', '10759', '12068', '17233', '17254', '41016', '28537', '20753', '11089', '8000', '14380', '41103', '4223', '41009', '26088', '23697', '22308', '30922', '32713', '8635', '15359', '39066', '23372', '30999', '2201', '25793', '39300', '38044', '24293', '17931', '9942', '34388', '9673', '24231', '18640', '26172', '16867', '39354', '39931', '20966', '6237', '9650', '32616', '41026', '3191', '881', '20891', '11853', '23825', '11339', '22426', '28999', '7090', '19716', '34695', '10262', '39875', '2525', '499', '10290', '16554', '33113', '35905', '7130', '23295', '24236', '19642', '33398', '1451', '18471', '30774', '39212', '34395', '14777', '35117', '34872', '5355', '33708', '7267', '282', '1600', '4286', '23075', '35479', '13330', '17221', '35052', '2152', '26409', '15672', '8956', '16757', '23061', '10942', '10494', '37724', '2775', '25713', '30670', '22970', '8503', '14171', '8312', '34259', '29786', '25666', '8533', '1382', '32908', '4273', '16314', '21468', '3761', '9590', '16529', '907', '3628', '8931', '30373', '10538', '18798', '6723', '35071', '7162', '12508', '4622', '19640', '10587', '13995', '17639', '39979', '9866', '9097', '30990', '22688', '19329', '22016', '24383', '36288', '900', '21142', '16187', '40238', '9273', '9675', '37261', '7121', '16278', '23062', '5453', '5577', '31224', '15973', '22956', '28812', '12796', '19996', '32545', '672', '29015', '35511', '1485', '15526', '39255', '6010', '13042', '28902', '17987', '37861', '39863', '28449', '19685', '24979', '29881', '13011', '14578', '34705', '23204', '2077', '21120', '7385', '21377', '2275', '40444', '10207', '13724', '30081', '33850', '12625', '29022', '34170', '12004', '6407', '33711', '37072', '12458', '19758']
            #intention2['40706', '23148', '40127', '6419', '37146', '2131', '11478', '33084', '15782', '8864', '12060', '10927', '8961', '31785', '29410', '11173', '4178', '31472', '33525', '12343', '20724', '13148', '28496', '26151', '35516', '22329', '832', '14327', '15237', '34856', '24753', '20905', '22159', '1492', '20732', '7147', '6223', '29217', '31919', '33162', '31147', '19749', '5890', '3302', '4711', '27301', '27917', '11648', '22339', '35243', '35692', '4432', '4222', '23732', '29319', '3885', '19796', '11734', '33137', '40941', '40068', '18643', '37310', '36816', '8984', '6705', '4903', '34551', '22056', '20076', '27832', '28356', '22208', '3922', '3254', '20929', '14553', '6773', '20391', '2712', '35720', '13098', '11549', '40577', '3504', '20926', '34375', '10963', '29206', '37015', '28984', '4894', '16808', '32417', '30752', '22041', '32405', '2424', '35076', '9601', '25774', '18072', '2117', '22981', '6025', '18873', '12389', '20968', '31606', '37507', '24361', '10576', '6120', '20421', '39923', '29740', '4745', '30851', '8598', '26839', '40495', '40283', '37132', '17783', '9830', '29807', '30970', '32721', '2553', '8802', '5034', '5805', '19704', '25200', '40474', '6066', '27778', '12022', '18116', '23824', '11958', '22861', '28030', '26412', '10216', '26424', '33950', '13716', '31563', '7165', '2847', '37797', '30176', '26929', '24167', '3095', '24218', '2197', '37415', '13006', '20037', '16106', '7088', '33382', '12441', '10220', '16311', '11095', '5291', '18924', '36074', '1392', '38487', '7889', '22504', '25025', '24574', '19214', '31849', '30512', '16335', '31260', '17308', '38934', '23488', '23521', '11453', '272', '36673', '30298', '19255', '11848', '21043', '9039', '34482', '742', '21596', '1656', '3694', '17174']
            #['25120', '40006', '36682', '138722', '50998', '85739', '29388', '89731', '66349', '90413', '56493', '97128', '173676', '19088', '169487', '147850', '79236', '19800', '193503', '56959', '35580', '150640', '154979', '62594', '180300', '101756', '16236', '176365', '153631', '42820', '194001', '104377', '120072', '170398', '46239', '30650', '125789', '18469', '15716', '171444', '20276', '108897', '145157', '163532', '99980', '184855', '13698', '92811', '34082', '59074', '122908', '93806', '90291', '173304', '159323', '145829', '198376', '113686', '27875', '64721', '148471', '107945', '180662', '185678', '49027', '172178', '143126', '136294', '16730', '20766', '52958', '37269', '154809', '157256', '151190', '20395', '147539', '128754', '22645', '102168', '116240', '121024', '62076', '40258', '67576', '140775', '172546', '125381', '67582', '98177', '152943', '120909', '173465', '187542', '154699', '177750', '59790', '121270', '87212', '183613', '143120', '112353', '26195', '163592', '165970', '158205', '68710', '126879', '77568', '21916', '49956', '142411', '146565', '147194', '170906', '198669', '14409', '58410', '171868', '22145', '138138', '60409', '112405', '43386', '56058', '55846', '154350', '131690', '14449', '27396', '153364', '31000', '165332', '178158', '174622', '82469', '145509', '82295', '69166', '65635', '116593', '32302', '135972', '125587', '52815', '162269', '71991', '171802', '67916', '96736', '17230', '119371', '53355', '110146', '160460', '89308', '35103', '77325', '185026', '12057', '117357', '34078', '135268', '75801', '84837', '72154', '54087', '105303', '66231', '108284', '117341', '198162', '135312', '179182', '83727', '99521', '50935', '176256', '29389', '125658', '118663', '174068', '24969', '148198', '66753', '160324', '105027', '78604', '25882', '194057', '39340', '90239', '40435', '84733', '172900', '35981', '76411', '193852', '10857', '186995', '158750', '175110', '43185', '164878', '143869', '152422', '154780', '48139', '77467', '162686', '111174', '70612', '156766', '64596', '68028', '183513', '69438', '100157', '54872', '111088', '192905', '89145', '42267', '40034', '132222', '163583', '104562', '72341', '137582', '171509', '107321', '85798', '72190', '29154', '102494', '40454', '124365', '133061', '151919', '25398', '129590', '159935', '205482', '7266', '8376', '206482', '209323', '9448', '1591', '206213', '1743', '208376', '206681', '8446', '201820', '211540', '6352', '7163', '2241', '208839', '204230', '203354', '209888', '209974', '207160']#['28934', '12514', '9929', '6575', '27287', '28741', '21639', '20567', '35897', '8234', '20270', '33911', '13967', '26702', '14063', '31492', '22904', '30119', '39373', '24063', '26145', '30674', '17994', '22550', '25310', '19590', '13626', '40835', '13765', '8359', '20132', '24', '6875', '26226']#[]#['13926','1838','21257','21807','22291','24955','25267','2531','25592','27106','2727','29126','34396','35483','36309','3744','5515']
            #for scenario_id in all_scenario_ids:
            #    baseline_minfde = self.baseline_result[scenario_id]['minFDE']
            #    moe_minfde = self.moe_result[scenario_id]['minFDE']
            #    if baseline_minfde > moe_minfde:
            #        self.better_scenario_ids.append(scenario_id)
        else:
            #self.better_scenario_ids = ['13926', '1838', '21257', '21807', '22291', '24955', '25267', '2531', '25592',
            #                            '27106', '2727', '29126', '34396', '35483', '36309', '3744', '5515']
            #self.better_scenario_ids = [
            #        '10392', '20311', '30124', '39355', '10460', '20381', '30136', '3938',
            #        '10888', '20516', '30198', '39524', '10898', '2082', '31262', '39951',
            #        '11232', '21117', '31472', '40055', '11899', '21461', '31571', '40123',
            #        '11984', '21593', '31625', '40350', '12043', '21603', '32494', '40653',
            #        '12755', '22701', '32589', '40780', '13173', '22734', '32727', '40862',
            #        '14499', '23034', '3278', '40915', '1466', '23541', '33164', '4502',
            #        '15583', '24328', '33441', '4976', '15888', '24539', '34531', '50',
            #        '1600', '24600', '34786', '5265', '16145', '25649', '34837', '5920',
            #        '16427', '2564', '35017', '5963', '16899', '25967', '35058', '7012',
            #        '17714', '26001', '36214', '7087', '17715', '2673', '36315', '7480',
            #        '17746', '27107', '36317', '7696', '18347', '27466', '37182', '8107',
            #        '18361', '27482', '38115', '8209', '19287', '27515', '38147', '83',
            #        '19302', '27649', '38164', '8820', '19923', '28289', '3839', '8946',
            #        '20003', '28380', '38419', '9578', '20127', '28432', '38509', '9792',
           #         '20265', '30001', '3875'
           # ]
           self.better_scenario_ids = ['40835','2727','13926','5515']#['28934', '12514', '9929', '6575', '27287', '28741', '21639', '20567', '35897', '8234', '20270', '33911', '13967', '26702', '14063', '31492', '22904', '30119', '39373', '24063', '26145', '30674', '17994', '22550', '25310', '19590', '13626', '40835', '13765', '8359', '20132', '24', '6875', '26226']

        # self._raw_file_names = os.listdir(self.raw_dir)[::self.sample_rate]
        # print("len of raw file names:", len(self._raw_file_names))

        # self._processed_file_names = [os.path.splitext(name)[0] + '.pt' for name in self.raw_file_names]
        self._processed_file_names = [str(scenario_id) + '.pt' for scenario_id in self.better_scenario_ids]
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self.processed_file_names]

        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.margin = margin

        self._turn_direction_type = ['NONE', 'LEFT', 'RIGHT']
        super(ArgoverseV1DatasetVisual, self).__init__(root=root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed_data')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        map_api = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            df = pd.read_csv(raw_path)
            data = dict()
            scenario_id = self.get_scenario_id(raw_path)
            city = self.get_city(df)
            data['city'] = city
            data['scenario_id'] = scenario_id
            data.update(self.get_features(df, map_api, self.margin, city))
            torch.save(data, os.path.join(self.processed_dir, scenario_id + '.pt'))

    @staticmethod
    def get_scenario_id(raw_path: str) -> str:
        return os.path.splitext(os.path.basename(raw_path))[0]

    @staticmethod
    def get_city(df: pd.DataFrame) -> str:
        return df['CITY_NAME'].values[0]

    def get_features(self,
                     df: pd.DataFrame,
                     map_api: ArgoverseMap,
                     margin: float,
                     city: str) -> Dict:
        data = {
            'agent': {},
            'lane': {},
            'centerline': {},
            ('centerline', 'lane'): {},
            ('lane', 'lane'): {}
        }
        # AGENT
        # filter out actors that are unseen during the historical time steps
        timestep_ids = list(np.sort(df['TIMESTAMP'].unique()))
        historical_timestamps = timestep_ids[:self.num_historical_steps]
        historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
        agent_ids = list(historical_df['TRACK_ID'].unique())
        num_agents = len(agent_ids)
        df = df[df['TRACK_ID'].isin(agent_ids)]

        agent_index = agent_ids.index(df[df['OBJECT_TYPE'] == 'AGENT']['TRACK_ID'].values[0])

        # initialization
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        length_mask = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.bool)

        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_length = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)

        for track_id, track_df in df.groupby('TRACK_ID'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = [timestep_ids.index(timestamp) for timestamp in track_df['TIMESTAMP']]

            visible_mask[agent_idx, agent_steps] = True

            length_mask[agent_idx, 0] = False
            length_mask[agent_idx, 1:] = ~(
                        visible_mask[agent_idx, 1:self.num_historical_steps] & visible_mask[agent_idx,
                                                                               :self.num_historical_steps - 1])

            agent_position[agent_idx, agent_steps] = torch.from_numpy(
                np.stack([track_df['X'].values, track_df['Y'].values], axis=-1)).float()
            motion = torch.cat(
                [agent_position.new_zeros(1, 2), agent_position[agent_idx, 1:] - agent_position[agent_idx, :-1]], dim=0)
            length, heading = compute_angles_lengths_2D(motion)
            agent_length[agent_idx] = length[:self.num_historical_steps]
            agent_heading[agent_idx] = heading[:self.num_historical_steps]
            agent_length[agent_idx, length_mask[agent_idx]] = 0
            agent_heading[agent_idx, length_mask[agent_idx]] = 0

        data['agent']['num_nodes'] = num_agents
        data['agent']['agent_index'] = agent_index
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading
        data['agent']['length'] = agent_length

        # MAP
        positions = agent_position[:, :self.num_historical_steps][visible_mask[:, :self.num_historical_steps]].reshape(
            -1, 2)
        left_boundary = min(positions[:, 0])
        right_boundary = max(positions[:, 0])
        down_boundary = min(positions[:, 1])
        up_boundary = max(positions[:, 1])
        lane_ids = map_api.get_lane_ids_in_xy_bbox((left_boundary + right_boundary) / 2,
                                                   (down_boundary + up_boundary) / 2, city,
                                                   max((right_boundary - left_boundary) / 2,
                                                       (up_boundary - down_boundary) / 2) + margin)

        num_lanes = len(lane_ids)
        lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_heading = torch.zeros(num_lanes, dtype=torch.float)
        lane_length = torch.zeros(num_lanes, dtype=torch.float)
        lane_is_intersection = torch.zeros(num_lanes, dtype=torch.uint8)
        lane_turn_direction = torch.zeros(num_lanes, dtype=torch.uint8)
        lane_traffic_control = torch.zeros(num_lanes, dtype=torch.uint8)

        num_centerlines = torch.zeros(num_lanes, dtype=torch.long)
        centerline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
        centerline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
        centerline_length: List[Optional[torch.Tensor]] = [None] * num_lanes

        lane_adjacent_edge_index = []
        lane_predecessor_edge_index = []
        lane_successor_edge_index = []
        for lane_id in lane_ids:
            lane_idx = lane_ids.index(lane_id)

            centerlines = torch.from_numpy(map_api.get_lane_segment_centerline(lane_id, city)[:, :2]).float()
            num_centerlines[lane_idx] = centerlines.size(0) - 1
            centerline_position[lane_idx] = (centerlines[1:] + centerlines[:-1]) / 2
            centerline_vectors = centerlines[1:] - centerlines[:-1]
            centerline_length[lane_idx], centerline_heading[lane_idx] = compute_angles_lengths_2D(centerline_vectors)

            lane_length[lane_idx] = centerline_length[lane_idx].sum()
            center_index = int(num_centerlines[lane_idx] / 2)
            lane_position[lane_idx] = centerlines[center_index]
            lane_heading[lane_idx] = torch.atan2(centerlines[center_index + 1, 1] - centerlines[center_index, 1],
                                                 centerlines[center_index + 1, 0] - centerlines[center_index, 0])

            lane_is_intersection[lane_idx] = torch.tensor(map_api.lane_is_in_intersection(lane_id, city),
                                                          dtype=torch.uint8)
            lane_turn_direction[lane_idx] = torch.tensor(
                self._turn_direction_type.index(map_api.get_lane_turn_direction(lane_id, city)), dtype=torch.uint8)
            lane_traffic_control[lane_idx] = torch.tensor(map_api.lane_has_traffic_control_measure(lane_id, city),
                                                          dtype=torch.uint8)

            lane_adjacent_ids = map_api.get_lane_segment_adjacent_ids(lane_id, city)
            lane_adjacent_idx = get_index_of_A_in_B(lane_adjacent_ids, lane_ids)
            if len(lane_adjacent_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_adjacent_idx, dtype=torch.long),
                                          torch.full((len(lane_adjacent_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_adjacent_edge_index.append(edge_index)
            lane_predecessor_ids = map_api.get_lane_segment_predecessor_ids(lane_id, city)
            lane_predecessor_idx = get_index_of_A_in_B(lane_predecessor_ids, lane_ids)
            if len(lane_predecessor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_predecessor_idx, dtype=torch.long),
                                          torch.full((len(lane_predecessor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_predecessor_edge_index.append(edge_index)
            lane_successor_ids = map_api.get_lane_segment_successor_ids(lane_id, city)
            lane_successor_idx = get_index_of_A_in_B(lane_successor_ids, lane_ids)
            if len(lane_successor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_successor_idx, dtype=torch.long),
                                          torch.full((len(lane_successor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_successor_edge_index.append(edge_index)

        data['lane']['num_nodes'] = num_lanes
        data['lane']['position'] = lane_position
        data['lane']['length'] = lane_length
        data['lane']['heading'] = lane_heading
        data['lane']['is_intersection'] = lane_is_intersection
        data['lane']['turn_direction'] = lane_turn_direction
        data['lane']['traffic_control'] = lane_traffic_control

        data['centerline']['num_nodes'] = num_centerlines.sum().item()
        data['centerline']['position'] = torch.cat(centerline_position, dim=0)
        data['centerline']['heading'] = torch.cat(centerline_heading, dim=0)
        data['centerline']['length'] = torch.cat(centerline_length, dim=0)

        centerline_to_lane_edge_index = torch.stack([torch.arange(num_centerlines.sum(), dtype=torch.long),
                                                     torch.arange(num_lanes, dtype=torch.long).repeat_interleave(
                                                         num_centerlines)], dim=0)
        data['centerline', 'lane']['centerline_to_lane_edge_index'] = centerline_to_lane_edge_index

        if len(lane_adjacent_edge_index) != 0:
            lane_adjacent_edge_index = torch.cat(lane_adjacent_edge_index, dim=1)
        else:
            lane_adjacent_edge_index = torch.tensor([[], []], dtype=torch.long)
        lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
        lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)

        data['lane', 'lane']['adjacent_edge_index'] = lane_adjacent_edge_index
        data['lane', 'lane']['predecessor_edge_index'] = lane_predecessor_edge_index
        data['lane', 'lane']['successor_edge_index'] = lane_successor_edge_index

        return data

    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int) -> HeteroData:
        return HeteroData(torch.load(self.processed_paths[idx]))
