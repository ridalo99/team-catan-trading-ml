import os
import glob

import torch
from torch import Tensor

from ai_games.learn_settlers.ai.team5.nn_bot.converter import Converter
from ai_games.learn_settlers.ai.team5.nn_bot.data_structure import TrainingData
from ai_games.learn_settlers.game.objects import GameState
from ai_games.learn_settlers.game.objects.actions import Action
from ai_games.learn_settlers.game.objects.actions.action_type import ActionType
from ai_games.learn_settlers.game.objects.actions.build_action import BuildCornerAction, BuildEdgeAction
from ai_games.learn_settlers.game.objects.building import BuildingType
from ai_games.learn_settlers.game.player.player import Player
from ai_games.learn_settlers.utils.log_reader import LogReader
from ai_games.learn_settlers.game.objects.actions.trade_action import TradeAction
import numpy as np
base_dir = os.path.dirname(__file__)
gamelog_path = os.environ.get("CATAN_GAMELOG_DIR", "gamelog_train")
data_set_path = os.path.join(base_dir, "..", "datasets") # , "road_dataset.pt")

data_file: dict[str, str] = {
    "road": "road_dataset.pt",
    "settlement": "settlement_dataset.pt",
    "city": "city_dataset.pt",
    "build": "build_dataset.pt",
    "trade":"trade_dataset.pt"
}

class SettlersDataset:
    def __init__(self, data: list[tuple[Action, GameState]]):
        self.data = data
        self.converter = Converter()

    def load_data(self, max_data: int = 0) -> list[tuple[Action, GameState]]:
        files = glob.glob(gamelog_path)
        count_data = 0
        res_vector: list[tuple[Action, GameState]] = []
        for file in files:
            if max_data > 0 and count_data >= max_data:
                break
            reader = LogReader(file)
            tuples = reader.generate_action_gs_tuples()
            res_vector += tuples
            count_data += 1
            if count_data % 100 == 0:
                print(f"Loaded {count_data} files, current data size: {len(res_vector)}")
        return res_vector
    
    def filter_data(self, player_no: int, 
                    data: list[tuple[Action, GameState]] , 
                    action_types: list[ActionType] = None) -> list[tuple[Action, GameState]]:
        return [
            entry for entry in data
            if (entry[0] is not None) and (entry[0].player_no == player_no) and
            (action_types is None or entry[0].action_type in action_types)
        ]
    
    def create_road_dataset(self, player_no: int, features: list[str],
                       action_types: list[ActionType] = None, max_data = 0) -> TrainingData:
        """
        Creates a dataset of training data for the neural network.
        :param player_no: The player number for which the dataset is created.
        :param features: The features to be used for the dataset.
        :param action_types: The action types to filter the dataset.
        :return: A list of tuples containing the feature tensor and the action tensor.
        """
        # TODO: Works for Roads, add other actions

        data = self.load_data(max_data)
        filtered_data = self.filter_data(player_no, data, action_types)

        data_set: list[TrainingData] = []
        for action, game_state in filtered_data:
            if isinstance(action, BuildEdgeAction):
                x = self.converter.create_feature_vector(game_state, features)
                y = self.converter.get_edge_label(action)
                data_set.append((torch.tensor(x, dtype=torch.float32), y))
        
        path = os.path.join(data_set_path, data_file["road"])
        torch.save(data_set, path)
        return data_set 
        # return PyTorchDataset(data_set)

    def create_settlement_dataset(self, player_no: int, features: list[str],
                                  action_types: list[ActionType] = None, max_data = 0) -> TrainingData:
        """
        Creates a dataset of training data for the neural network.
        :param player_no: The player number for which the dataset is created.
        :param features: The features to be used for the dataset.
        :param action_types: The action types to filter the dataset.
        :return: A list of tuples containing the feature tensor and the action tensor.
        """
        data = self.load_data(max_data)
        print("Data loaded, filtering...")
        filtered_data = self.filter_data(player_no, data, action_types)
        print(f"Filtered data contains {len(filtered_data)} entries.")

        data_set: list[TrainingData] = []
        for action, game_state in filtered_data:
            if isinstance(action, BuildCornerAction) and (action.building_type in [BuildingType.SETTLEMENT, BuildingType.HARBOR_SETTLEMENT]):
                x = self.converter.create_feature_vector(game_state, features)
                y = self.converter.get_corner_label(action)
                data_set.append((torch.tensor(x, dtype=torch.float32), y))

                if len(data_set) % 100 == 0:
                    print(f"Processed {len(data_set)} entries for settlements.")
        
        path = os.path.join(data_set_path, data_file["settlement"])
        torch.save(data_set, path)
        return data_set
    
    def create_city_dataset(self, player_no: int, features: list[str],
                                  action_types: list[ActionType] = None, max_data = 0) -> TrainingData:
        """
        Creates a dataset of training data for the neural network.
        :param player_no: The player number for which the dataset is created.
        :param features: The features to be used for the dataset.
        :param action_types: The action types to filter the dataset.
        :return: A list of tuples containing the feature tensor and the action tensor.
        """
        data = self.load_data(max_data)
        filtered_data = self.filter_data(player_no, data, action_types)

        data_set: list[TrainingData] = []
        for action, game_state in filtered_data:
            if isinstance(action, BuildCornerAction) and (action.building_type in [BuildingType.CITY, BuildingType.HARBOR_CITY]):
                x = self.converter.create_feature_vector(game_state, features)
                y = self.converter.get_corner_label(action)
                data_set.append((torch.tensor(x, dtype=torch.float32), y))

                if len(data_set) % 100 == 0:
                    print(f"Processed {len(data_set)} entries for cities.")
        
        path = os.path.join(data_set_path, data_file["city"])
        torch.save(data_set, path)
        return data_set
    
    def create_build_dataset(self, player_no: int, features: list[str],
                            action_types: list[ActionType] = None, max_data = 0) -> TrainingData:
        
        data = self.load_data(max_data)
        filtered_data = self.filter_data(player_no, data, action_types)

        data_set: list[TrainingData] = []
        for action, game_state in filtered_data:
            x = self.converter.create_feature_vector(game_state, features)
            y = self.converter.get_building_label(action)
            data_set.append((torch.tensor(x, dtype=torch.float32), y))

            if len(data_set) % 100 == 0:
                print(f"Processed {len(data_set)} entries for build options.")
        
        path = os.path.join(data_set_path, data_file["build"])
        torch.save(data_set, path)
        return data_set
    
    
    def create_trade_dataset(self, player_no: int, features: list[str],
                        action_types: list[ActionType] = None, max_data=0) -> TrainingData:
        data = self.load_data(max_data)
        filtered_data = self.filter_data(player_no, data,action_types)
        filtered_data = [(a, g) for a, g in filtered_data if a.target_player >= 0]


        data_set : list[TrainingData] = []
        for action, game_state in filtered_data:
                x = self.converter.get_trade_features(game_state, player_no)
                # Label: Angebotene Ressourcen (5), Geforderte Ressourcen (5), Partner (int)
                offer = [o / 4 for o in list(action.out_resources[:5])]
                request = [r / 4 for r in list(action.in_resources[:5])]
                partner = action.target_player
                if sum(offer) == 0 and sum(request) == 0:
                    continue  # überspringe leere Trades
                y = offer + request + [partner]
                data_set.append((torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)))
        # Speicherort
        trade_data_set_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "trade_dataset.pt")
        os.makedirs(os.path.dirname(trade_data_set_path), exist_ok=True)
        torch.save(data_set, trade_data_set_path)
        return data_set


def main():
    dataset = SettlersDataset([])
    data_set = dataset.create_trade_dataset(
            player_no=0,  # Passe ggf. die Player-Nummer an!
            features=["victory_points", "resources", "possible_corners", "possible_edges", "dice_values_tiles", "resources_tiles", "longest_road"],
            action_types=[ActionType.TRADE],
            max_data=1500
        )
    print(f"Trade-Dataset created with {len(data_set)} entries.")
    
    y = np.stack([y.numpy() for _, y in data_set])
    print("Partner-Verteilung:", np.bincount(y[:, -1].astype(int)))
    print("Angebot-Summe pro Ressource:", y[:, :5].sum(axis=0))
    print("Nachfrage-Summe pro Ressource:", y[:, 5:10].sum(axis=0))
    x = np.stack([x.numpy() for x, _ in data_set])
    print("Feature-Mittelwerte:", x.mean(axis=0))
    print("Feature-Standardabweichung:", x.std(axis=0))
    if data_set:
        print(f"Feature shape: {data_set[0][0].shape}")
    #data_set = dataset.create_build_dataset(
     #   player_no=4,  # TODO Fix player number
    #    features=["victory_points", "resources", "possible_corners", "possible_edges", "dice_values_tiles", "resources_tiles", "longest_road"],
    #    action_types=[ActionType.BUILD],
     #   max_data=150  # Limit to 10 entries for testing
   #)


    # data_set = dataset.create_settlement_dataset(
    #     player_no=0,  # TODO Fix player number
    #     features=["victory_points", "resources", "possible_corners", "dice_values_tiles", "resources_tiles"],
    #     action_types=[ActionType.BUILD],
    #     max_data=25000  # Limit to 10 entries for testing
    # )

  #  data_set = dataset.create_road_dataset(
   #      player_no=0, # TODO Fix player number
    #     features=["victory_points", "resources", "possible_edges", "dice_values_tiles", "resources_tiles", "longest_road"],
     #    action_types=[ActionType.BUILD],
      #   max_data=10  # Limit to 10 entries for testing
    # )

    print(f"Dataset created with {len(data_set)} entries.")
    # for x, y in data_set:
    #     print(f"Features: {x}, Action: {y}")
    print(f"Data saved to {data_set_path}")

    # print the feature shape
    if data_set:
        print(f"Feature shape: {data_set[0][0].shape}")

if __name__ == "__main__":
    main()
