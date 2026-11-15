
from ai_games.learn_settlers.game.logic.game_logic import GameLogic
from ai_games.learn_settlers.game.objects.actions.action import Action
from ai_games.learn_settlers.game.objects.actions.build_action import BuildCornerAction, BuildEdgeAction
from ai_games.learn_settlers.game.objects.board_objects import Edge, Pos
from ai_games.learn_settlers.game.objects.building import BuildingType
from ai_games.learn_settlers.game.objects.game_state import GameState

from ai_games.learn_settlers.ai.team5.nn_bot.utility import (encode_corner, decode_corner_label, encode_edge, decode_edge_label, 
                                                             encode_possible_edge_ids, encode_possible_corner_ids , 
                                                             get_dice_value, get_tile_resources, get_building_type)
from ai_games.learn_settlers.game.objects.terrain import Terrain
from ai_games.learn_settlers.game.objects.resource import Resource


FEATURES: dict[str, callable] = {
    "victory_points": lambda converter, game_state: converter.get_victory_points(game_state),  # 4 entries
    "resources": lambda converter, game_state: converter.get_resources(game_state),  # 20 entries
    "possible_edges": lambda converter, game_state: converter.get_possible_edges(game_state), # 72 entries
    "possible_corners": lambda converter, game_state: converter.get_possible_corners(game_state), # 54 entries
    "dice_values_tiles": lambda converter, game_state: converter.get_dice_values_tiles(game_state),  # 19 entries
    "resources_tiles": lambda converter, game_state: converter.get_resources_tiles(game_state),  # 95 (19 * 5) entries
    "longest_road": lambda converter, game_state: converter.get_longest_road(game_state),  # 4 entries
    }

DEFAULT_FEATURES: list[str] = [FEATURES.keys()]

class Converter:
    def __init__(self, player_number = 0, model = None):
        self.player_number = player_number
        self.model = model
        self.game_logic = GameLogic()

    def get_edge_label(self, action: BuildEdgeAction) -> int:
        return encode_edge(action.edge.tiles)
    
    def get_edge_position(self, label: int) -> tuple[Pos]:
        return decode_edge_label(label)
    
    def get_corner_label(self, action: BuildCornerAction) -> int:
        return encode_corner(action.corner.tiles)
    
    def get_corner_position(self, label:int) -> list[Pos]:
        return decode_corner_label(label)
    
    def get_building_label(self, action: Action) -> int:
        return get_building_type(action)
    
    def create_feature_vector(self, game_state: GameState, features_selected: list[str] = DEFAULT_FEATURES) -> list[float]:
        """
        Converts the game state into a feature vector suitable for the neural network.
        This method should be implemented based on the specific features of the game state.
        :param game_state: The current game state.
        :param features_selected: The features to be included in the feature vector.
        :return: A list of floats representing the feature vector.

        Default features includes all features defined in FEATURES.
        """
        features = []
        for feature in features_selected:
            if feature in FEATURES:
                feature_value = FEATURES[feature](self, game_state)
                features.extend(feature_value)
        return features 


    def get_victory_points(self, game_state: GameState) -> list[float]:
        return [player.victory_points for player in game_state.players]

    def get_resources(self, game_state: GameState) -> list[float]:
        return [res for player in game_state.players for res in player.resources[:5]]
    
    def get_possible_edges(self, game_state: GameState) -> list[int]:
        edges: list[Edge] = game_state.board.edges
        result_edges = set()
        for edge in edges:
            if edge.building.player_no == self.player_number:
                adjacent_corners = game_state.board.get_neighbors_of_edge(edge.id)
                for corner in adjacent_corners:
                    adjacent_edges = game_state.board.get_neighbors_of_corner(corner.id)
                    for adjacent_edge in adjacent_edges:
                        if adjacent_edge.id != edge.id and adjacent_edge.building.building_id == BuildingType.EMPTY:
                            result_edges.add(edge.id)
        #print("\nPossible edges:")
        #for edge in result_edges:
        #    print(f"{edge}", end='; ')
        #print(f"{sum(result_edges)} / {len(edges)} edges found.")

        return encode_possible_edge_ids(result_edges, len(edges)) # create vector of size 72 with 1s for possible edges and 0s for others
        # return [edge.tiles for edge in edges if edge.building is None]

    def get_possible_corners(self, game_state: GameState) -> list[int]:
        corners = game_state.board.corners
        result_corners = set()

        possible_actions = self.game_logic.get_possible_build_actions(game_state)
        for action in possible_actions:
            if isinstance(action, BuildCornerAction) and action.building_type == BuildingType.SETTLEMENT:
                result_corners.add(action.corner.id)
        
        return encode_possible_corner_ids(result_corners, len(corners))

        for corner in corners:
            if corner.building.player_no == self.player_number:
                game_state.board.get_neighbors_of_corner(corner.id)

        return [corner.id for corner in corners if corner.building.building_id == BuildingType.EMPTY]

    def get_dice_values_tiles(self, game_state: GameState) -> list[float]:
        tiles = game_state.board.tiles
        result_tiles = []
        for tile_row in tiles:
            for tile in tile_row:
                if tile is not None and tile.terrain is not None and tile.terrain != Terrain.Water and  tile.dice is not None:
                    result_tiles.append(get_dice_value(tile))
        return result_tiles

    def get_resources_tiles(self, game_state: GameState) -> list[int]:
        tiles = game_state.board.tiles
        result_tiles = []
        for tile_row in tiles:
            for tile in tile_row:
                if tile is not None and tile.terrain is not None and tile.terrain != Terrain.Water:
                    result_tiles.extend(get_tile_resources(tile))
        return result_tiles
    
    def get_longest_road(self, game_state: GameState) -> int:
        longest_road = [0] * 4
        for index, player in enumerate(game_state.players):
            if player.longest_road:
                longest_road[index] = 1
        #print(f"Longest road: {longest_road}")
        return longest_road
    def get_trade_features(self, game_state:GameState, my_no:int):
        player=game_state.players[my_no]
        features = []
        features += [r / 19 for r in player.resources[:5]]
        features.append(player.victory_points/10)
        features.append(game_state.turn/3)
        features.append(len(player.roads)/15)
        features.append(player.settlement_count/5)
        features.append(player.city_count/4)

    
        if hasattr(player, "harbors"):
            harbors = list(getattr(player, "harbors", [False]*6))
            harbors = harbors + [0]*(6-len(harbors)) if len(harbors) < 6 else harbors[:6]
            features += [int(h) for h in harbors]
        else:
            features += [0]*6
        # Entwicklungskarten (5, normiert, immer 5 Werte)
        if hasattr(player, "development_cards"):
            dev_cards = list(player.development_cards)
            dev_cards = dev_cards + [0]*(5-len(dev_cards)) if len(dev_cards) < 5 else dev_cards[:5]
            features += [c / 5 for c in dev_cards]
        else:
            features += [0]*5
        for i, p in enumerate(game_state.players):
            if i != my_no:
                features.append(p.victory_points/10)
                features.append(len(p.roads)/15)
                features.append(p.settlement_count/5)
                features.append(p.city_count/4)    
        return features
