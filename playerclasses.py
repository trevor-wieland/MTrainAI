from abc import ABC, abstractmethod
import treeclasses
import random
import copy
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
import pandas
import mtrain

class Player(ABC):

    def __init__(self, player_num):
        self.player_num = player_num
        super().__init__()

    @abstractmethod
    def play_train(self, dominos, start_value):
        pass

    @abstractmethod
    def play_forced_double(self, dominos, start_value):
        pass

    @abstractmethod
    def play_normally(self, dominos, trains, round_number, turn_number):
        pass

    def can_play_on_single(self, dominos, targets):
        """
        Checks to find all possible single domino plays that can play on the available targets
        """
        potential_plays = []
        for target in targets:
            for domino in dominos:
                if domino[0] == domino[1]:
                    continue
                else:
                    if domino[0] == target[1]:
                        potential_plays.append((target, (domino[0], domino[1])))
                    if domino[1] == target[1]:
                        potential_plays.append((target, (domino[1], domino[0])))

        return potential_plays

    def can_play_on_double_good(self, dominos, targets):
        """
        Checks to find all possible good double pair dominos that can play on the available targets
        """
        potential_plays = []
        for target in targets:
            for domino1 in dominos:
                if domino1[0] == domino1[1] and domino1[0] == target[1]:
                    for domino2 in dominos:
                        if domino1[0] == domino2[0] and domino1[1] == domino2[1]:
                            continue
                        elif domino1[0] == domino2[0]:
                            potential_plays.append((target, domino1, (domino2[0], domino2[1])))
                        elif domino1[0] == domino2[1]:
                            potential_plays.append((target, domino1, (domino2[1], domino2[0])))
                        else:
                            continue
                else:
                    continue
            
        return potential_plays

    def can_play_on_double_bad(self, dominos, targets):
        """
        Checks to find all possible bad double pair dominos that can play on the available targets
        """
        potential_plays = []
        for target in targets:
            for domino in dominos:
                if domino[0] == domino[1] and domino[0] == target[1]:
                    potential_plays.append((target, domino))
                else:
                    continue   
        return potential_plays
    
    def find_longest_path(self, node, touched):
        children = []
        for con in node.connections:
            if not con.weight in touched:
                children.append(con)
        touched.append(node.weight)
        if len(children) == 0:
            return [node.weight]
        weight = 0
        path = []
        for child in children:
            pt = self.find_longest_path(child, touched)
            if sum(pt) > weight:
                weight = sum(pt)
                path = pt
        return [node.weight] + path

    def build_network(self, dominos, value, touched):
        next_tiles = []
        for dom in dominos:
            if dom[0] == value:
                next_tiles.append((dom[0], dom[1]))
            elif dom[1] == value:
                next_tiles.append((dom[1], dom[0]))
        touched.append(value)
        nodes = []
        for tile in next_tiles:
            if not tile[1] in touched:
                nodes.append(self.build_network(dominos, tile[1], touched))
            elif tile[1] == value:
                count = 0
                for t in touched:
                    if t == tile[1]:
                        count += 1
                if count == 2:
                    continue
                else:
                    nodes.append(self.build_network(dominos, tile[1], touched))
        nd = treeclasses.WeightedNode(str(value), value)
        for node in nodes:
            nd.add_connection(node=node)
        return nd 

class GreedyPlayer(Player):

    def play_train(self, dominos, start_value):
        """
        Returns a list of the longest possible series of dominos to play
        """
        nd = self.build_network(dominos, start_value, [])
        path = self.find_longest_path(nd, [])
        train = []
        for x in range(0, len(path) - 1):
            train.append((path[x], path[x+1]))
        return train

    def play_forced_double(self, dominos, train):
        """
        Returns a list of the longest possible series of dominos to play
        """
        needed = train.get_last()
        plays = self.can_play_on_single(dominos, [needed])
        if len(plays) == 0:
            return [], []
        else:
            scores = []
            play_data = []
            for play in plays:
                scores.append(play[1][0] + play[1][1])
                play_data.append(play[1])
            max_score = -1
            index = -1
            for play_num in range(0, len(plays)):
                if scores[play_num] > max_score:
                    max_score = scores[play_num]
                    index = play_num
            return [plays[index][1]], play_data

    def play_normally(self, dominos, trains, round_number, turn_number):
        """
        Returns a list of the longest possible series of dominos to play
        """
        targets = []

        for x in range(0, len(trains)):
            train = trains[x]
            if train.marker_up:
                if train.get_last() == (-1, -1):
                    targets.append((round_number, round_number))
                else:
                    targets.append(train.get_last())

        targets.append(trains[self.player_num].get_last())

        scores = [[],[],[]]
        potential_plays = [[],[],[]]

        potential_plays[0] = self.can_play_on_single(dominos, targets)
        potential_plays[1] = self.can_play_on_double_good(dominos, targets)
        potential_plays[2] = self.can_play_on_double_bad(dominos, targets)

        for play in potential_plays[0]:
            scores[0].append(play[1][0] + play[1][1])
        
        for play in potential_plays[1]:
            scores[1].append(play[1][0] + play[1][1] + play[2][0] + play[2][1])
        
        for play in potential_plays[2]:
            scores[2].append(play[1][0] + play[1][1])
        
        max_score = -1
        x_index = -1
        y_index = -1

        for x in range(0, 3):
            for y in range(0, len(scores[x])):
                if scores[x][y] > max_score:
                    max_score = scores[x][y]
                    x_index = x
                    y_index = y

        if x_index == -1 or y_index == -1:
            return -1, [], []
        play = potential_plays[x_index][y_index]
        target = play[0]
        t_num = -1
        plays = []
        for x in range(1, len(play)):
            plays.append(play[x])
        
        for x in range(0, len(trains)):
            train = trains[x]
            if train.marker_up or x == self.player_num:
                if train.get_last() == (-1, -1) and target == (round_number, round_number):
                    t_num = x
                elif target == train.get_last():
                    t_num = x

        return t_num, plays, potential_plays

class RandomPlayer(Player):
    def play_train(self, dominos, start_value):
        """
        Returns a list of the longest possible series of dominos to play
        """
        nd = self.build_network(dominos, start_value, [])
        path = self.find_longest_path(nd, [])
        train = []
        for x in range(0, len(path) - 1):
            train.append((path[x], path[x+1]))
        return train

    def play_forced_double(self, dominos, train):
        """
        Returns a list of the longest possible series of dominos to play
        """
        needed = train.get_last()
        plays = self.can_play_on_single(dominos, [needed])
        if len(plays) == 0:
            return [], []
        else:
            play_data = []
            for play in plays:
                play_data.append(play[1])
            index = random.randrange(0, len(plays))
            return [plays[index][1]], play_data

    def play_normally(self, dominos, trains, round_number, turn_number):
        """
        Returns a list of the longest possible series of dominos to play
        """
        targets = []

        for x in range(0, len(trains)):
            train = trains[x]
            if train.marker_up:
                if train.get_last() == (-1, -1):
                    targets.append((round_number, round_number))
                else:
                    targets.append(train.get_last())

        targets.append(trains[self.player_num].get_last())

        potential_plays = [[],[],[]]

        potential_plays[0] = self.can_play_on_single(dominos, targets)
        potential_plays[1] = self.can_play_on_double_good(dominos, targets)
        potential_plays[2] = self.can_play_on_double_bad(dominos, targets)

        x_index = -1
        y_index = -1
        for x in [1, 0, 2]:
            if len(potential_plays[x]) == 0:
                continue
            else:
                x_index = x
                y_index = random.randrange(0, len(potential_plays[x]))

        if x_index == -1 or y_index == -1:
            return -1, [], []
        
        play = potential_plays[x_index][y_index]
        target = play[0]
        t_num = -1
        plays = []
        for x in range(1, len(play)):
            plays.append(play[x])
        
        for x in range(0, len(trains)):
            train = trains[x]
            if train.marker_up or x == self.player_num:
                if train.get_last() == (-1, -1) and target == (round_number, round_number):
                    t_num = x
                elif target == train.get_last():
                    t_num = x

        return t_num, plays, potential_plays

class ProbabilityPlayer(Player):

    def __init__(self, player_num, domino_size):
        super().__init__(player_num)
        self.domino_size = domino_size

    def get_unknown_dominos(self, trains, dominos, round_number):
        known_dominos = copy.deepcopy(dominos)
        known_dominos += [(round_number, round_number)]
        for train in trains:
            known_dominos += train.train_list
        unknown_dominos = []
        for x in range(0, self.domino_size + 1):
            for y in range(x, self.domino_size + 1):
                if not (x,y) in known_dominos and not (y,x) in known_dominos:
                    unknown_dominos.append((x,y))
        return unknown_dominos

    def playability_probabilities(self, unknown_dominos):
        play_probs = []
        for value in range(0, self.domino_size+1):
            left_over_doms = []
            for domino in unknown_dominos:
                if domino[0] == value or domino[1] == value:
                    left_over_doms.append(domino)
            
            prob = float(len(left_over_doms)) / float(len(unknown_dominos))
            play_probs.append(prob)
        return play_probs

    def play_train(self, dominos, start_value):
        """
        Returns a list of the longest possible series of dominos to play
        """
        nd = self.build_network(dominos, start_value, [])
        path = self.find_longest_path(nd, [])
        train = []
        for x in range(0, len(path) - 1):
            train.append((path[x], path[x+1]))
        return train

    def play_forced_double(self, dominos, train):
        """
        Returns a list of the longest possible series of dominos to play
        """
        needed = train.get_last()
        plays = self.can_play_on_single(dominos, [needed])
        if len(plays) == 0:
            return [], []
        else:
            scores = []
            play_data = []
            for play in plays:
                scores.append(play[1][0] + play[1][1])
                play_data.append(play[1])
            max_score = -1
            index = -1
            for play_num in range(0, len(plays)):
                if scores[play_num] > max_score:
                    max_score = scores[play_num]
                    index = play_num

            return [plays[index][1]], play_data

    def play_normally(self, dominos, trains, round_number, turn_number):
        """
        Returns a list of the longest possible series of dominos to play
        """
        targets = []

        for x in range(0, len(trains)):
            train = trains[x]
            if train.marker_up:
                if train.get_last() == (-1, -1):
                    targets.append((round_number, round_number))
                else:
                    targets.append(train.get_last())

        targets.append(trains[self.player_num].get_last())

        scores = [[],[],[]]
        probs = [[],[],[]]
        play_off_score = [[],[],[]]
        potential_plays = [[],[],[]]

        potential_plays[0] = self.can_play_on_single(dominos, targets)
        potential_plays[1] = self.can_play_on_double_good(dominos, targets)
        potential_plays[2] = self.can_play_on_double_bad(dominos, targets)

        unknown_dominos = self.get_unknown_dominos(trains, dominos, round_number)
        playability_probabilities = self.playability_probabilities(unknown_dominos)

        for play in potential_plays[0]:
            scores[0].append(play[1][0] + play[1][1])
            try:
                probs[0].append(playability_probabilities[play[1][1]])
            except TypeError:
                print("Play: " + str(play))
                print("Targets: " + str(targets))
                print("Dominos: " + str(dominos))
                print("Dominos size: " + str(len(dominos)))
                print("Playernum: " + str(self.player_num))
                raise TypeError("Issue")
            
            doms = [dom for dom in dominos if not dom == play[1]]
            targs = [tar for tar in targets if not tar == play[0]]
            targs += [play[1]]
            if len(self.can_play_on_single(doms, targs)) > 0 \
                    or len(self.can_play_on_double_good(doms, targs)) > 0:
                play_off_score[0].append(1)
            else:
                play_off_score[0].append(0)
        
        for play in potential_plays[1]:
            scores[1].append(play[1][0] + play[1][1] + play[2][0] + play[2][1])
            probs[1].append(playability_probabilities[play[2][1]])

            doms = [dom for dom in dominos if not dom == play[1] and not dom == play[2]]
            targs = [tar for tar in targets if not tar == play[0]]
            targs += [play[2]]
            if len(self.can_play_on_single(doms, targs)) > 0 \
                    or len(self.can_play_on_double_good(doms, targs)) > 0:
                play_off_score[1].append(1)
            else:
                play_off_score[1].append(0)
        
        for play in potential_plays[2]:
            scores[2].append(play[1][0] + play[1][1])
            probs[2].append(playability_probabilities[play[1][1]])

            doms = [dom for dom in dominos if not dom == play[1]]
            targs = [tar for tar in targets if not tar == play[0]]
            targs += [play[1]]
            if len(self.can_play_on_single(doms, targs)) > 0 \
                    or len(self.can_play_on_double_good(doms, targs)) > 0:
                play_off_score[2].append(1)
            else:
                play_off_score[2].append(0)
        
        combined_score = [[0 for x in scores[0]],[0 for x in scores[1]],[0 for x in scores[2]]]
        for x in range(0, 3):
            for y in range(0, len(scores[x])):
                combined_score[x][y] = .5 * scores[x][y] / (self.domino_size * 2)
                combined_score[x][y] += .5 * (1 - probs[x][y]) * play_off_score[x][y]

        max_net_score = -1
        x_index = -1
        y_index = -1

        for x in range(0, 3):
            for y in range(0, len(combined_score[x])):
                if combined_score[x][y] > max_net_score:
                    max_net_score = combined_score[x][y]
                    x_index = x
                    y_index = y

        if x_index == -1 or y_index == -1:
            return -1, [], []
        play = potential_plays[x_index][y_index]
        target = play[0]
        t_num = -1
        plays = []
        for x in range(1, len(play)):
            plays.append(play[x])
        
        for x in range(0, len(trains)):
            train = trains[x]
            if train.marker_up or x == self.player_num:
                if train.get_last() == (-1, -1) and target == (round_number, round_number):
                    t_num = x
                elif target == train.get_last():
                    t_num = x
        return t_num, plays, potential_plays

class NeuralPlayer(Player):
    def __init__(self, player_num, domino_size, filename, num_players):
        super().__init__(player_num)
        self.num_players = num_players
        self.domino_size = domino_size
        self.network = joblib.load(filename + ".pkl")
        self.features = ["round_number", "turn_number", "t_num"]
        self.features += ["play", "hand", "unknown", "potential_plays"]
        for num in range(0, num_players + 1):
            self.features.append("train_" + str(num))
            self.features.append("marker_" + str(num))
        
    
    def predict_scores_of_plays(self, play_data):
        if play_data.shape[0] < 1:
            return []
        scores = self.network.predict(play_data)
        return scores
    
    def get_unknown_dominos(self, trains, dominos, round_number):
        known_dominos = copy.deepcopy(dominos)
        known_dominos += [(round_number, round_number)]
        for train in trains:
            known_dominos += train.train_list
        unknown_dominos = []
        for x in range(0, self.domino_size + 1):
            for y in range(x, self.domino_size + 1):
                if not (x,y) in known_dominos and not (y,x) in known_dominos:
                    unknown_dominos.append((x,y))
        return unknown_dominos
    
    def build_dataframe(self, round_number, turn_number, trains, dominos, potential_plays):
        first_frame = pandas.DataFrame(columns=self.features)
        for x in range(0, len(potential_plays)):
            first_frame.loc[x, "round_number"] = round_number
            first_frame.loc[x, "turn_number"] = turn_number

            plays = [potential_plays[y][1] for y in range(0, len(potential_plays))]
            first_frame.loc[x, "potential_plays"] = mtrain.create_one_hot(self.domino_size, plays)

            unknown = self.get_unknown_dominos(trains, dominos, round_number)
            first_frame.loc[x, "unknown"] = mtrain.create_one_hot(self.domino_size, unknown)

            first_frame.loc[x, "hand"] = mtrain.create_one_hot(self.domino_size, dominos)
            
            for train_num in range(0, len(trains)):
                first_frame.loc[x, "train_" + str(train_num)] = mtrain.create_one_hot(self.domino_size, trains[train_num].train_list)
                if trains[train_num].marker_up:
                    first_frame.loc[x, "marker_" + str(train_num)] = 1
                else:
                    first_frame.loc[x, "marker_" + str(train_num)] = 0
                
                if trains[train_num].marker_up or train_num == self.player_num:
                    target = trains[train_num].get_last()
                    if target == (-1, -1) and potential_plays[x][0] == (round_number, round_number):
                        first_frame.loc[x, "t_num"] = train_num
                    elif target == potential_plays[x][0]:
                        first_frame.loc[x, "t_num"] = train_num
                
            first_frame.loc[x, "play"] = mtrain.create_one_hot(self.domino_size,[potential_plays[x][1]])
        
        true_data = pandas.DataFrame()
        true_data["round_number"] = first_frame["round_number"]
        true_data["turn_number"] = first_frame["turn_number"]
        true_data["t_num"] = first_frame["t_num"]

        domino_count = int(self.domino_size + 1 + ((self.domino_size + 1) * self.domino_size) / 2.0)
        for current in range(0, domino_count):
                true_data["play_" + str(current)] = [x[current] for x in first_frame["play"]]
                true_data["hand_" + str(current)] = [x[current] for x in first_frame["hand"]]
                true_data["unknown_" + str(current)] = [x[current] for x in first_frame["unknown"]]
                true_data["potential_plays_" + str(current)] = [x[current] for x in first_frame["potential_plays"]]
        
        for num in range(0, self.num_players + 1):
            true_data["marker_" + str(num)] = first_frame["marker_" + str(num)]
            for current in range(0, domino_count):
                true_data["train_" + str(num) + "_" + str(current)] = [x[current] for x in first_frame["train_" + str(num)]]
        
        return true_data

    def play_train(self, dominos, start_value):
        """
        Returns a list of the longest possible series of dominos to play
        """
        nd = self.build_network(dominos, start_value, [])
        path = self.find_longest_path(nd, [])
        train = []
        for x in range(0, len(path) - 1):
            train.append((path[x], path[x+1]))
        return train

    def play_forced_double(self, dominos, train):
        """
        Returns a list of the longest possible series of dominos to play
        """
        needed = train.get_last()
        plays = self.can_play_on_single(dominos, [needed])
        if len(plays) == 0:
            return [], []
        else:
            scores = []
            play_data = []
            for play in plays:
                scores.append(play[1][0] + play[1][1])
                play_data.append(play[1])
            max_score = -1
            index = -1
            for play_num in range(0, len(plays)):
                if scores[play_num] > max_score:
                    max_score = scores[play_num]
                    index = play_num
            return [plays[index][1]], play_data

    def play_normally(self, dominos, trains, round_number, turn_number):
        """
        Returns a list of the longest possible series of dominos to play
        """
        targets = []

        for x in range(0, len(trains)):
            train = trains[x]
            if train.marker_up or x == self.player_num:
                if train.get_last() == (-1, -1):
                    targets.append((round_number, round_number))
                else:
                    targets.append(train.get_last())


        potential_plays = [[],[],[]]

        potential_plays[0] = self.can_play_on_single(dominos, targets)
        potential_plays[1] = self.can_play_on_double_good(dominos, targets)
        potential_plays[2] = self.can_play_on_double_bad(dominos, targets)

        all_plays = copy.deepcopy(potential_plays[0])
        all_plays += copy.deepcopy(potential_plays[1])
        all_plays += copy.deepcopy(potential_plays[2])
        
        formatted_plays = []
        for play in all_plays:
            formatted_plays.append((play[0], play[-1]))
        
        try:
            data = self.build_dataframe(round_number, turn_number, trains, dominos, formatted_plays)
            scores = self.predict_scores_of_plays(data)
        except ValueError:
            print("Round: " + str(round_number))
            print("Turn: " + str(round_number))
            for tag in targets:
                print("Targets: " + str(tag))
            for dom in dominos:
                print("Dominos: " + str(dom))
            for pl in formatted_plays:
                print("Plays: " + str(pl))
            print(data.head())
            raise ValueError
            

        min_score = 1000
        index = -1
        for score_num in range(0, len(scores)):
            if scores[score_num] < min_score:
                index = score_num
                min_score = scores[score_num]
        
        if index == -1:
            return -1, [], []
        
        play = all_plays[index]
        target = play[0]
        t_num = -1
        plays = []
        for x in range(1, len(play)):
            plays.append(play[x])
        
        for x in range(0, len(trains)):
            train = trains[x]
            if train.marker_up or x == self.player_num:
                if train.get_last() == (-1, -1) and target == (round_number, round_number):
                    t_num = x
                elif target == train.get_last():
                    t_num = x

        return t_num, plays, potential_plays