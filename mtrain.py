import random
import treeclasses
import playerclasses
import dominoclasses
import pandas as pd
import numpy as np
import copy

def generate_players(num_players, modes, domino_size, filename):
    """
    Creates player objects based on modes passed to this script
    Returns a list of the player objects in order of creation
    """
    players = []
    for num in range(0, num_players):
        if modes[num] == "Greedy":
            players.append(playerclasses.GreedyPlayer(num))
        elif modes[num] == "Random":
            players.append(playerclasses.RandomPlayer(num))
        elif modes[num] == "Probability":
            players.append(playerclasses.ProbabilityPlayer(num, domino_size))
        elif modes[num] == "Neural":
            players.append(playerclasses.NeuralPlayer(num, domino_size, filename, num_players))
    return players

def create_one_hot(domino_size, dominos):
    """
    Creates a list of 0s and 1s where each 0 or 1 corresponds to having a specific domino
    Used for collecting data on what dominos are in a hand, in a train, etc. 
    This is later decoded when training the neural net
    """
    domino_count = int(domino_size + 1 + ((domino_size + 1) * domino_size) / 2.0)
    one_hot = np.zeros(domino_count)
    for domino in dominos:
        try:
            location = int(max(domino) * (domino_size + 1 - (max(domino) + 1) / 2.0) + min(domino))
        except TypeError:
            print(dominos)
            print(type(domino))
            print(domino)
            print(domino[0])
            print(domino[1])
            raise TypeError
        one_hot[location] = 1
    return one_hot.tolist()

def strip_potential_plays(potential_plays, domino_size):
    """
    Filters through potential_plays (A matrix of plays) and creates a list of all dominos potentially to be played
    Returns a one_hot of the dominos that can potentially be played
    """
    dominos = []
    for potentials in potential_plays:
        for play in potentials:
            dominos.append(play[-1])
    return create_one_hot(domino_size, dominos)


def mexicantrain(num_players=2, domino_size=12, data_collection=False, debug=True, 
                 modes=["Greedy", "Random"], data_index=0, file_name="PlayData/data2_12_100"):
    """
    A function that runs a single game of mexican train from start to finish. A full guide of the
    rules can be found in the README.MD file. 

    Default parameters support a head-to-head match, but passing in additional players and changing the
    parameters accordingly is supported as well. 

    When not using a Neural player, file_name does not matter

    Returns the scores, the index of the winning player, and the data collected if in data_collection mode
    """
    #Check player number
    if not num_players in range(2, 9):
        raise ValueError("Number of players must be between 2 and 8, inclusive")
    
    #Set up DataFrame for data collection mode
    column_names = ["round_number", "turn_number", "player_number", "play", 
                "t_num", "hand", "unknown", "potential_plays", "points"]
    for num in range(0, num_players + 1):
        column_names.append("train_" + str(num))
        column_names.append("marker_" + str(num))

    data = pd.DataFrame(dtype=object, columns=column_names)

    #Hand size rule
    hand_sizes = [16, 16, 15, 14, 12, 10, 9]
    hand_size = hand_sizes[num_players - 2]

    scores = []
    for ind in range(0, num_players):
        scores.append(0)

    #Generate the players for the game
    players = generate_players(num_players, modes, domino_size, file_name)

    #Start game
    for round_number in range(domino_size, -1, -1):
        if debug: print("Round start: " + str(round_number))

        #Create Shuffled Deck
        if debug: print("Creating Deck")
        deck = dominoclasses.Deck(domino_size)
        trains = []
        for playernum in range(0, num_players + 1):
            trains.append(dominoclasses.Train())
        trains[num_players].set_marker(True)

        #Generate Random Hands for each player
        if debug: print("Creating Hands")
        hands = []
        for playernum in range(0, num_players):
            dominos = deck.draw(hand_size)
            hands.append(dominoclasses.Hand(dominos))
        
        #Check who has the current target double, if no one has it, everyone draws one domino
        if debug: print("Checking for player with needed domino:")
        start_player = -1
        draw_again = True
        while draw_again:
            for playernum in range(0, num_players):
                if hands[playernum].check_double(round_number) == 1:
                    draw_again = False
                    start_player = playernum
                    hands[start_player].remove_domino((round_number, round_number))
                    break
            if draw_again:
                for playernum in range(0, num_players):
                    domino = deck.draw(1)
                    hands[playernum].add_dominos(domino)
            else:
                break
        if debug: print("Domino found, round beginning")
        #Start round

        round_over = False
        current_player = start_player
        double_up = (False, -1)
        doom_counter = 0
        turn_number = 1
        while not round_over:
            if debug: print("Player " + str(current_player) + " is now playing")
            end_turn = False
            active_player = players[current_player]
            #If another player has played a double, the double must be covered
            if double_up[0]:
                #Get potential play
                if debug: print("Forced to play on double on train " + str(double_up[1]))
                if debug: print("Last domino on train " + str(double_up[1]) + " is " + str(trains[double_up[1]].get_last()))
                if debug: print("Current player's hand is: " + str(hands[current_player].dominos))
                play, play_data = active_player.play_forced_double(hands[current_player].dominos, trains[double_up[1]])
                #If no play exists, try again
                if len(play) == 0:
                    if debug: print("No play available, drawing again")
                    hands[current_player].add_dominos(deck.draw(1))
                    if debug: print("Current player's hand is: " + str(hands[current_player].dominos))
                    play, play_data = active_player.play_forced_double(hands[current_player].dominos, trains[double_up[1]])
                    if len(play) == 0:
                        if debug: print("No play available, continuing to next player")
                        trains[current_player].set_marker(True)
                        end_turn = True
                        doom_counter += 1
                #Play play if it exists, and return double state to normal
                if not end_turn:
                    #Collect data on the play if necessary
                    if data_collection:
                        data.loc[data_index, "round_number"] = round_number
                        data.loc[data_index, "turn_number"] = turn_number / num_players
                        data.loc[data_index, "player_number"] = current_player
                        data.loc[data_index, "play"] = create_one_hot(domino_size, play)
                        data.loc[data_index, "t_num"] = double_up[1]
                        data.loc[data_index, "hand"] = create_one_hot(domino_size, hands[current_player].dominos)
                        unknown = copy.deepcopy(deck.dominos)
                        for x in range(0, len(hands)):
                            if x == current_player:
                                continue
                            else:
                                unknown += hands[x].dominos
                        data.loc[data_index, "unknown"] = create_one_hot(domino_size, unknown)
                        data.loc[data_index, "potential_plays"] = create_one_hot(domino_size, play_data)
                        for x in range(0, len(trains)):
                            data.loc[data_index, "train_" + str(x)] = create_one_hot(domino_size, trains[x].train_list)
                            if trains[x].marker_up:
                                data.loc[data_index, "marker_" + str(x)] = 1
                            else:
                                data.loc[data_index, "marker_" + str(x)] = 0
                        data_index += 1
                    
                    #Play the play onto the target train and remove from hand
                    trains[double_up[1]].add_domino(play[0])
                    for pl in play:
                        hands[current_player].remove_domino(pl)
                    
                    #End the double_up mode and reset the doom_counter
                    double_up = (False, -1)
                    doom_counter = 0
            #If no double is needed to be covered and the player hasn't played their train yet,
            #they play their train now
            elif trains[current_player].empty():
                #Get the train play from the player class
                play = active_player.play_train(hands[current_player].dominos, round_number)

                #Verify an actual train is being played, if not draw and try again
                if len(play) == 0:
                    hands[current_player].add_dominos(deck.draw(1))
                    play = active_player.play_train(hands[current_player].dominos, round_number)
                    if len(play) == 0:
                        trains[current_player].set_marker(True)
                        doom_counter += 1
                        end_turn = True
                
                #Play the train the player came up with
                if not end_turn:
                    doom_counter = 0
                    trains[current_player].add_train(play)
                    for pl in play:
                        hands[current_player].remove_domino(pl)
                    
                    #Check if the final domino played is a double, if so deal with that case
                    if(play[-1][0] == play[-1][1]):
                        hands[current_player].add_dominos(deck.draw(1))
                        play = active_player.play_forced_double(hands[current_player].dominos, trains[double_up[1]])
                        if len(play) == 0:
                            trains[current_player].set_marker(True)
                            end_turn = True
                            doom_counter += 1
                    #End the turn afterwards   
                    end_turn = True
            #In most cases during the game, this last else will occur, which causes players to play normally
            else:
                #Get a standard play from the current player
                t_num, play, play_data = active_player.play_normally(hands[current_player].dominos, trains, round_number, turn_number)
                
                #If the play doesn't exist, try again and process this
                if len(play) == 0:
                    hands[current_player].add_dominos(deck.draw(1))
                    t_num, play, play_data = active_player.play_normally(hands[current_player].dominos, trains, round_number, turn_number)
                    if len(play) == 0:
                        trains[current_player].set_marker(True)
                        doom_counter += 1
                        end_turn = True
                
                #If the play does exist, deal with the three possibilities the play could look like
                if not end_turn:
                    doom_counter = 0
                    #Single play that isn't a double, the easiest case to deal with
                    if len(play) == 1 and not (play[0][0] == play[0][1]):
                        #Collect data as necessary
                        if data_collection:
                            data.loc[data_index, "round_number"] = round_number
                            data.loc[data_index, "turn_number"] = turn_number / num_players
                            data.loc[data_index, "player_number"] = current_player
                            data.loc[data_index, "play"] = create_one_hot(domino_size, play)
                            data.loc[data_index, "t_num"] = t_num
                            data.loc[data_index, "hand"] = create_one_hot(domino_size, hands[current_player].dominos)
                            unknown = copy.deepcopy(deck.dominos)
                            for x in range(0, len(hands)):
                                if x == current_player:
                                    continue
                                else:
                                    unknown += hands[x].dominos
                            data.loc[data_index, "unknown"] = create_one_hot(domino_size, unknown)
                            data.loc[data_index, "potential_plays"] = strip_potential_plays(play_data, domino_size)
                            for x in range(0, len(trains)):
                                data.loc[data_index, "train_" + str(x)] = create_one_hot(domino_size, trains[x].train_list)
                                if trains[x].marker_up:
                                    data.loc[data_index, "marker_" + str(x)] = 1
                                else:
                                    data.loc[data_index, "marker_" + str(x)] = 0
                            data_index += 1
                        
                        #Play domino on train and remove it from the player's hand
                        trains[t_num].add_domino(play[0])
                        for pl in play:
                            hands[current_player].remove_domino(pl)
                    #Single play that is a double, signally cover double mode
                    elif len(play) == 1 and (play[0][0] == play[0][1]):
                        #Collect data as necessary
                        if data_collection:
                            data.loc[data_index, "round_number"] = round_number
                            data.loc[data_index, "turn_number"] = turn_number / num_players
                            data.loc[data_index, "player_number"] = current_player
                            data.loc[data_index, "play"] = create_one_hot(domino_size, play)
                            data.loc[data_index, "t_num"] = t_num
                            data.loc[data_index, "hand"] = create_one_hot(domino_size, hands[current_player].dominos)
                            unknown = copy.deepcopy(deck.dominos)
                            for x in range(0, len(hands)):
                                if x == current_player:
                                    continue
                                else:
                                    unknown += hands[x].dominos
                            data.loc[data_index, "unknown"] = create_one_hot(domino_size, unknown)
                            data.loc[data_index, "potential_plays"] = strip_potential_plays(play_data, domino_size)
                            for x in range(0, len(trains)):
                                data.loc[data_index, "train_" + str(x)] = create_one_hot(domino_size, trains[x].train_list)
                                if trains[x].marker_up:
                                    data.loc[data_index, "marker_" + str(x)] = 1
                                else:
                                    data.loc[data_index, "marker_" + str(x)] = 0
                            data_index += 1
                        
                        #Play domino and remove from hand of player
                        trains[t_num].add_domino(play[0])
                        for pl in play:
                            hands[current_player].remove_domino(pl)
                        
                        #Draw to attempt to cover the domino
                        hands[current_player].add_dominos(deck.draw(1))
                        play_2, play_data_2 = active_player.play_forced_double(hands[current_player].dominos, trains[t_num])
                        
                        #If no play is available, start double_up mode
                        if len(play_2) == 0:
                            double_up = (True, t_num)
                            trains[current_player].set_marker(True)
                            end_turn = True
                        
                        #If a play is available, play it and collect data
                        if not end_turn:
                            #Collect Data as neccesary
                            if data_collection:
                                data.loc[data_index, "round_number"] = round_number
                                data.loc[data_index, "turn_number"] = turn_number / num_players
                                data.loc[data_index, "player_number"] = current_player
                                data.loc[data_index, "play"] = create_one_hot(domino_size, play_2)
                                data.loc[data_index, "t_num"] = t_num
                                data.loc[data_index, "hand"] = create_one_hot(domino_size, hands[current_player].dominos)
                                unknown = copy.deepcopy(deck.dominos)
                                for x in range(0, len(hands)):
                                    if x == current_player:
                                        continue
                                    else:
                                        unknown += hands[x].dominos
                                data.loc[data_index, "unknown"] = create_one_hot(domino_size, unknown)
                                data.loc[data_index, "potential_plays"] = create_one_hot(domino_size, play_data_2)
                                for x in range(0, len(trains)):
                                    data.loc[data_index, "train_" + str(x)] = create_one_hot(domino_size, trains[x].train_list)
                                    if trains[x].marker_up:
                                        data.loc[data_index, "marker_" + str(x)] = 1
                                    else:
                                        data.loc[data_index, "marker_" + str(x)] = 0
                                data_index += 1

                            #Play domino drawn to train and remove from hand
                            trains[t_num].add_domino(play_2[0])
                            for pl in play_2:
                                hands[current_player].remove_domino(pl)
                    
                    #Case when a double and a followup are played
                    else:
                        #Collect Data as needed
                        if data_collection:
                            data.loc[data_index, "round_number"] = round_number
                            data.loc[data_index, "turn_number"] = turn_number / num_players
                            data.loc[data_index, "player_number"] = current_player
                            data.loc[data_index, "play"] = create_one_hot(domino_size, play)
                            data.loc[data_index, "t_num"] = t_num
                            data.loc[data_index, "hand"] = create_one_hot(domino_size, hands[current_player].dominos)
                            unknown = copy.deepcopy(deck.dominos)
                            for x in range(0, len(hands)):
                                if x == current_player:
                                    continue
                                else:
                                    unknown += hands[x].dominos
                            data.loc[data_index, "unknown"] = create_one_hot(domino_size, unknown)
                            data.loc[data_index, "potential_plays"] = strip_potential_plays(play_data, domino_size)
                            for x in range(0, len(trains)):
                                data.loc[data_index, "train_" + str(x)] = create_one_hot(domino_size, trains[x].train_list)
                                if trains[x].marker_up:
                                    data.loc[data_index, "marker_" + str(x)] = 1
                                else:
                                    data.loc[data_index, "marker_" + str(x)] = 0
                            data_index += 1
                        #Play both the dominos to the train and remove from the player's hand
                        trains[t_num].add_domino(play[0])
                        trains[t_num].add_domino(play[1])
                        for pl in play:
                            hands[current_player].remove_domino(pl)
                        end_turn = True
            
            #Check if the current player is out of dominos, or if doom counter has been exceeded
            if hands[current_player].winning():
                if debug: print("Player " + str(current_player) + " ran out of dominos!")
                round_over = True
            elif doom_counter > num_players * 5:
                if debug: 
                    print("Doom counter exceeded maximum!")
                    for hand in hands:
                        print(hand.dominos)
                round_over = True
            
            #Continue on to the next player and increase the turn_number
            current_player += 1
            if current_player > num_players - 1:
                current_player = 0
            turn_number += 1
            if debug and turn_number % 5 == 0: print("Turn %s now", turn_number)
        
        #Once the round is over, calculate the scores of each player for that round
        round_scores = []
        for playernum in range(0, num_players):
            scores[playernum] += hands[playernum].score
            round_scores.append(hands[playernum].score)
        if debug: 
            print("Round " + str(round_number) + "over")
            print("Round Scores: ")
            print(round_scores)
            print("Total Scores: ")
            print(scores)
        #Collect data as needed
        if data_collection:
            for ind in data.index:
                data.loc[ind, "points"] = hands[int(data.loc[ind, "player_number"])].score
            
            
    #Find overall winner
    lowest_score = 1000000
    index = -1
    for playernum in range(0, num_players):
        if scores[playernum] < lowest_score:
            lowest_score = scores[playernum]
            index = playernum
    
    #Return results of game
    if debug: print("Game over, player" + str(index) + " won")
    return scores, index, data