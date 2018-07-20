import mtrain
import numpy as np
import pandas as pd
import random

def simulate_games(num_players=4, domino_size=12, num_games=250, collect_data=True, 
                    debug=False, players=["Random", "Greedy", "Probability", "Neural"], 
                    file_name="PlayData/data4_12_250"):
    """
    Runs the mexican train game repeatedly with different combinations of players to
    generate data to be used in testing and training the neural net. 

    If collect_data is on, the play data is retrieved and stored into a .xlsx file for later use
    The format for the file name for this is as follows:
    PlayData/data + num_players + _ + domino_size + _ + num_games + .xlsx
    This spreadsheet is to be used when training the neural net.

    This script has no required parameters, and will run the game with the default params if
    unchanged.

    If collect_data is on, the players are selected randomly each game from:
    ["Random", "Greedy", "Probability"]
    If collect_data is off, the players are selected in order from the parameter players.
    When collect_data is off: len(players) must equal num_players

    Returns a tuple of lists: (score_averages, win_percentage) corresponding to the players
    """

    #Sets column names for building dataframe later on
    column_names = ["round_number", "turn_number", "player_number", "play", 
                    "t_num", "hand", "unknown", "potential_plays", "points"]
    
    #Depending on mode of use, sets players and checks validity of player values
    modes = []
    if collect_data:
        modes = ["Random", "Greedy", "Probability"]
    else:
        if not len(players) == num_players:
            raise RuntimeError("len(players) must equal num_players when collect_data is off")
        modes = players

    #Simulates num_games of games
    scores = np.ndarray((num_players, num_games))
    wins = np.ndarray((num_players, num_games))
    full_data = pd.DataFrame(columns=column_names)
    current_index = 0
    for game_num in range(0, num_games):

        #Randomize players if in collect_data mode
        game_modes = []
        if collect_data:
            for select in range(0, num_players):
                game_modes.append(random.choice(modes))
        else:
            game_modes = modes
        
        #Run game with parameters
        results = mtrain.mexicantrain(num_players, domino_size, debug=debug, 
                                        modes=game_modes, 
                                        data_collection=collect_data,
                                        data_index=current_index, file_name=file_name)
        #If collecting data, data is stored into the dataframe
        if collect_data:
            current_index = results[2].index[-1] + 1
            full_data = pd.concat([full_data, results[2]])
        
        #Scores and wins are recorded into their respective arrays
        for player_num in range(0, num_players):
            scores[player_num, game_num] = results[0][player_num]
            if results[1] == player_num:
                wins[player_num, game_num] = 1
            else:
                wins[player_num, game_num] = 0

    #Calculates performance of the players
    score_averages = np.ndarray((num_players))
    win_percentage = np.ndarray((num_players))
    for player_num in range(0, num_players):
        score_averages[player_num] = np.mean(scores[player_num, :])
        win_percentage[player_num] = np.mean(wins[player_num, :])

    #If collecting data, prints data to a .xlsx file
    if collect_data:
        filename = "PlayData/data" + str(num_players) + "_" + str(domino_size) + "_" + str(num_games) + ".xlsx"
        writer = pd.ExcelWriter(filename)
        full_data.to_excel(writer, "Sheet1")
        writer.save()

    #Prints results and returns them as well
    if debug: print(score_averages)
    if debug: print(win_percentage)
    return score_averages, win_percentage