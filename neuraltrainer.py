import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import ast
from sklearn.externals import joblib

def train_neural_net(file_name, num_players, domino_size, debug=False, train_size=.6, validation_size=.2, test_size=.2, num_layers=3):
    """
    A function to train a neural net on how to play Mexican Train.
    This neural net is a regressor to determine the points gained at the end
    of a match, given the details about a specific play. This can then be used to
    determine which play is best when playing Mexican Train, by choosing the play
    which is predicted to yield the lowest amount of points at the end of the round.

    The regressor object is stored in a .pkl file after training and testing, and separate
    regressors will need to be trained when using different amounts of players or domino sizes.
    This function should be run after the mtrainsimulator.py file is run, as it requires the data
    generated into an .xlsx file from this module.

    This function takes in the following required parameters:
    file_name: The path to the .xlsx file without the .xlsx ending
    num_players: The number of players the data was generated from
    domino_size: Which size domino the game was played using

    Returns the final score that the trained neural net acquires
    Outputs the Regressor object to a .pkl file
    """
    #Set up numerical and categorical features
    numerical_features = ["round_number", "turn_number", "points", "t_num"]
    vector_features = ["play", "hand", "unknown", "potential_plays"]
    for num in range(0, num_players + 1):
        vector_features.append("train_" + str(num))
        numerical_features.append("marker_" + str(num))
    
    #Read in Excel Data from spreadsheet
    if debug: print("Reading Excel Data...")
    full_data = pd.read_excel(file_name + ".xlsx", "Sheet1")
    full_data.drop('player_number', axis=1, inplace=True)

    #Replace the strings read in with lists
    if debug: print("Replacing Strings with Lists...")
    for col in vector_features:
        try:
            full_data[col] = [ast.literal_eval(full_data.loc[x, col]) for x in full_data.index]
        except ValueError:
            print(col)
            raise ValueError
    
    #Randomize the data
    if debug: print("Randomizing Data Order...")
    full_data = full_data.reindex(np.random.permutation(full_data.index))

    #Turn categorical data into hundreds of features
    if debug: print("Building Usable Dataframe...")
    true_data = pd.DataFrame()
    true_data["round_number"] = full_data["round_number"]
    true_data["turn_number"] = full_data["turn_number"]
    true_data["points"] = full_data["points"]
    true_data["t_num"] = full_data["t_num"]

    domino_count = int(domino_size + 1 + ((domino_size + 1) * domino_size) / 2.0)
    for current in range(0, domino_count):
            true_data["play_" + str(current)] = [x[current] for x in full_data["play"]]
            true_data["hand_" + str(current)] = [x[current] for x in full_data["hand"]]
            true_data["unknown_" + str(current)] = [x[current] for x in full_data["unknown"]]
            true_data["potential_plays_" + str(current)] = [x[current] for x in full_data["potential_plays"]]
    
    for num in range(0, num_players + 1):
        true_data["marker_" + str(num)] = full_data["marker_" + str(num)]
        for current in range(0, domino_count):
            true_data["train_" + str(num) + "_" + str(current)] = [x[current] for x in full_data["train_" + str(num)]]

    #Train Neural Network
    if debug: print("Splitting Data into Training, Validation, and Test sets")
    train_size = int(train_size * true_data.shape[0])
    validation_size = int(validation_size * true_data.shape[0]) + train_size
    test_size = int(test_size * true_data.shape[0]) + validation_size

    train_data = true_data.iloc[0:train_size, :]
    validation_data = true_data.iloc[train_size+1:validation_size, :]
    test_data = true_data.iloc[validation_size+1:test_size, :]

    if debug: print("Training Neural Network...")
    layers = tuple([700 for x in range(0, num_layers)])
    regressor = MLPRegressor(max_iter=500, activation="relu", hidden_layer_sizes=layers, learning_rate_init=.0015)
    regressor.fit(train_data.drop(["points"], axis=1), train_data["points"])

    #Test the network
    if debug: print("Testing Neural Network")
    final_score = regressor.score(test_data.drop(["points"], axis=1), test_data["points"])
    if debug: print("Final Test Score of " + str(final_score))
    
    #Store and return the neural network
    joblib.dump(regressor, file_name + ".pkl")

    return final_score