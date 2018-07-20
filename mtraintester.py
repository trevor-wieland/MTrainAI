import mtrain
import neuraltrainer
import mtrainsimulator

if __name__ == "__main__":
    """
    Main testing function, call the other functions from here to easily test each part of the
    program
    """
    #results = mtrainsimulator.simulate_games(num_games=100, debug=False)
    #results = mtrainsimulator.simulate_games(debug=False, collect_data=False, num_games=100, file_name="PlayData/data4_12_250")
    #results = neuraltrainer.train_neural_net(num_players=4, domino_size=12, file_name="PlayData/data4_12_250", debug=True)
    results = mtrain.mexicantrain(num_players=4, domino_size=12, data_collection=False, 
                                    debug=False, modes=["Random", "Greedy", "Probability", "Neural"],
                                    file_name="PlayData/data4_12_250")
    print(results)