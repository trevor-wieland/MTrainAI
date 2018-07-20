# MTrainAI

### Overview:
##### What is Mexican Train:
Mexican Train is a domino game my family tends to play at reunions and other family events. An unfortunate side effect of this is that I too must play Mexican Train. In order to make this event considerably less fun for everyone involved, I decided to find the optimal strategy to play Mexican Train using AI. The way the game is played in my family is like so:

* The game takes place over 13 rounds, from round 12 to round 0
* Each player draws a certain given amount of shuffled 12-12 dominos
* Whoever has the double associated with the round number goes first, placing the special domino in the middle of the table.
* They then use the rest of their dominos to try to build as long a train as possible, and play it off the side of the domino in the middle. This is their "train". 
* Players play in clockwise order their longest possible train, which must start with a value equal to the double in the center.
* After all players have played their long trains, players may now play a domino at a time, on their own train, or on any train with a marker. 
* Trains get markers when someone is unable to play anywhere, and after drawing one domino from the pile, is still unable to play. 
* If a double is played, the player must play again, "covering" the double. If the player is unable to do so, they must draw and try again. If they are still unable, they must place a marker up on their train, and the next player must attempt to cover the double. This repeats until someone can cover it, or the attempts to cover the double have circled around the players twice, in which case the round ends.
* At any time after the first player trains are played, a player may start the nominal "Mexican Train", by playing a single domino as a new train coming from the center of the original double domino. Only one "Mexican Train" is allowed per round.
* The round ends when no one can play any more, or when a player plays their last domino.
* Each player gains points equal to the umber of dots on their dominos in their hand, and the next round starts.
* After all 13 rounds, the player with the least amount of points wins. 
* More precise rule information can be found in the Rules Section

As the goal at the end of the game is to have the lowest sum of points, and each round the deck is reshuffled and redealt, and thus independent of any other round, the goal of an optimal player is to minimize their points in a given round.
To do this, I created three different strategic players, and use their simulated plays to train a Neural Network. The hand-made strategies are the following:
1. Random: This player looks at their potential plays each turn and chooses one at random.
2. Greedy: This player looks at their potential plays each turn and chooses the highest score domino to play
3. Probability: This player combines the strategy of the Greedy player, and also attempts to make other players unable to play, while making the player itself able to, as an effort to force the other players to draw.

These players were created in order to simulate games to train the Neural Network. 
##### Rules:
A full guide to the rules of mexican train can be found [here](http://www.tactic.net/site/rules/UK/02588.pdf).
<br>
Some rules that were changed in developing this version were the following:
1. If no one can play for twice as many turns as the number of players, the round will end.
2. If no one can play on a double for twice as many turns as the number of players, the round will end.
3. Playing a full train of all dominos in the players first turn will end the round as a victory. 
4. The hands are dealt first, then whoever has the double for this round plays it, and this player builds their train first, then turn order proceeds in numerical order of player creation.
5. A player may only play on other trains once their train is started. If another player starts their train for them, they no longer get to play multiple dominos at the beginning.
6. Perhaps the largest change to the rules is the ability to play as many dominos as possible in a "train" when the player is first starting their train. This speeds the game up considerably when done in real life, and accounts for most of the pre-game strategy. 
### Requirements:
* Python 3.5
* Installation of packages in requirements.txt using pip
### Files:
- `dominoclasses.py` - Holds Game Classes on Dominos
- `treeclasses.py` - Holds Game Classes on Trees
- `playerclasses.py` - Holds Game Classes on Players
- `mtrain.py` - Holds Game Method
- `mtrainsimulator.py` - Holds Simulator Method
- `mtraintester.py` - Holds Debugging Methods
- `requirements.txt` - Holds needed package information
- `README.md` - The file you're reading now
- `PlayData` - Folder Holding Training Data

##### PlayData Folder:
Holds sheets generated by previous plays. These plays then are used to train the Neural Net Player, using the goal of minimizing the points gained at the end of a completed round. This is done by solving a regression problem, by trying to estimate the number of points at the end of a round a player will have by playing a given move. The features currently used are:

- Turn Number
- Round Number
- Play
- Train Number Played on
- Hand
- Unknown Dominos
- Each Train
- Each Train's Marker
- All Potential Plays
- Points at End of Round

Separate sheets are created for different player numbers and different sizes of domino. These sheets are added to when the program is run in "Data" mode, and afterwards, the "Train" mode should be run to retrain the algorithm on the newly generated data. 


