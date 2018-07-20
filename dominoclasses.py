import random

class Deck:

    def __init__(self, domino_value):
        self.dominos = []
        for x in range(0, domino_value + 1):
            for y in range(x, domino_value + 1):
                self.dominos.append((x, y))
        
        random.shuffle(self.dominos)

    def draw(self, number):
        drawnpile = []
        if len(self.dominos) == 0:
            return [(-1, -1)]
        elif len(self.dominos) < number:
            number = len(self.dominos)
        
        drawnpile = self.dominos[0:number]
        self.dominos = self.dominos[number:]
        return drawnpile

class Train:

    def __init__(self):
        self.train_list = []
        self.marker_up = False
    
    def add_train(self, dominos):
        for dom in dominos:
            self.add_domino(dom)

    def add_domino(self, domino):
        self.train_list.append(domino)
    
    def set_marker(self, marker):
        self.marker_up = marker
    
    def empty(self):
        if len(self.train_list) > 0:
            return False
        else:
            return True
    
    def get_last(self):
        if len(self.train_list) == 0:
            return (-1, -1)
        return self.train_list[len(self.train_list) - 1]
    
class Hand:

    def __init__(self, dominos):
        self.dominos = dominos
        self.score = 0
        for domino in self.dominos:
            total = domino[0] + domino[1]
            self.score += total
    
    def update_score(self):
        self.score = 0
        for domino in self.dominos:
            self.score += domino[0] + domino[1]
    
    def remove_domino(self, domino):
        new_dominos = []
        for dom in self.dominos:
            if not (domino[0] == dom[0] and domino[1] == dom[1]) \
                    and not (domino[1] == dom[0] and domino[0] == dom[1]):
                new_dominos.append(dom)
        
        self.dominos = new_dominos
        self.update_score()
    
    def add_dominos(self, dominos):
        self.dominos = self.dominos + list(dominos)
        self.remove_domino((-1, -1))
        self.update_score()
    
    def check_double(self, value):
        for dom in self.dominos:
            if (value == dom[0] and value == dom[1]):
                return 1
        return 0
    
    def winning(self):
        if len(self.dominos) == 0:
            return True
        else:
            return False