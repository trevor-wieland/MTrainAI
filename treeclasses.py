class Node:
    def __init__(self, value):
        self.value = value
        self.connections = []
    
    def add_connection(self, value):
        self.connections.append(Node(value))
    
    def get_connection_value(self, value):
        for connection in self.connections:
            if connection.value == value:
                return connection
        return None
    
    def print_tree(self):
        h = self.height()
        for i in range(1, h+1):
            self.print_level(i)
            print()
    
    def print_level(self, level):
        if level == 1:
            print(self.value, end=" ")
        else:
            for connection in self.connections:
                connection.print_level(level-1)

    def height(self):
        if len(self.connections) == 0:
            return 1
        else:
            heights = []
            for connection in self.connections:
                heights.append(connection.height())
            return max(heights) + 1

class WeightedNode:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight
        self.connections = []
    
    def add_connection(self, name="", weight=0, node=""):
        if node == "":
            self.connections.append(WeightedNode(name, weight))
        else:
            self.connections.append(node)
    
    def get_connection_name(self, name):
        for connection in self.connections:
            if connection.name == name:
                return connection
        return None
    
    def print_tree(self):
        h = self.height()
        for i in range(1, h+1):
            self.print_level(i)
            print()
    
    def print_level(self, level):
        if level == 1:
            print(self.name, end=" ")
        else:
            for connection in self.connections:
                connection.print_level(level-1)

    def height(self):
        if len(self.connections) == 0:
            return 1
        else:
            heights = []
            for connection in self.connections:
                heights.append(connection.height())
            return max(heights) + 1