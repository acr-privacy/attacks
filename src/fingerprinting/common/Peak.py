class Peak:
    def __init__(self, x: int, y: int):
        self.x = x  # Frame number
        self.y = y  # Frequency bin

    def __eq__(self, other):
        if not isinstance(other, Peak):
            return False
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def __str__(self):
        return f"({self.x},{self.y})"

    def __hash__(self):
        return hash((self.x, self.y))