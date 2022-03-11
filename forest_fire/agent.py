from mesa import Agent


class TreeCell(Agent):
    """
    A tree cell.

    Attributes:
        x, y: Grid coordinates
        condition: Can be "Fine", "On Fire", or "Burned Out"
        unique_id: (x,y) tuple.

    unique_id isn't strictly necessary here, but it's good
    practice to give one to each agent anyway.
    """

    def __init__(self, pos, model, wind):
        """
        Create a new tree.
        Args:
            pos: The tree's coordinates on the grid.
            model: standard model reference for agent.
        """
        super().__init__(pos, model)
        self.pos = pos
        self.condition = "Fine"
        self.wind = wind

    def step(self):
        if self.condition == "On Fire":
            # Se a árvore estiver pegando fogo e o vento estiver acima de 40 m/s existe uma chance de até 25% dele apagar o fogo
            if self.wind >= 40 and self.random.random() < ((self.wind - 40) / 40):
                self.condition = "Fire Put Out"
            # Caso contrário as árvore vizinhas irão pegar fogo dependendo da velocidade do vento
            else:
                for neighbor in self.model.grid.neighbor_iter(self.pos):
                    if neighbor.condition == "Fine" and self.random.random() < (self.wind / 50):
                        neighbor.condition = "On Fire"
                self.condition = "Burned Out"

