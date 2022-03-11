from doctest import run_docstring_examples
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Grid
from mesa.time import RandomActivation
from mesa.batchrunner import BatchRunner
from datetime import datetime

from .agent import TreeCell


class ForestFire(Model):
    """
    Simple Forest Fire model.
    """

    def __init__(self, width=100, height=100, density=0.65, wind=0):
        """
        Create a new forest fire model.

        Args:
            width, height: The size of the grid to model
            density: What fraction of grid cells have a tree in them.
        """
        # Set up model objects
        self.schedule = RandomActivation(self)
        self.grid = Grid(width, height, torus=False)
        self.count_step = 1
        self.density = density
        self.wind = wind

        self.datacollector = DataCollector(
            model_reporters={
                "Fine": lambda m: self.count_type(m, "Fine"),
                "On Fire": lambda m: self.count_type(m, "On Fire"),
                "Burned Out": lambda m: self.count_type(m, "Burned Out"),
                "Fire Put Out": lambda m: self.count_type(m, "Fire Put Out"),
                "Total steps of the fire forest": lambda m: self.count_step
            },
            agent_reporters={
                "Condition of determined tree": lambda x: x.condition
            }
        )

        # Place a tree in each cell with Prob = density
        for (contents, x, y) in self.grid.coord_iter():
            if self.random.random() < density:
                # Create a tree
                new_tree = TreeCell((x, y), self, wind)
                # Set all trees in the first column on fire.
                if x == 0:
                    new_tree.condition = "On Fire"
                self.grid._place_agent((x, y), new_tree)
                self.schedule.add(new_tree)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step.
        """
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

        self.count_step = self.count_step + 1

        # Halt if no more fire
        if self.count_type(self, "On Fire") == 0:
            self.running = False
            
            run_model_data = self.datacollector.get_model_vars_dataframe()
            run_model_data1 = self.datacollector.get_agent_vars_dataframe()

            now = str(datetime.now()).replace(':', '-')
            file_name_sufix = ("_wind="+str(self.wind)+"_density="+str(self.density)+"_"+now)
            run_model_data.to_csv("model_data"+file_name_sufix+".csv")
            run_model_data1.to_csv("agent_data"+file_name_sufix+".csv")

    @staticmethod
    def count_type(model, tree_condition):
        """
        Helper method to count trees in a given condition in a given model.
        """
        count = 0
        for tree in model.schedule.agents:
            if tree.condition == tree_condition:
                count += 1
        return count
