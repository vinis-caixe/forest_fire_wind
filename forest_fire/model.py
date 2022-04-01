from doctest import run_docstring_examples
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Grid
from mesa.time import RandomActivation
from mesa.batchrunner import BatchRunner
from datetime import datetime
import numpy as np
from scipy.ndimage import measurements

from .agent import TreeCell

class ForestFire(Model):
    """
    Simple Forest Fire model.
    """

    def __init__(self, width=100, height=100, density=0.65, wind=0.0):
        """
        Create a new forest fire model.

        Args:
            width, height: The size of the grid to model
            density: What fraction of grid cells have a tree in them.
        """
        # Set up model objects
        self.schedule = RandomActivation(self)
        self.grid = Grid(width, height, torus=False)
        self.width = width
        self.height = height
        self.count_step = 1

        # Variáveis de controle
        self.density = density
        self.wind = wind

        self.datacollector_cluster_fine = DataCollector(
            {
                "Number of clusters (Fine)": lambda m: self.count_clusters(m, self.width, self.height, "Fine"),
                
            }
        )
        self.datacollector_cluster_fireputout = DataCollector(
            {
                "Number of clusters (Fire Put Out)": lambda m: self.count_clusters(m, self.width, self.height, "Fire Put Out"),
            }
        )
        self.datacollector = DataCollector(
            {
                "Fine": lambda m: self.count_type(m, "Fine"),
                "On Fire": lambda m: self.count_type(m, "On Fire"),
                "Burned Out": lambda m: self.count_type(m, "Burned Out"),
                "Fire Put Out": lambda m: self.count_type(m, "Fire Put Out"),
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
        self.datacollector_cluster_fine.collect(self)
        self.datacollector_cluster_fireputout.collect(self)
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step.
        """
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        self.datacollector_cluster_fine.collect(self)
        self.datacollector_cluster_fireputout.collect(self)

        self.count_step = self.count_step + 1

        # Halt if no more fire
        if self.count_type(self, "On Fire") == 0:
            self.running = False
            
    # Conta a quantidade de árvores em determinada condição, é uma variável dependente
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

    # Conta a quantidade de clusters de determinada condição de árvore, é uma variável dependente
    @staticmethod
    def count_clusters(model, width, height, condition):
        grid_zeros = np.zeros(shape = (width, height), dtype = int)

        for x in range(width):
            for y in range(height):
                if model.grid[x, y] and model.grid[x, y].condition == condition:
                    grid_zeros[x, y] = 1

        vizinhos = np.ones(shape = (3, 3), dtype = int)

        _, num = measurements.label(input = grid_zeros, structure = vizinhos)

        return num

    @staticmethod
    def total_steps(model):
        return model.count_step

    def batch_run(self):
        fix_params = {
            "width": 100,
            "height": 100
        }
        variable_params = {
            # Variáveis de controle
            "density": [0.65, 0.5],
            "wind": [43.0, 48.0],
        }
        experiments_per_parameter_configuration = 150
        max_steps_per_simulation = 100
        batch_run = BatchRunner(
            ForestFire,
            variable_params,
            fix_params,
            iterations = experiments_per_parameter_configuration,
            max_steps = max_steps_per_simulation,
            model_reporters = {
                # Variáveis dependentes
                "Number of clusters (Fine)": lambda m: self.count_clusters(m, self.width, self.height, "Fine"),
                "Number of clusters (Fire Put Out)": lambda m: self.count_clusters(m, self.width, self.height, "Fire Put Out"),
                "Total steps of the fire forest": lambda m: self.total_steps(m),
                "Fine": lambda m: self.count_type(m, "Fine"),
                "On Fire": lambda m: self.count_type(m, "On Fire"),
                "Burned Out": lambda m: self.count_type(m, "Burned Out"),
                "Fire Put Out": lambda m: self.count_type(m, "Fire Put Out"),
            },
            agent_reporters = {
                #"Condition of tree": lambda x: x.condition
            }
        )

        batch_run.run_all()

        run_model_data = batch_run.get_model_vars_dataframe()
        #run_agent_data = batch_run.get_agent_vars_dataframe()

        now = str(datetime.now()).replace(':', '-')
        file_name_sufix = ("_iter_"+str(experiments_per_parameter_configuration)+"_steps_"+str(max_steps_per_simulation)+"_"+now)
        run_model_data.to_csv("model_data"+file_name_sufix+".csv")
        #run_agent_data.to_csv("agent_data"+file_name_sufix+".csv")

forest = ForestFire()

forest.batch_run()