from mesa.visualization.modules import CanvasGrid, ChartModule, PieChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import ForestFire

COLORS = {"Fine": "#00AA00", "On Fire": "#880000", "Burned Out": "#000000", "Fire Put Out": "#0000FF"}


def forest_fire_portrayal(tree):
    if tree is None:
        return
    portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0}
    (x, y) = tree.pos
    portrayal["x"] = x
    portrayal["y"] = y
    portrayal["Color"] = COLORS[tree.condition]
    return portrayal


canvas_element = CanvasGrid(forest_fire_portrayal, 100, 100, 500, 500)
cluster_chart_fine = ChartModule(
    [{"Label": "Number of clusters (Fine)", "Color": "#00AA00"}], data_collector_name="datacollector_cluster_fine"
)
cluster_chart_fireputout = ChartModule(
    [{"Label": "Number of clusters (Fire Put Out)", "Color": "#0000FF"}], data_collector_name="datacollector_cluster_fireputout"
)
tree_chart = ChartModule(
    [{"Label": label, "Color": color} for (label, color) in COLORS.items()]
)
pie_chart = PieChartModule(
    [{"Label": label, "Color": color} for (label, color) in COLORS.items()]
)

model_params = {
    "height": 100,
    "width": 100,
    "density": UserSettableParameter("slider", "Tree density", 0.65, 0.01, 1.0, 0.01),
    "wind": UserSettableParameter("slider", "Wind velocity (m/s)", 10.0, 0.0, 50.0, 0.5)
}
server = ModularServer(
    ForestFire, [canvas_element, tree_chart, pie_chart, cluster_chart_fine, cluster_chart_fireputout], "Forest Fire", model_params
)
