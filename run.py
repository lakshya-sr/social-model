import math
import mesa, colormap
from model import *
import networkx as nx
from HistogramModule import HistogramModule
import sys

def network_portrayal(G):

    c = colormap.Colormap()
    cmap = c.cmap_linear("red", "yellow", "green")
    
    def node_color(agent):
        return colormap.rgb2hex(*cmap((agent.opinion+1)/2))

    portrayal = {}
    portrayal["nodes"] = [
        {
            "size": 9 if agents[0].posting else 6,
            "color": node_color(agents[0]),
            "tooltip": f"id: {agents[0].unique_id}<br>state: {agents[0].opinion}",
        }
        for (_, agents) in G.nodes.data("agent")
    ]
    portrayal["edges"] = [ {"source": source, "target": target, "color": "grey", "width": 1} for (source, target) in G.edges]

    return portrayal

config = {}
if len(sys.argv) == 2:
    with open(sys.argv[1], "rb") as f:
        config = pickle.load(f)
else:
    config = {"num_persons":50,
              "influence_factor":0.1,
              "d_1":0.2,
              "posting_prob":0.2,
              "recommendation_post_num":2,
              "graph_degree":3,
              "cluster_min_dist":0.3,
              "G":"random_internet_as_graph",
              "influence_function":"bounded_confidence",
              "recommendation_algorithm":"AlgorithmRandom",
              "interest_function":"linear_interest",
              }
        

network = mesa.visualization.NetworkModule(network_portrayal, 500, 500)
chart = mesa.visualization.ChartModule([{"Label": "Average Opinion", "Color": "#FF0000"}])
clusters_chart = mesa.visualization.ChartModule([{"Label":"Clusters", "Color":"#00FF00"}])
bar_chart = HistogramModule(40, "Opinion")



model_params = {
    "num_persons": mesa.visualization.Slider(
        "Number of agents",
        config["num_persons"],
        10,
        100,
        1,
        description="Choose how many agents to include in the model"
    ),
    "influence_factor": mesa.visualization.Slider(
        "Influence factor",
        config["influence_factor"],
        0,
        1,
        0.01,
        description="How much are agents affected by other posts"
    ),
    "d_1": mesa.visualization.Slider(
        "Confidence bound",
        config["d_1"],
        0,
        1,
        0.01,
        description="Confidence bound for bounded confidence model"
    ),
    "posting_prob": mesa.visualization.Slider(
        "Posting probability",
        config["posting_prob"],
        0,
        1,
        0.01,
        description="Probability that an agent posts in a step"
    ),
    "recommendation_post_num": mesa.visualization.Slider(
        "No. of recommended posts",
        config["recommendation_post_num"],
        0,
        10,
        1,
        description="No of posts that are recommended in a step"
    ),
    "graph_degree": mesa.visualization.Slider(
        "Avg Node Degree", config["graph_degree"], 3, 8, 1, description="Avg Node Degree"
    ),
    "cluster_min_dist": mesa.visualization.Slider(
        "Cluster min dist", config["cluster_min_dist"], 0, 2, 0.01
    ),
    "G": mesa.visualization.Choice(
        "Network graph generator",
        value=config["G"],
        choices=["gnp_random_graph", "gnm_random_graph", "newman_watts_strogatz_graph", "random_internet_as_graph"]
    ),
    "influence_function": mesa.visualization.Choice(
        "Influence function",
        value=config["influence_function"],
        choices=["bounded_confidence", "relative_agreement", "gaussian_bounded_confidence"]
    ),
    "recommendation_algorithm": mesa.visualization.Choice(
        "Recommendation algorithm",
        value=config["recommendation_algorithm"],
        choices=["AlgorithmRandom", 
                 "AlgorithmSimilarity", 
                 "AlgorithmCollaborativeFiltering", 
                 "AlgorithmProximity", 
                 "AlgorithmHybrid", 
                 "AlgorithmPopularity", 
                 "AlgorithmPopularityProximity",
                 "AlgorithmProximityRandom"]
    ),
    "interest_function": mesa.visualization.Choice(
        "Interest function",
        value=config["interest_function"],
        choices=["linear_interest"]
    )
}

server = mesa.visualization.ModularServer(
    SocialNetwork,
    [network, chart, bar_chart],
    "Social Model",
    model_params,
)
server.port = 8000

server.launch(open_browser=False)
