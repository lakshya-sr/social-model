import math
import mesa, colormap
from model import *
import networkx as nx
from HistogramModule import HistogramModule

def network_portrayal(G):

    c = colormap.Colormap()
    cmap = c.cmap_linear("red", "yellow", "green")
    
    def node_color(agent):
        return colormap.rgb2hex(*cmap((agent.opinion+1)/2))

    portrayal = {}
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0]),
            "tooltip": f"id: {agents[0].unique_id}<br>state: {agents[0].opinion}",
        }
        for (_, agents) in G.nodes.data("agent")
    ]
    portrayal["edges"] = [ {"source": source, "target": target, "color": "grey", "width": 1} for (source, target) in G.edges]
    
    return portrayal

network = mesa.visualization.NetworkModule(network_portrayal, 500, 500)
chart = mesa.visualization.ChartModule(
    [
        {"Label": "Average Opinion",
         "Color": "#FF0000"
        }
    ]
)

bar_chart = HistogramModule(20, "Opinion")


model_params = {
    "num_persons": mesa.visualization.Slider(
        "Number of agents",
        10,
        10,
        100,
        1,
        description="Choose how many agents to include in the model"
    ),
    "influence_factor": mesa.visualization.Slider(
        "Influence factor",
        0.1,
        0,
        1,
        0.01,
        description="How much are agents affected by other posts"
    ),
    "d_1": mesa.visualization.Slider(
        "Confidence bound",
        0.2,
        0,
        1,
        0.01,
        description="Confidence bound for bounded confidence model"
    ),
    "posting_prob": mesa.visualization.Slider(
        "Posting probability",
        0.2,
        0,
        1,
        0.01,
        description="Probability that an agent posts in a step"
    ),
    "recommendation_post_num": mesa.visualization.Slider(
        "No. of recommended posts",
        2,
        0,
        10,
        1,
        description="No of posts that are recommended in a step"
    ),
    "graph_degree": mesa.visualization.Slider(
        "Avg Node Degree", 3, 3, 8, 1, description="Avg Node Degree"
    ),
    "G": mesa.visualization.Choice(
        "Network graph generator",
        value="gnp_random_graph",
        choices=["gnp_random_graph", "gnm_random_graph", "newman_watts_strogatz_graph"]
    ),
    "influence_function": mesa.visualization.Choice(
        "Influence function",
        value="bounded_confidence",
        choices=["bounded_confidence", "relative_agreement", "gaussian_bounded_confidence"]
    ),
    "recommendation_algorithm": mesa.visualization.Choice(
        "Recommendation algorithm",
        value="AlgorithmRandom",
        choices=["AlgorithmRandom", "AlgorithmSimilarity", "AlgorithmCollaborativeFiltering"]
    ),
    "interest_function": mesa.visualization.Choice(
        "Interest function",
        value="linear_interest",
        choices=["linear_interest"]
    )
}

server = mesa.visualization.ModularServer(
    SocialNetwork,
    [network, chart, bar_chart],
    "Social Model",
    model_params,
)
server.port = 8001

server.launch(open_browser=True)
