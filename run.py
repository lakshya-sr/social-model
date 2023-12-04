import math

import mesa, colormap

from model import SocialNetwork


def network_portrayal(G):

    c = colormap.Colormap()
    cmap = c.cmap_linear("red", "yellow", "green")
    
    def node_color(agent):
        return colormap.rgb2hex(*cmap((agent.opinion+1)/2))
    # colormap.rgb2hex(*(round((agents[0].opinion+1)*(255/2)),)*3)
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
        },
        {"Label": "Cluster",
         "Color": "#00FF00"}
        
    ]
)



model_params = {
    "num_persons": mesa.visualization.Slider(
        "Number of agents",
        10,
        10,
        100,
        1,
        description="Choose how many agents to include in the model",
    ),
    "influence_factor": mesa.visualization.Slider(
        "Influence factor",
        0.1,
        0,
        1,
        0.01,
        description="How much are agents affected by other posts",
    ),
    "graph_degree": mesa.visualization.Slider(
        "Avg Node Degree", 3, 3, 8, 1, description="Avg Node Degree"
    )
}

server = mesa.visualization.ModularServer(
    SocialNetwork,
    [network, chart],
    "Social Model",
    model_params,
)
server.port = 8011

server.launch(open_browser=True)
