from model import SocialNetworkBatchModel, SocialNetworkAgent, bounded_confidence
from utils import interval
import mesa, colormap
import utils



def agent_portrayal(a):
    c = colormap.Colormap()
    cmap = c.cmap_linear("red", "yellow", "green")
    
    color = colormap.rgb2hex(*cmap((a.m.average_opinion()+1)/2))
    portrayal = {"Color" : color,
                 "Layer":0,
                 "Shape": "rect", 
                 "Filled": "true", 
                 "w": 1, "h": 1, 
                 "tooltip": f"{a.m.d_1:.3f}, {a.m.influence_factor:.3f}"}
    return portrayal
 
if __name__ == "__main__":
    
    influence_factor = interval(0,1,10)
    d_1 = interval(0,0.5,10)
    variables = [influence_factor, d_1]
    network = mesa.visualization.CanvasGrid(agent_portrayal, 10, 10, 500, 500)
    chart = mesa.visualization.ChartModule(
        [
            
            
        ]
    )

    model_params = {
        "num_threads": mesa.visualization.Slider(
            "No of threads to execute models", 
            2, 
            1,
            8,
            1),
        "configs": {
            "num_persons" : 50,
            "influence_factor" : influence_factor,
            "d_1" : d_1,
            "d_2" : 1,
            "posting_prob" : 0.2,
            "recommendation_post_num" : 2,
            "graph_degree" : 3,
            "collect_data" : False,
            "G" : "gnp_random_graph",
            "influence_function" : "bounded_confidence",
            "recommendation_algorithm" : "AlgorithmCollaborativeFiltering"}
    }

    server = mesa.visualization.ModularServer(
        SocialNetworkBatchModel,
        [network, chart],
        "Social Model",
        model_params,
    )
    server.port = 8000
    server.launch(open_browser=True)
