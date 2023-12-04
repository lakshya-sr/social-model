from model import SocialNetworkBatchModel, SocialNetworkAgent, bounded_confidence
from utils import interval
import mesa, colormap



def agent_portrayal(a):
    portrayal = {"Color" : "red", "Layer":0}
    return portrayal
 
if __name__ == "__main__":

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
            "influence_factor" : interval(0, 1, 10),
            "d_1" : interval(0, 0.4, 10),
            "d_2" : 1,
            "posting_prob" : 0.2,
            "recommendation_post_num" : 2,
            "graph_degree" : 3,
            "collect_data" : True,
            "G" : None,
            "influence_function" : bounded_confidence,
            "recommendation_algorithm" : None}
    }

    server = mesa.visualization.ModularServer(
        SocialNetworkBatchModel,
        [network, chart],
        "Social Model",
        model_params,
    )
    server.port = 8011
    server.launch(open_browser=True)
