import random
import networkx as nx

def clamp(x, minv, maxv):
    return max(minv, min(maxv, x))


def interval(start, stop, n):
    return [start+((stop-start)*i/n) for i in range(n)]

def random_array(n):
    return [random.random() for i in range(n)]

def histogram(data, bins=20, x_range=(-1,1)):
    maxv, minv = x_range[1], x_range[0]
    dx = (maxv-minv)/bins
    start, end = minv, minv+dx
    hist_data = []
    for i in range(bins):
        count = 0
        for v in data:
            if start < v <= end:
                count += 1
        hist_data.append((count, start, end))
        start += dx
        end += dx
    return hist_data

def generate_graph(graph_spec, num_persons, graph_degree, directed=True):
    if type(graph_spec) == nx.DiGraph:
            return graph_spec
    elif type(graph_spec) == str:
        if graph_spec == "gnp_random_graph":
            return nx.gnp_random_graph(num_persons, graph_degree/num_persons, directed=directed)
                
        elif graph_spec == "gnm_random_graph":
            return nx.gnm_random_graph(num_persons, graph_degree*num_persons, directed=directed)

        elif graph_spec == "newman_watts_strogatz_graph":
            G = nx.newman_watts_strogatz_graph(num_persons, graph_degree, graph_degree/num_persons).to_directed()
            for _ in range(len(G.edges)//2):
                G.remove_edge(random.choice(G.edges))
            return G
    
                    
def get_algorithm_config(algorithm, G, interest_matrix, pair_dist):
    config = {}
    if algorithm == "AlgorithmCollaborativeFiltering":
        config = {"interest_matrix": interest_matrix}
    elif algorithm == "AlgorithmProximity":
        config = {"pair_dist": pair_dist}
    elif algorithm == "AlgorithmHybrid":
        config = {"pair_dist":pair_dist, "interest_matrix":interest_matrix}
    return config
