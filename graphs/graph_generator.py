import networkx as nx
import matplotlib.pyplot as plt
import pickle

n=100
g = nx.random_internet_as_graph(n)


directed_g = g.to_directed()
edges = [e for e in directed_g.edges]
nodes = [n for n in directed_g.nodes]

print(g.to_directed.__doc__)
print(g.remove_edge.__doc__)
print(g.adj.__doc__)
for n in directed_g.nodes:
    if len(directed_g.adj[n]) == 1:
        for n2 in directed_g.adj[n]: 
            directed_g.remove_edge(n2,n)

nx.draw(directed_g, arrows=True)
plt.show()
with open("internet_like.graph", "wb") as f:
    pickle.dump(directed_g, f)

