from collections import deque
import networkx as nx
import random
import utils
import math
import pickle
import time
import mesa
from multiprocessing import Pool


like_threshold = 4

logoff_threshold = 20
post_opinion_delta = 0.02

CREATE_POST = 1
FOLLOW = 2
UNFOLLOW = 3
LIKE = 4
FORWARD = 5

    

def bounded_confidence(person_opinion, post_opinion, person_confidence, post_confidence, influence_factor):
    x, x_, u, u_, mu = person_opinion, post_opinion, person_confidence, post_confidence, influence_factor
    k = 1 if abs(x-x_) < u else 0
    return mu*k*(x_ - x), mu*k*(u_ - u)
    
def bounded_confidence_repulsion(person_opinion, post_opinion, person_confidence, repulsion_cutoff, influence_factor):
    x, x_, u, u_, mu = person_opinion, post_opinion, person_confidence, repulsion_cutoff, influence_factor
    d = x_-x
    k = 1 if abs(d) < u else (-1 if abs(d) > u_ else 0)
    return mu*k*(x_ - x), 0

def gaussian_bounded_confidence(person_opinion, post_opinion, person_confidence, post_confidence, influence_factor):
    x, x_, u, u_, mu = person_opinion, post_opinion, person_confidence, post_confidence, influence_factor
    k = math.e**(-((x-x_)**2)/u)
    return mu*k*(x_ - x), 0
    
def relative_agreement(person_opinion, post_opinion, person_confidence, post_confidence, influence_factor):
    x, x_, u, u_, mu = person_opinion, post_opinion, person_confidence, post_confidence, influence_factor
    v = min(x+u, x_+u_) - max(x-u, x_-u_)
    k = (v/u_) - 1 if v > u_ else 0
    return mu*k*(x_ - x), mu*k*(u_ - u)
    
def moving_average(data, window_width):
    data_padded = [0 if i-window_width<=0 else(0 if i+window_width > len(data) else data[i-window_width]) for i in range(len(data)+2*window_width)]
    data_final = [0 for i in range(len(data))]
    for i in range(window_width, len(data)+window_width):
        window = data_padded[i-window_width:i+window_width]
        data_final[i-window_width] = sum(window)/(2*window_width)
    return data_final

def num_clusters(model):
    data = [p.opinion for p in model.schedule.agents]
    data.sort()
    max_consecutive_dist = 0
    for i in range(len(data)-1):
        max_consecutive_dist = data[i]-data[i+1] if data[i]-data[i+1] > max_consecutive_dist else max_consecutive_dist
    return max_consecutive_dist
        
    


def interest(person_opinion, post_opinion):
    diff = abs(person_opinion-post_opinion)
    a = 1.1
    x_0 = 1
    k = 5
    return (-1/(math.e**(-k*(diff-x_0))))+0.5


class SocialNetwork(mesa.Model):
    def __init__(self, num_persons=10,
                       influence_factor=0.5,
                       d_1=0.2,
                       d_2=0.8, 
                       posting_prob=0, 
                       recommendation_post_num=2, 
                       graph_degree=3, 
                       collect_data=True, 
                       G=None, 
                       influence_function=bounded_confidence,
                       recommendation_algorithm=None):
        self.posts = []

        self.num_persons = num_persons
        self.influence_factor, self.d_1, self.d_2 = influence_factor, d_1, d_2 
        self.posting_prob = posting_prob
        self.recommendation_post_num = recommendation_post_num
        self.collect_data = collect_data
        self.influence = influence_function

        self.algorithm = recommendation_algorithm() if recommendation_algorithm else AlgorithmRandom()
        self.datacollector = mesa.DataCollector({"Average Opinion": self.average_opinion, "Cluster": num_clusters})
        self.G = G if G else nx.gnp_random_graph(self.num_persons, graph_degree/self.num_persons, directed=True)
        self.network = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)

        for i, node in enumerate(self.G.nodes()):
            a = Person(i, self)
            self.schedule.add(a)
            self.network.place_agent(a, node)

        for p in self.schedule.agents:
            p.create_post()
        
    def step(self):
        self.schedule.step()
        if self.collect_data:
            self.datacollector.collect(self)

    
    def run(self, iterations, filename=None):
        start_time = time.perf_counter()
        for i in range(iterations):
            self.step()
        exec_time = time.perf_counter() - start_time
        if filename:
            with open(filename, "wb") as f:
                pickle.dump(self.datacollector.steps, f)
        return exec_time
        
    def average_opinion(self):
        return sum([p.opinion for p in self.schedule.agents])/self.num_persons
        
class Post():
    def __init__(self, opinion, creator, confidence):
        self.opinion = opinion
        self.creator = creator
        self.confidence = self.creator.model.d_2
        self.truth = 0
        self.likes = 0
        
        
class Person(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model
        self.feed = []
        self.logged_in = True
        self.interest = 0
        self.history = deque([], maxlen=10)
        self.logoff_time = 0
        self.opinion = random.uniform(-1, 1)
        self.confidence = self.model.d_1 if self.model.influence in [bounded_confidence, bounded_confidence_repulsion, gaussian_bounded_confidence] else random.uniform(-0.5, 0.5)
        self.followers = []

    def step(self):
        self.feed.extend(self.model.algorithm.select_posts(self, self.model.posts, self.model.recommendation_post_num))
        if len(self.feed) > 0:
            p = random.choice(self.feed)
            if p: self.feed.remove(p)
            
            if type(p) == Post:
                opinion_delta, confidence_delta = self.model.influence(self.opinion, p.opinion, self.confidence, p.confidence, self.model.influence_factor)
                self.opinion = utils.clamp(self.opinion + opinion_delta, -1, 1)
                self.confidence = utils.clamp(self.confidence + confidence_delta, -1, 1)

                
        if random.random() < self.model.posting_prob:
            self.create_post()
        pass
        

    def create_post(self):
        post = Post(random.uniform(self.opinion-post_opinion_delta, self.opinion+post_opinion_delta), self, self.model.d_2 if self.model.influence == bounded_confidence_repulsion else self.confidence)
        self.history.append((CREATE_POST, post))
        self.model.posts.append(post)
        return post

    def forward(self, person, post):
        if person != self:
            person.feed.append(post)
            self.history.append((FORWARD, person, post))

    def follow(self, person):
        if person != self:
            self.model.network.add_edge(self, person)
            person.followers.append(self)
            self.history.append((FOLLOW, person))

    def unfollow(self, person):
        try:
            self.model.network.remove_edge(self, person)
            person.followers.remove(self)
        except:
            return
        self.history.append((UNFOLLOW, person))

    def like(self, post):
        post.likes += 1
        self.history.append((LIKE, post))



class AlgorithmRandom():
    def __init__(self):
        pass

    def select_persons(self, target, persons, n):
        return random.sample(persons, n)
    
    def select_posts(self, target, posts, n):
        return random.sample(posts, n)


class AlgorithmSimilarity():
    def __init__(self):
        pass

    def select_posts(self, target, posts, n):
        selected = []
        idx = 0
        for p in posts:
            if target.opinion > p.opinion:
                idx += 1
            else:
                break
        
        selected = posts[idx-n:idx+n]
        final = []
        for p in selected:
            if p.creator != target:
                final.append(p)
        
        if len(final) > n:
            return random.sample(final, n)
        else:
            return final

class MultithreadedBaseScheduler(mesa.time.BaseScheduler):
    def __init__(self, model, num_threads):
        super().__init__(model)
        self.model = model
        self.pool = Pool(num_threads)

    def on_each(self, method):
        for agent_key in self.get_agent_keys():
            f = getattr(self._agents[agent_key], method)
            self.pool.apply_async(f)

class SocialNetworkBatchModel(mesa.Model):
    def __init__(self, num_threads, configs):
        self.schedule = MultithreadedBaseScheduler(self, num_threads)
        self.variables = []
        self.num_agents = 0
        self.datacollector = mesa.DataCollector()
        for k,v in configs.items():
            if type(v) == list:
                self.variables.append(k)

        config = {}
        if len(self.variables) > 2:
            print("Too many variable parameters")
            return
        elif len(self.variables) == 2:
            self.grid = mesa.space.SingleGrid(len(configs[self.variables[0]]), len(configs[self.variables[1]]), False)
            for i in range(len(configs[self.variables[0]])):
                for j in range(len(configs[self.variables[1]])):
                    for k,v in configs.items():
                        if k in self.variables:
                            config[k] = configs[k][i if self.variables.index(k) == 1 else j]
                        else:
                            config[k] = v
                    s = SocialNetworkAgent(self.num_agents, self, config)
                    self.schedule.add(s)
                    self.grid.place_agent(s, (i,j))
                    self.num_agents += 1
        elif len(self.variables) == 1:
            self.grid = mesa.space.SingleGrid(len(configs[self.variables[0]]), 0, False)
            for i in range(len(configs[self.variables[0]])):
                for k,v in configs.items():
                    if k in self.variables:
                        config[k] = configs[k][i]
                    else:
                        config[k] = v
                s = SocialNetworkAgent(self.num_agents, self, config)
                self.schedule.add(s)
                self.grid.place_agent(s, (i,0))
                self.num_agents += 1

    def step(self):
        self.schedule.step()


class SocialNetworkAgent(mesa.Agent):
    def __init__(self, unique_id, model, config):
        self.m = SocialNetwork(config["num_persons"],
                                   config["influence_factor"],
                                   config["d_1"],
                                   config["d_2"],
                                   config["posting_prob"],
                                   config["recommendation_post_num"],
                                   config["graph_degree"],
                                   config["collect_data"],
                                   config["G"],
                                   config["influence_function"],
                                   config["recommendation_algorithm"]
                                  )
        self.unique_id = unique_id
        self.model = model
        self.stats = None

    def step(self):
        print(f"Stepped {self.unique_id}")
        self.m.step()
        self.collect_stats(self.model)

    def collect_stats(self, model):
        pass
