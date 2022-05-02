# Independent Cascade Model code provided by Yryskeldi Emilbek uulu;
# Helper functions provided by Dr. Patrick Shepherd of Berea College;

import networkx as nx
import matplotlib.pyplot as plt
from random import random, shuffle
import random
from heapq import nlargest
import numpy
import numpy as np


def union(list1, list2):
    """
    Returns the union of two lists
    :param list1: first list
    :param list2: second list
    :return: A list containing the union over list1 and list2
    """
    ret = list1[:]
    for i in list2:
        if i not in ret:
            ret.append(i)
    return ret


def mean(mylist):
    """
    Calculates the arithmetic average of mylist
    :param mylist: The list to take the average over
    :return: Arithmetic average of mylist
    """

    # List is empty
    if len(mylist) == 0:
        return 0
    else:
        return sum(mylist) / len(mylist)


def coin_flip(prob_of_heads):
    """
    A function to return True with probability prob_of_heads,
    and False with probability 1 - prob_of_heads
    :param prob_of_heads: The probability of returning True
    :return: A boolean
    """
    if random.random() <= prob_of_heads:
        return True
    return False


def get_directed_caveman_graph(m, n, p):
    """
    Create a directed caveman graph
    :param m: Number of communities
    :param n: Number of nodes per community
    :param p: Rewiring probability
    :return: A directed caveman graph
    """
    h = nx.relaxed_caveman_graph(m, n, p)
    g = nx.DiGraph()
    for (u, v) in h.edges():
        g.add_edge(u, v)
    del h
    return g


def get_directed_small_world_graph(n, k, p):
    """
    Create a directed small world graph
    :param n: number of nodes
    :param k: number of neighbors
    :param p: rewiring probability
    :return: A networkx DiGraph object
    """
    h = nx.watts_strogatz_graph(n, k, p)
    g = nx.DiGraph()
    for (u, v) in list(h.edges()):
        g.add_edge(u, v)
    del h
    return g


def get_k_random_nodes(ic, k):
    """
    Return the k most influential nodes in ic's graph
    :param ic: An independent cascade object
    :param k: The number of nodes to find
    :return: A set of the most influential nodes
    """
    nodes = list(ic.graph.nodes())
    shuffle(nodes)
    return nodes[:k]


def update_plot(fig, ax, pos, ic, speed='lo'):
    """
    Update the plot for run_simulation if animation is turned on
    :param fig: Matplotlib figure object
    :param ax: Matplotlib axes object
    :param pos: A dictionary of [node]-->[x, y coordinates]
    :param ic: An independent cascade object
    :param isdone: Boolean, whether the simulation is finished
    :param speed: How fast to animate.  Can be 'lo', 'med', or 'hi'
    :return: None
    """

    # Clear anything off the current plot
    ax.clear()
    fig.canvas.flush_events()

    isdone = ic.is_done()

    # Calculate colors for all nodes
    colors = []
    for i in list(ic.graph.nodes()):
        if ic.node_stats[i]:
            colors.append((1., 0., 0., .5))
        else:
            if not isdone:
                colors.append((0., 0., 1., .5))
            else:
                colors.append((0., 1., 0., .5))

    # Draw the network
    nx.draw_networkx(ic.graph, node_color=colors, pos=pos)
    fig.canvas.draw()

    # Wait a different amount of time based on the speed requested
    if speed == 'hi':
        plt.pause(.2)
    if speed == 'med':
        plt.pause(.5)
    if speed == 'lo':
        plt.pause(1.)


def run_simulation(ic, animate=False):
    """
    Run a single simulation on an independent cascade model
    :param ic: An independent cascade object
    :param animate: Boolean, whether or not to animate the simulation
    :return: None
    """

    ic.reset()

    # Initialize fig, ax, plot
    # Create plot if desired
    fig, ax, pos = None, None, None
    if animate:
        fig, ax = plt.subplots()
        pos = nx.spring_layout(ic.graph)
        update_plot(fig, ax, pos, ic)

    # Update the model as long as there are still possible
    # activations
    while not ic.is_done():
        ic.update()
        if animate:
            update_plot(fig, ax, pos, ic)

    # Plot final state of the network if desired
    if animate:
        update_plot(fig, ax, pos, ic, speed='hi')
        plt.show()


class ICModel():
    """
    A class to carry out Independent Cascade diffusion
    """

    def __init__(self, g, activation_prob):
        """
        Constructor for Independent Cascade model
        :param g: A networkx graph
        :param nodes: A list of nodes to be activated at the beginning
        :param activation_prob: The probability for each edge to activate a neighbor
        """

        # Store information about node and edge activation status
        # A node u is 'active' if self.node_stats[u]['active'] == True
        # An edge (u,v) is 'active' if self.edge_stats[u,v]['active'] == True
        # Edge weights are the probability that u will activate v
        self.graph = g
        self.node_stats = {}
        self.edge_stats = {}
        self.edge_weights = {}

        # Store references to initial conditions
        self.initial_nodes = []
        self.initial_activation_prob = activation_prob

        # Set all nodes' and edges' active status, as well as activation
        # probabilities
        self.reset()

    def set_initial_node_status(self):
        """
        Set initial statuses to False for all nodes not in the list
        nodes.  Set status to True for all nodes in the list nodes.
        :return: None
        """
        for i in list(self.graph.nodes()):
            if i in self.initial_nodes:
                self.node_stats[i] = True
            else:
                self.node_stats[i] = False

    def activate_nodes(self, nodes):
        """
        Set the list of initially active nodes to 'nodes'
        :param nodes: The list of nodes to activate
        :return: None
        """
        self.initial_nodes = nodes
        self.set_initial_node_status()

    def set_initial_edge_status(self):
        """
        Set initial activation status of all edges to True.
        If an edge's status is True, the edge can still be used.
        :return: None
        """
        for i in self.graph.edges():
            self.edge_stats[i] = True
            self.edge_weights[i] = self.initial_activation_prob

    def reset(self):
        """
        Reset the initial statuses of all nodes and edges.
        :return: None
        """
        self.set_initial_node_status()
        self.set_initial_edge_status()

    def get_num_activated(self):
        """
        Calculate the number of active nodes.
        :return: Number of active nodes
        """
        stats = [self.node_stats[key] for key in self.node_stats]
        return stats.count(True)

    def get_activated_nodes(self):
        """
        Get the indexes of all activated nodes.
        :return: List of node indexes
        """
        return [i for i in list(self.graph.nodes()) if self.node_stats[i]]

    def is_done(self):
        """
        Checks to see if there are any more possible activations.
        :return: True if there are no more activations possible, False otherwise
        """

        # Iterate through all edges
        for (u, v) in self.graph.edges():
            u_active = self.node_stats[u]
            v_active = self.node_stats[v]
            edge_active = self.edge_stats[u, v]

            # If the source node is activated, the destination node is
            # not, and the edge can still be used, then there is another
            # potential activation waiting to be tried.
            if (u_active) and (not v_active) and (edge_active):
                return False

        # If no potential activations, we are done.
        return True

    def update(self):
        """
        Update activation statuses of all nodes and edges.
        :return: None
        """

        # Store updates to execute later.
        updates = []
        edges = list(self.graph.edges())

        # Iterate over all edges
        for (u, v) in edges:
            # Do nothing if u is not currently active
            if not self.node_stats[u]:
                continue
            # Do nothing if v is already activated
            if self.node_stats[v]:
                continue
            # Do nothing if edge (u, v) has already been
            # tried but failed
            if not self.edge_stats[u, v]:
                continue

            # Otherwise, flip a coin and update v if necessary.
            heads = coin_flip(self.edge_weights[u, v])
            if heads:
                updates.append(v)
            self.edge_stats[u, v] = False

        # Perform all queued updates
        for node in updates:
            self.node_stats[node] = True



def get_average_influence_set_size(ic, node, numreps=20):
    """
    Calculate the average number of nodes activated, directly and
    indirectly, by 'node'
    :param ic: The independent cascade object to examine
    :param node: The node to affect initially
    :param numreps: Number of steps to average over
    :return: The average number of other nodes activated by 'node'
    """
    # Task 1
    # The loop below resets the graph for each repetition and then activates
    # the node indicated.  You must fill in the rest of the code to store
    # activation set sizes.
    avglst = []
    for i in range(numreps):
        ic.reset()
        ic.activate_nodes([node])
        ic.update()
        while not ic.is_done():
            ic.update()
        avglst.append(ic.get_num_activated())
    return mean(avglst)


def get_influenced_neighbors(ic, node, numreps=20):
    """
    Record all nodes activated by 'node' over numreps different runs.
    :param ic: The independent cascade object to examine
    :param node: The node to affect initially
    :param numreps: Number of steps to average over
    :return: A list of the k most frequently influenced nodes
    """
    # Task 2
    # The loop below resets the graph for each repetition and then activates
    # the node indicated.  You must fill in the rest of the code to store
    # neighbor sets.
    activatedNodes = []
    for i in range(numreps):
        ic.reset()
        ic.activate_nodes([node])
        ic.update()
        while not ic.is_done():
            ic.update()
        activatedNodes = union(activatedNodes, ic.get_activated_nodes())
    return activatedNodes


def get_k_influential_nodes_a(ic, k):
    """
    Return the k most influential nodes in ic's graph, based on the
    function get_average_influence_set_size
    :param ic: An independent cascade object
    :param k: The number of nodes to find
    :return: A set of the most influential nodes
    """
    # Task 3
    myDict = {}
    for i in list(ic.graph.nodes()):
        myDict[i] = get_average_influence_set_size(ic, i)
    kHighestNodes = nlargest(k, myDict, key=myDict.get)
    return kHighestNodes


def get_k_influential_nodes_b(ic, k):
    """
    Return the k most influential nodes in ic's graph, based on the
    function get_influenced_neighbors
    :param ic: An independent cascade object
    :param k: The number of nodes to find
    :return: A set of the most influential nodes
    """
    # Task 4
    myDict = {}  # to store influence set of each node
    for i in list(ic.graph.nodes()):
        myDict[i] = get_influenced_neighbors(ic, i)
    LongestInfluenceSetNode = max(myDict, key=lambda x: len(set(myDict[x])))
    SetOfNodes = []
    SetOfNodes.append(LongestInfluenceSetNode)
    if k == 1:
        return SetOfNodes
    else:
        for i in range(k - 1):
            myDict2 = {}  # to store gain in unique nodes
            for x in list(ic.graph.nodes()):
                if x not in SetOfNodes:
                    HypotheticalSet = union(SetOfNodes, myDict[x])
                    gain = len(HypotheticalSet) - len(SetOfNodes)
                    myDict2[x] = gain
            HighestGainNode = nlargest(1, myDict2, key=myDict2.get)
            for y in HighestGainNode:
                SetOfNodes.append(y)
        return SetOfNodes

def upload_edges(): # upload edges
    G=nx.Graph()
    nodeData = open('quaker-nodes.csv', "r")
    nodeList = nodeData.readlines()
    Nodes = [x[:-1] for x in nodeList[1:]]
    G.add_nodes_from(Nodes)
    edgeData = open('quaker-edges.csv', "r")
    newList = edgeData.readlines()
    testEdges = [x[:-1] for x in newList[1:]]
    Edges = [tuple(map(str, sub.split(','))) for sub in testEdges]
    G.add_edges_from(Edges)
    # Debugging code (had to change all code instances from range(ic.graph.number_of_nodes) to list(ic.graph.nodes()) because nodes were not consecutive
    # def checkConsecutive(l):
    #     return sorted(l) == list(range(min(l), max(l) + 1))
    # isit = checkConsecutive(G.nodes)
    return G


def common_data(list1, list2):
    """
    From geeks2geeks: checks if two lists have common elements.
    :param list1: first list to check
    :param list2: second list to check
    :return:
    """
    result = False

    # traverse in the 1st list
    for x in list1:

        # traverse in the 2nd list
        for y in list2:

            # if one common
            if x == y:
                result = True
                return result

    return result

def main():
    G = upload_edges()
    print("uploaded edges")
    # pos = nx.kamada_kawai_layout(G)
    # nx.draw(G,with_labels=False, node_size=25, alpha=0.90, linewidths=5, pos=pos)
    # plt.show()
    ic = ICModel(G, .25)
    myDict3 = {}  # to store how many nodes were activated for each of the 10 simulations for each number of initial nodes
    for i in range(40):  # how many simulations
        print("starting simulation " + str(i))
        for x in ['Random algorithm', 'Algorithm A', 'Algorithm B']:  # how many initial nodes to start with
            NodesToActivate = get_k_influential_nodes_a(ic,10)
            RandomNodes = get_k_random_nodes(ic, 2)
            for i in NodesToActivate:
                if i in RandomNodes:
                    RandomNodes.remove(i)
                    NodesToActivate.append(get_k_random_nodes(ic,1))
            ic.activate_nodes(NodesToActivate)
            print("found activation nodes for simulation " + str(i) + " iteration " + str(x))
            run_simulation(ic, animate=False)
            try:  # if it's the first simulation, first define value in dict as a list
                myDict3[x] = [ic.get_num_activated()]
                ic.reset()
            except:  # if latter simulation, append to the existing list as value in dict
                myDict3[x] += [ic.get_num_activated()]
                ic.reset()
            print("done: simulation " + str(i) + ", initial nodes: " + str(x))
    print(myDict3)

    plt.boxplot(list(myDict3.values()))
    plt.xlabel('Number of initial nodes')
    plt.ylabel('Number of Activated Nodes')
    plt.xticks(list(range(11)),list(range(0, 21, 2)))
    plt.show()

    # for ANOVA:
    # myList = []
    # for i in range(40):  # how many simulations
    #     print("starting simulation " + str(i))
    #     AlgorithmNodesToActivate = get_k_influential_nodes_b(ic,10)
    #     RandomNodesToActivate = get_k_random_nodes(ic,2)
    #     if common_data(AlgorithmNodesToActivate, RandomNodesToActivate):
    #         RandomNodesToActivate.append(get_k_random_nodes(ic,1))
    #     NodesToActivate = union(AlgorithmNodesToActivate, RandomNodesToActivate)
    #     ic.activate_nodes(NodesToActivate)
    #     print("found activation nodes for simulation " + str(i))
    #     run_simulation(ic, animate=False)  # if it's the first simulation, first define value in dict as a list
    #     myList.append(ic.get_num_activated())
    #     ic.reset()
    # print(myList)

    # for t-tests
    # myList = []
    # for i in range(40):  # how many simulations
    #     print("starting simulation " + str(i))
    #     NodesToActivate = get_k_influential_nodes_b(ic,12)
    #     ic.activate_nodes(NodesToActivate)
    #     print("found activation nodes for simulation " + str(i))
    #     run_simulation(ic, animate=False)  # if it's the first simulation, first define value in dict as a list
    #     myList.append(ic.get_num_activated())
    #     ic.reset()
    # print(myList)


main()
