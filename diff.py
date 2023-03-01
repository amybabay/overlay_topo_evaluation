import sys
import networkx as nx
from networkx.algorithms.connectivity import edge_disjoint_paths, node_disjoint_paths
from networkx.algorithms.flow import shortest_augmenting_path
import matplotlib.pyplot as plt
import statistics
from itertools import islice
import numpy as np
import copy
from operator import itemgetter

graph = {}
E = {}
topology = {}
num=None

    
    
def read_positions(pos_file,ignore):
    """ Read node positions from the given pos_file. Return a dictionary that
    maps each node in the file to a tuple (x,y) with its x,y coordinates """
    pos = {}
    with open(pos_file) as f:
        for line in f:
            # each line is like:     1 41.505880 -81.609169 # Case Western
            parts = line.split()
            node = int(parts[0])
            if node in ignore: continue # to ignore any nodes
            lat = float(parts[1])
            lon = float(parts[2])
            pos[node] = (lon, lat)
    return pos
    
def create_graph(input_graph,ignore):
    with open(input_graph) as f:
        for line in f:
            parts = line.split()
            if parts != []:
                node1 = int(parts[0])
                node2 = int(parts[1])
                if node1 in ignore or node2 in ignore : continue
                weight = int(parts[2])
                E[(node1,node2)] = int(weight)
                if not(node1 in graph):
                    graph[node1]={}
                graph[node1][node2] = int(weight)

def create_topology(input_graph,ignore):
    with open(input_graph) as f:
        for line in f:
            parts = line.split()
            if parts != []:
                node1 = int(parts[0])
                node2 = int(parts[1])
                if node1 in ignore or node2 in ignore : continue
                weight = int(parts[2])
                if not(node1 in topology):
                    topology[node1]={}
                topology[node1][node2] = int(weight)

def dfs(visited,node,end,path,edge):
    if node in visited:
        return 
    path.append(node)
    
    if node == end:
        edge.append(copy.deepcopy(path))
        path.remove(node)
        return
    visited.append(node)
    for nei in topology[node]:
        dfs(visited,nei,end,path,edge)
    visited.remove(node)
    path.remove(node)
    
def add_path(path_sum,n,m,edge):
    l=len(edge)
    print(f'has {l} paths')
    if l not in list(path_sum.keys()):
        path_sum[l]=[]
    path_sum[l].append([n,m])
    
    #print(path_sum)

def add_cal_weight(edge, weights,min_P,max_P,max_all_P,Orig):
    print(f"direct:{Orig}\n")
    for i in edge:
        weight = 0
        for j in range(len(i)-1):
            #print(E[(i[j],i[j+1])])
            weight += E[(i[j],i[j+1])]/2
        weights.append((weight,i))
    
    if len(edge) >= 3:
        weights.sort(key=lambda x:x[0])
        Min = weights[0]
        Max = weights[2]
        Max_of_all = weights[-1]
        min_P.append(((Min[0] - Orig)/Orig*100,Min[0] - Orig,Min[1]))
        max_P.append(((Max[0] - Orig)/Orig*100,Max[0] - Orig,Max[1]))
        max_all_P.append(((Max_of_all[0] - Orig)/Orig*100,Max_of_all[0] - Orig,Max_of_all[1]))
        print(weights)
        print(f"min : {Min}, max : {Max}, original : {Orig}, max of 6th pathes: {Max_of_all}")
        print(f"min : {round((Min[0] - Orig)/Orig*100)}% {Min[0] - Orig} , max : {round((Max[0] - Orig)/Orig*100)}% {Max[0] - Orig}, max of 6th pathes: {round((Max_of_all[0] - Orig)/Orig*100)}% {Max_of_all[0] - Orig}\n")
    else:
        print("not enough paths")

def sum_path(path_sum):
    for num in sorted(path_sum.keys()):
        print(f"{len(list(path_sum[num]))} pairs of nodes have {num} disjoint path")

def sum_weight(p):
    l=[]
    for i in p: l.append(i[0])
    form("min:",min(p))
    form("max:", max(p))
    print(f"mean:{round(statistics.mean(l))}%")

def form (mes,l):
    print(f"{mes} {round(l[0])}%             weight: {l[1]}             path: {l[2]}")

def get_total_edge():
    result=0
    for n in topology:
        result += len(topology[n])-1
    return result/2
    

def each_degree():
    degree={}
    total_dis_over_deg = 0
    print( "\nDegrees   total disjoint path   total disjoint path/Degrees: ")
    for i in topology:
        degree[i]=len(topology[i])-1
        print(f"node {i} has {degree[i]}            {topology[i]['disjoint']}            {round(topology[i]['disjoint']/degree[i],2)}  ")
        total_dis_over_deg += topology[i]['disjoint']/degree[i]
    print(total_dis_over_deg/get_total_edge())

 
    
def main(argv):
    inputGraph = argv[0]
    posGraph = argv[1]
    result = argv[2]
    ignore = [int(i) for i in argv[3:]]
    print ("--ignore nodes: ",ignore,"--\n")
    

    create_graph(inputGraph,ignore)
    create_topology(result,ignore)
    pos = read_positions(posGraph,ignore)
    
    

    draw = nx.DiGraph()
    for n in topology:
        for m in topology[n]:
            weight = topology[n][m]
            draw.add_edge(n, m, weight=weight)
            draw.add_edge(m, n, weight=weight)

    """
    from here is the caculation of one-way absolute latency difference 
    """

    path_sum={}
    min_P = []
    max_P = []
    max_all_P = []
    for n in graph:
        total_disjoint_path=0
        for m in graph[n]:
            edge = []
            visited = []
            path = []
            weights = []
            Orig = E[(n,m)]/2
            print(f"{n}->{m}")
            #dfs(visited, n, m,path,edge)
            edge = list(nx.node_disjoint_paths(draw,n,m,flow_func=shortest_augmenting_path,cutoff=num))
            add_path(path_sum,n,m,edge)
            add_cal_weight(edge, weights,min_P,max_P,max_all_P,Orig)
            total_disjoint_path += len(edge)
        topology[n]['disjoint'] = total_disjoint_path

        
    print ("\n--ignore nodes: ",ignore,"--\n")        
    sum_path(path_sum)
    print("\nthe first min weight sum:")
    sum_weight(min_P)
    print("\nthe 3ed min weight sum:")
    sum_weight(max_P)
    if num != 3:
        print("\nthe max weight sum:")
        sum_weight(max_all_P)
    totalEdges = get_total_edge()
    print(f"total edges is {totalEdges}")
    degree = each_degree()
    nx.draw_networkx(draw, pos)
    plt.show()
     
        


if __name__ == "__main__":
    main(sys.argv[1:])