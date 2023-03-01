import sys
import argparse
import re
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import dijkstra_path_length, dijkstra_path
from networkx.algorithms.connectivity import node_disjoint_paths
from networkx.algorithms.flow import shortest_augmenting_path
import matplotlib.pyplot as plt
import statistics
from itertools import islice
import numpy as np

graph = {}

E = {}
disjoint = 4


def connectivity(topology):
    for u in graph:
        for v in graph:
            if u != v:
                #check that there are 3 paths less than 130 in the complete graph
                ignore = []
                paths_found = True
                direct=False
                for p in range(0,3):
                    
                    length,path = dijkstras(graph,u,v,ignore,direct)
                    #test with 20ms difference
                    if length > 110:
                        paths_found=False
                        break
                    for node in path:
                        if node != u and node != v:
                            ignore.append(node)
                        if len(path) == 2:
                            direct=True
                            
                #if the complete graph containts 3 disjoint paths verify that the spanner does
                if paths_found:
                    ignore = []
                    direct=False
                    for p in range(0,3):
                        length,path = dijkstras(topology,u,v,ignore,direct)
                        if not path:
                            print("spanner failed for u = " + str(u) + " v = " + str(v))
                            break
                        
                        if length > 130:
                            print("spanner failed for u = " + str(u) + " v = " + str(v))
                            
                            #break
                        for node in path:
                            if node != u and node != v:
                                ignore.append(node)
                            if len(path) == 2:
                                direct=True


#breadth first search with a set of nodes to ignore
def BFS(start, goal,t,ignore,topology):
    explored = []
    queue = [[start]]
    value = {}
    for a in graph: value[int(a)]=0
    while queue:
        path = queue.pop(0)
        node = path[-1]
        
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = topology[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                ##print(graph[node][neighbour])
                ##print(t)
                if value[node] + graph[node][neighbour] <= t and neighbour not in ignore:
                    if not("direct" in ignore and neighbour == goal and node == start):
                        new_path = list(path)
                        new_path.append(neighbour)
                        queue.append(new_path)
                        value[neighbour]= value[node] + graph[node][neighbour]
                         
                        # Condition to check if the
                        # neighbour node is the goal
                        if neighbour == goal:
                            return (new_path, value[neighbour])
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    return None, 0

#runs dijkstras with a set of nodes to ignore                         
def dijkstras(topology, start, second, ignore, direct):
    best = nx.DiGraph()
    for n in topology:
        for m in topology[n]:
            if m not in ignore and n not in ignore:
                if not(n==start and m==second and direct) and not(m==start and n==second and direct):
                    weight = topology[n][m]
                    best.add_edge(n, m, weight=weight)
                    best.add_edge(m, n, weight=weight)
                    
    try:
        length = dijkstra_path_length(G=best, source=start, target=second, weight="weight")
    except nx.NetworkXNoPath:
        return 0,None
    except nx.NodeNotFound:
        return 0,None
    path = dijkstra_path(G=best, source=start, target=second, weight="weight")
    return length,path

def Max_Flow(topology,start,second,ignore,direct):
    best = nx.DiGraph()
    for n in topology:
        for m in topology[n]:
            if m not in ignore and n not in ignore:
                if not(n==start and m==second) and not(m==start and n==second):
                    weight = topology[n][m]
                    best.add_edge(n, m, weight=weight)
                    best.add_edge(m, n, weight=weight)
                    
    try:
        path = list(nx.node_disjoint_paths(best, start, second, flow_func=shortest_augmenting_path,cutoff=disjoint))
    except (nx.NetworkXNoPath, nx.NetworkXError) as e :
        return 0,None
    return path

def Max_weight(path):
    Max = 0
    for i in path:
        result = 0
        for j in range(len(i)-1):
            result += E[(i[j],i[j+1])]
        Max = max(result,Max)
    return Max
    
# k*graph[edge[0]][edge[1]], disjoint, edge[0], edge[1],topology,mode
def LBC(t, disjoint, start, second,topology,mode):
    ignore = []
    direct=False
    #run either bfs and dijkstras a set number of times
    if mode ==2:
        path = Max_Flow(topology,start,second,ignore,direct)
        if len(path) < disjoint or Max_weight(path) > t:    
            if(len(path) >= disjoint):
                print( "add",start, second,path, Max_weight(path),t)
            else:
                print( "add", start, second,path,t)
            return True
        else: 
            print( "dont",start, second,path, Max_weight(path),t)
            return False
    for i in range(disjoint):
        if mode == 0:
            result, value = BFS(start,second,t,ignore,topology)
        elif mode == 1:
            value, result = dijkstras(topology,start,second,ignore,direct)
        print(f"{start} -> {second}  {result,value}")
        if result is None or value > t:
            return True
        else:
            #adding nodes in path to f
            print(f"{start} -> {second} :{value} < {t}")
            #add nodes into the ignore set
            for node in result:
                
                if node != start and node != second:
                    ignore.append(node)
                if len(result) == 2:
                    ignore.append("direct")
                    direct = True
    return False

def greedy(k, disjoint, topology,mode):
    count = 0
    #order the edges in terms of length
    for edge in sorted(E.items(), key=lambda x: x[1]):
        edge=edge[0]
        if edge[0] in topology and edge[1] in topology[edge[0]]:
            continue
        ret = LBC(k*graph[edge[0]][edge[1]], disjoint , edge[0], edge[1],topology,mode)
        
        #if lbc returns true add the edge to topology
        if ret == True:
            
            topology[edge[0]][edge[1]] = graph[edge[0]][edge[1]]
            topology[edge[1]][edge[0]] = graph[edge[1]][edge[0]]
            count = count + 1
    return count

    
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

    
def compute_metrics(topology, k):
    connectivity(topology)
    
    total = 0
    cnt = 0
    for node in topology:
        for edge in topology[node]:
            cnt = cnt + 1
            total = total + topology[node][edge]
    average = total/cnt
    prev = 0
    num = 0
    for node in graph:
        for edge in graph[node]:
            num = num+1
            prev = prev + graph[node][edge]
    avg = prev/num
    print("edge count = " + str(cnt/2))
    print("average hop distance " + str(average))
    print()
    
   
    
def main(argv):
    inputGraph = argv[0]
    posGraph = argv[1]
    kStart = float(argv[2])
    kTries = int(argv[3])
    mode = argv[4]
    ignore = [int(i) for i in argv[5:]]
    print ("--ignore nodes: ",ignore,"--\n")
    

    create_graph(inputGraph,ignore)
    
    pos = read_positions(posGraph,ignore)
    
    if mode == 'b':
        mode = 0
    elif mode == 'd':
        mode = 1
    elif mode == "f":
        mode = 2
    

    k = kStart
    topology = {}
    
    #run on different values of k
    for x in range(0, int(kTries)):
        for a in graph: 
            topology[int(a)]={}

        print("k value = " + str(k))
        # 3 here refers to the number of disjoint paths to find
        greedy(k,disjoint,topology,mode)
        
        
        compute_metrics(topology, k)
        
        
        #pos = read_positions("12node_pos.txt")
        
        
        #draws the graph from position graph
        draw = nx.DiGraph()
        for n in topology:
            for m in topology[n]:
                weight = topology[n][m]
                draw.add_edge(n, m, weight=weight)
                draw.add_edge(m, n, weight=weight)

    

        nx.draw_networkx(draw, pos)
        plt.show()
        

        
        #write the spanner to file
        if mode == 0:
            with open("bfs"+str(round(k, 1))+inputGraph, 'w') as f:
                for u in topology:
                    for v in topology[u]:
                        f.write(str(u) + " " + str(v) + " " + str(topology[u][v]) + "\n")
        elif mode == 1:
            with open("dijk"+str(round(k, 1))+inputGraph, 'w') as f:
                for u in topology:
                    for v in topology[u]:
                        f.write(str(u) + " " + str(v) + " " + str(topology[u][v]) + "\n")
        elif mode == 2:
            with open("flow"+str(round(k, 1))+inputGraph, 'w') as f:
                for u in topology:
                    for v in topology[u]:
                        f.write(str(u) + " " + str(v) + " " + str(topology[u][v]) + "\n")
                    
        
        #update the k value
        k=k+0.1
        k = round(k,1)
if __name__ == "__main__":
    main(sys.argv[1:])

    