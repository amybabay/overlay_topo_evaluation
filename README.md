# overlay_topo_evaluation

## Files :

1. get topology
```
python spanner.py 12node_edges.txt 12node_pos.txt initialKValue iterations mode ignoreNodes(optional)
```
- To generate overlay network topologys output file such as: dijk1.712node_edges.txt

- the spanner.py file is edited based on https://github.com/kjt32/SpannerGenerator/spanner.py



2. evaluate topology
```
python diff.py 12node_edges.txt 12node_pos.txt topologyOutPut.txt ignoreNodes(optional) 
```
- To print evaluation based on flexibility, Fault-tolerance, cost, and latency of the giving topology( used '-> topologyName.txt' to store the info, such as bfs.txt).  

- To see more details of each path, enable :
   ```
   #print(path_sum)           <-- def add_path

   #print(E[(i[j],i[j+1])])   <-- def add_cal_weight
   ```
    should work



3. the summary.txt included a summary of evaluations of conf topology, bfs topology, dijk topology, and max flow topology.



## Instructions of summary.txt:
```
Data summary for MIN weight disjoint path: Min : -25% , Average : 2%, Max : 59%

Data summary for MAX weight disjoint path: Min : 2% , Average : 152%, Max : 2150%

Data summary for MAX 6th weight disjoint path: Min : 4% , Average : 514%, Max : 3180%
```
- the summary of MIN weight is like in the situation of no fault, the evaluation of min, average, and max of all the min weight(shortest) path of each node to node.

- the "MAX weight" indecate 3ed best path or 3ed lowest weight path. 

- 'Average : 2%' the average of the difference between each min weight path and their direct path

