# 常见的规划类算法
## 1 Search-based Planning Methods 基于搜索的  
   基本思想是将状态空间通过确定的方式离散成一个图，然后利用各种启发式搜索算法搜索可行解甚至是最优解，该种类别算法比较成熟。  
   基于搜索的算法的基础是状态格子，状态格子是状态空间离散化，由状态结点和从该结点出发到达相邻结点的运动基元组成，一个状态结点可以通过其运动基元变换到另一个状态结点。状态格子就将原来连续的状态空间转化为一个搜索图，运动规划问题就变成了在图中搜索出一系列将初始状态变换到目标状态运动基元，构建起状态格子后就可以使用图搜索算法来搜索最优轨迹。
### 1.1 BFS和DFS————最原始的，最暴力的
DFS path: 根据预先定义好的方向向深了走，每次走到边界才去转换方向，这样无法保证距离最短。（纯**苯方法**，不展开）  
BFS path: 从起点开始根据层优先的方式，当它搜到终点后，根据距离值做个回溯，就可以得到一个最短路径。它只有把相同的层节点都探索过之后，才能得出结果。<br>
按照理解，要有效搜索最好还是用BFS以及基于BFS的一系列方法，都算是基于队列的。

### 1.2 Dijkstra算法————最短路径
Dijkstra算法也算是老朋友。它是一种贪心算法，每次都选择最短的路径，直到找到终点。它的缺点是无法处理**负权边**，因为它是基于正权边的。<br>
它相比BFS算法的区别就是它维护了一个新的变量g(n)，表征当前节点n距离起点的累积代价。有了这个信息后，每次弹出的时候就知道了弹哪一个能保证整体距离代价最小。所以我们把队列换成优先级队列Priority queue，它的区别在于队列中是按照优先级排序的（小值优先队列/大值优先队列) <br>

优点：最优性的保障，只要运行完一次Dijkstra Algorithm之后，只要算法遍历过的节点，就可以得到从起点到任意点的最短路径的。

缺点：整个过程没有很强的导向性，不知道任何终点信息，所以整个的搜索是盲目的。(也很好理解，因为本质就是BFS中的队列被换成了优先队列g(n), 可用A*算法解决盲目性)


Algorithm Dijkstra (G, start) : 

    •let open_list be priority queue           //首先定义一个优先级队列open_list
    •open_list .push(start, 0)                 //把起点start加入到open_list，同时维护每个节点的g[n]
    •g[start] := 0

    •while (open_list is not empty) : 
        •current := open_list.pop()            //从当前的优先级队列中弹出一个元素，current
        •mark current as visited
        •if current is the goal :              //如果current是终点，那么我们就认为找到了一条最短路径，接下来做回溯就可以了
            •return current
        •for all unvisited neighbours next of current in Graph G :   //如果不是终点，那么我们找出这个节点的邻居节点next
            •next_cost := g[current] + cost(current, next)
            •if next is not in open_list:                            //如果没有被访问过，那么我们就把它加入到列表中
                •open_list.push(next, next_cost)
            •else: 
                •if g[next] > next_cost:                             //如果之前被访问过，我们对比之前的那条路径和当前新发现的路径哪条是更小的，我们把更小cost的这条路径给存储下来。
                •g[next] := next_cost

Dijkstra的代码参考python robotic写在[这里](/rpa_code/dijkstra.py)。