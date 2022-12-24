def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}  # dist from init node to nth node
    parents = {}
    g[start_node] = 0  # g[S] =0
    parents[start_node] = start_node  # parent = {'S':'S'}
    while len(open_set) > 0:
        n = None
        for v in open_set:
            # T or F =T
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v  # v = 1)S 2)A 3)B
        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m, weight) in get_neighbors(n):  # get the neighbors of n m= A ,B, C weight = 1, 2, 13
                if m not in open_set and m not in closed_set:  # T coz m not a source or a dest
                    open_set.add(m)
                    parents[m] = n  # s is the parent of 1) A 2)B
                    g[m] = g[n] + weight  # 1){Á}:1 2){B}
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n == None:
            print('Path does not exist!')
            return None
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path
        open_set.remove(n)  # remove S from open_set and add it to closed_set
        closed_set.add(n)
    print('Path does not exist!')
    return None


def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]  # return the neighbors of
    else:
        return None


def heuristic(n):
    H_dist = {
        'S': 5,
        'A': 4,
        'B': 5,
        'E': 0,
    }
    return H_dist[n]


Graph_nodes = {
    'S': [('A', 1), ('B', 2)],
    'A': [('E', 13), ],
    'B': [('E', 5)]
}
aStarAlgo('S', 'E')
