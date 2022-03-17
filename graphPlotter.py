from mayavi import mlab

def plot_graph(g,signal,xyz):
    mlab.points3d(xyz[0],xyz[1],xyz[2],scale_factor=0.6)
    for i,edge in enumerate(g.edge_index.T):
        print(i)
        mlab.plot3d([xyz[0, edge[0]], xyz[0,edge[1]]],
                    [xyz[1, edge[0]], xyz[1,edge[1]]],
                    [xyz[2, edge[0]], xyz[2,edge[1]]])
        if i>500:
            break
    mlab.show()