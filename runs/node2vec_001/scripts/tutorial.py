import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN, KMeans
import csv
import pandas as pd

def plot_umap(embedding, colors, colormap, type, num_discrete_values, save, output_png_filename):
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111)
    s = 30
    if type == 'discrete':
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, s=s, edgecolors = 'gray')
        plt.gca().set_aspect('equal', 'datalim')
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
        cbar = fig.colorbar(im, cax=cax, ticks=colors, boundaries = np.arange(1, num_discrete_values + 2) - 0.5, orientation='vertical')
    elif type == 'theta':
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, vmin=-pi/2, vmax= pi/2, cmap=colormap, s=s)
        # im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, clim = (-pi/2,pi/2), s=s)
        # im = ax.imshow(seg, cmap=colors, clim = (-pi/2,pi/2))
        plt.gca().set_aspect('equal', 'datalim')
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
        # cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar = fig.colorbar(im, cax=cax, ticks=[-pi/2, 0, pi/2], orientation='vertical')
        cbar.ax.set_yticklabels(['-pi/2', '0', 'pi/2'])
    elif type == 'continuous':
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, s=s)
        # im = ax.imshow(seg, cmap=colors, clim = (-pi/2,pi/2))
        plt.gca().set_aspect('equal', 'datalim')
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    elif type == 'none':
        im = ax.scatter(embedding[:, 0], embedding[:, 1], s=s)
        plt.gca().set_aspect('equal', 'datalim')
    ax.set_xlabel('UMAP 1', fontsize=24)
    ax.set_ylabel('UMAP 2', fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])
    if save == 'T':
        plt.savefig(output_png_filename, bbox_inches = 'tight', transparent = True)
    else:
        plt.show()
    plt.close()

# Set up
sim_dir = '/workdir/bmg224/hiprfish/graph_simulations/runs/node2vec_001/simulation/tutorial'
data_dir = '/workdir/bmg224/hiprfish/graph_simulations/runs/node2vec_001/data'

# Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
# graph = nx.erdos_renyi_graph(100, 0.15)
# graph = nx.watts_strogatz_graph(100, 3, 0.1)
# graph = nx.barabasi_albert_graph(100, 3)
# graph = nx.random_lobster(100, 0.9, 0.9)
# graph = nx.karate_club_graph()
# graph = nx.les_miserables_graph()
les_miserables_graph_filename = '{}/{}'.format(data_dir, 'out.moreno_lesmis_lesmis')

graph = nx.Graph()
file = pd.read_csv(les_miserables_graph_filename, delim_whitespace = True, comment = '%', names = ['u','v','weight'])
for index, row in file.iterrows():
    u,v,w = row[['u','v','weight']]
    # graph.add_edge(u,v)
    graph.add_edge(u,v,weight=w)

fig = plt.figure()
fig.set_size_inches(20,10)
plt.subplot(111)
pos = nx.spring_layout(graph,k = 1.0, iterations=300)
degree = dict(nx.degree(graph))
node_size = [v*20 + 60 for v in degree.values()]
# nx.draw(graph, pos=nx.kamada_kawai_layout(graph), node_size = 50)
nodes = nx.draw_networkx_nodes(graph, pos=pos, node_size = node_size)
nodes.set_edgecolor('gray')
nx.draw_networkx_edges(graph, pos=pos, node_size=node_size, edge_color='gray')
# nx.draw(graph, pos=nx.spring_layout(graph), node_size = 50)
# plt.savefig('{}/{}'.format(sim_dir, 'graph.png'))
plt.show()
plt.close()

k_s = np.linspace(0.001,1,6)
iterations = np.arange(1,302,50)
for i in iterations:
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(20,10)
    fig.suptitle('iterations=' + str(i))
    for ax, k in zip(axes.flatten(), k_s):
        plt.sca(ax)
        pos = nx.spring_layout(graph, k = k, iterations=i)
        nodes = nx.draw_networkx_nodes(graph, pos=pos)
        nodes.set_edgecolor('gray')
        nx.draw_networkx_edges(graph, pos=pos, edge_color='gray')
        ax.set_title('k=' + str(k))
    plt.show()
    plt.close()

dimensions = 16
p = 1 # Return hyperparameter
q = 0.5 # outward hyperparameter
num_walks = 20
walk_length = 10
# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length , p = p, q = q, num_walks=num_walks, workers=4)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Get node vectors
node_array = np.empty((0, dimensions), dtype = 'f')
for node in graph.nodes:
        node_vector = model.wv[str(node)][np.newaxis]
        if node == 0:
            print('node_vector.shape',node_vector.shape)
            print('node_array.shape',node_array.shape)
        node_array = np.append(node_array, node_vector, axis=0)


# cluster nodes in model
n_clusters = 6
cluster_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(node_array)
cluster_names = np.unique(cluster_labels)
cluster_labels = cluster_labels + 1-cluster_names[0]

# Plot on graph
degree = dict(nx.degree(graph))
node_sie = [v*20 + 60 for v in degree.values()]
fig, ax = plt.subplots(1,1)
fig.set_size_inches(20,10)
fig.sca(ax)
pos = nx.spring_layout(graph,k = 1.0, iterations=300)
degree = dict(nx.degree(graph))
node_size = [v*20 + 60 for v in degree.values()]
# nx.draw(graph, pos=nx.kamada_kawai_layout(graph), node_size = 50)
nodes = nx.draw_networkx_nodes(graph, pos=pos, node_size = node_size, node_color=cluster_labels, with_labels=True, cmap=plt.cm.gist_ncar)
nodes.set_edgecolor('gray')
nx.draw_networkx_edges(graph, pos=pos, edge_color='gray')
pos = ax.get_position().bounds
cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
num_clusters = cluster_names.shape[0]
boundaries = np.arange(1, num_clusters + 2) - 0.5
cbar = fig.colorbar(nodes, cax=cax, ticks=cluster_labels, boundaries = boundaries, orientation='vertical')
plt.savefig('{}/{}'.format(sim_dir, 'graph_q_0.5.png'))
# plt.show()
plt.close()

# UMAP reduction
reducer = umap.UMAP(n_neighbors = 10, min_dist = 0.5)
reducer.fit(node_array)
embedding = reducer.transform(node_array)

# Plot on umap
save = 'T'
umap_cl_filename = '{}/{}'.format(sim_dir, 'umap_p_0.5.png')
plot_umap(embedding, colors=cluster_labels, colormap='gist_ncar', type='discrete', num_discrete_values=num_clusters, save=save, output_png_filename=umap_cl_filename)






# Save embeddings for later use
model.wv.save_word2vec_format('/workdir/bmg224/hiprfish/graph_simulations/runs/node2vec_001/simulation/tutorial/embedding')

# Save model for later use
model.save('/workdir/bmg224/hiprfish/graph_simulations/runs/node2vec_001/simulation/tutorial/model')



# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder

edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()

# Look for most similar edges - this time tuples must be sorted and as str
edges_kv.most_similar(str(('1', '2')))

# Save embeddings for later use
edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)
