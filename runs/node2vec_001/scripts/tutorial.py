import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN, OPTICS

def plot_umap(embedding, colors, colormap, type, num_discrete_values, save, output_png_filename):
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111)
    s = 30
    if type == 'discrete':
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, s=s)
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

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

fig = plt.figure()
fig.set_size_inches(10,10)
plt.subplot(111)
nx.draw(graph)
# plt.savefig('{}/{}'.format(sim_dir, 'graph.png'))
plt.show()
plt.close()

dimensions = 64
p = 1 # Return hyperparameter
q = 1 # outward hyperparameter
# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=30, p = p, q = q, num_walks=200, workers=4)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Get node vectors
node_array = np.empty((0, dimensions), dtype = 'f')
for node in graph.nodes:
        node_vector = model.wv[node]
        if node == 0:
            print('node_vector',node_vector)
        node_array = np.append(node_array, node_vector, axis=0)
# UMAP reduction
reducer = umap.UMAP(n_neighbors = 20, min_dist = 0.5)
reducer.fit(node_array)
embedding = reducer.transform(node_array)
save = 'F'
umap_bw_filename = '{}/{}',format(sim_dir, 'umap_bw.png')
plot_umap(embedding, 'none', 'none', 'none', 'F', save, umap_bw_filename)

# cluster nodes in model
cluster_labels = DBSCAN(eps=3, min_samples=5).fit_predict(node_array)
# Plot on umap
num_clusters = np.unique(cluster_labels).shape[0]
umap_cl_filename = '{}/{}',format(sim_dir, 'umap_cl.png')
plot_umap(embedding, colors=cluster_labels, colormap='Spectral', type='discrete', num_discrete_values=num_clusters, save=save, output_png_filename=umap_bw_filename)
# Plot on graph
fig = plt.figure()
fig.set_size_inches(10,10)
plt.subplot(111)
nx.draw(graph, node_color=cluster_labels, cmap=plt.cm.Spectral)
plt.show()
plt.savefig('{}/{}'.format(sim_dir, 'graph_cl.png'))





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
