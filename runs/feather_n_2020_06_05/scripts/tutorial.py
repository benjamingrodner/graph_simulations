import networkx as nx
import numpy as np
from karateclub import DeepWalk
from karateclub.node_embedding.neighbourhood import Walklets
from karateclub import GraphReader
from karateclub import Diff2Vec
from karateclub import FeatherGraph
from karateclub import FeatherNode
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import coo_matrix

####################################################################################
# Functions
####################################################################################
def plot_umap(embedding, type='none', colors='', colormap='', save='F', output_png_filename=''):
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.add_subplot(111)
    s = 30
    if type == 'discrete':
        discrete_values = np.unique(colors)
        num_discrete_values = discrete_values.shape[0]
        if discrete_values[0] == 0:
            discrete_values = discrete_values + 1
            colors = np.array(colors) + 1
        im = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=colormap, s=s, edgecolors = 'gray')
        plt.gca().set_aspect('equal', 'datalim')
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
        boundaries = np.arange(1, num_discrete_values + 2) - 0.5
        cbar = fig.colorbar(im, cax=cax, ticks=discrete_values, boundaries = boundaries, orientation='vertical')
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


def cluster_nodes(node_array, n_clusters):
    n_clusters = 6
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(node_array)
    cluster_names = np.unique(cluster_labels)
    cluster_labels = cluster_labels - cluster_names[0] + 1
    return(cluster_names, cluster_labels)


def draw_graph(graph, node_class = 'F', colormap = '', with_labels = False, size_scalar=1, save='F', output_png_filename=''):
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    fig.set_size_inches(20,10)
    fig.sca(ax)
    pos = nx.spring_layout(graph,k = 1.0, iterations=150)
    degree = dict(nx.degree(graph))
    node_size = [v*20*size_scalar + 60*size_scalar for v in degree.values()]
    # node_size = [v*20 + 60 for v in degree.values()]
    if node_class == 'F':
        nodes = nx.draw_networkx_nodes(graph, pos=pos, node_size = node_size)
        nodes.set_edgecolor('gray')
        nx.draw_networkx_edges(graph, pos=pos, edge_color='gray')
    else:
        discrete_values = np.unique(node_class)
        num_discrete_values = discrete_values.shape[0]
        if discrete_values[0] == 0:
            discrete_values = discrete_values + 1
            node_class = np.array(node_class) + 1
        nodes = nx.draw_networkx_nodes(graph, pos=pos, node_size = node_size, node_color=node_class, cmap=colormap)
        nodes.set_edgecolor('gray')
        nx.draw_networkx_edges(graph, pos=pos, edge_color='gray')
        axpos = ax.get_position().bounds
        cax = fig.add_axes([axpos[0] + axpos[2] + 0.02, axpos[1], 0.02, axpos[3]])
        boundaries = np.arange(1, num_discrete_values + 2) - 0.5
        cbar = fig.colorbar(nodes, cax=cax, ticks=node_class, boundaries = boundaries, orientation='vertical')
    if save == 'T':
        plt.savefig(output_png_filename)
    else:
        plt.show()
    plt.close()
    return



####################################################################################
# Main
####################################################################################
#
# DeepWalk
g = nx.newman_watts_strogatz_graph(100, 20, 0.05)
draw_graph(g)
model = DeepWalk()
model.fit(g)
embedding = model.get_embedding()
#
# Walklets
model = Walklets()
model.fit(g)
embedding = model.get_embedding()
#
# Community Detection with LabelPropagation
# Get graph
reader = GraphReader("facebook")
graph = reader.get_graph()
draw_graph(graph, 'F','F','')
target = reader.get_target()
# Fit
model = LabelPropagation()
model.fit(graph)
cluster_membership = model.get_memberships()
# Evaluate
cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]
draw_graph(graph, cluster_membership, 'F','')
nmi = normalized_mutual_info_score(target, cluster_membership)
'{:.4f}'.format(nmi)
#
# Node embedding with FeatherNode
# Get graph
reader = GraphReader("wikipedia")
graph = reader.get_graph()
nodes_list = list(graph.nodes())
len(nodes_list)
edges_list = list(graph.edges)
len(edges_list)
# nodes_list_sorted = nodes_list.copy()
# nodes_list_sorted.sort()
# Get features
features = reader.get_features()
fs = features.shape[1]
fs
# get true classification
y = reader.get_target()
# Subset the graph
edges_list_sub = edges_list[:1000]
nodes_sub = np.unique([i for tup in edges_list_sub for i in tup])
graph_sub = graph.subgraph(nodes_sub)
index_remapping = dict(zip(nodes_sub,list(range(nodes_sub.shape[0]))))
graph_sub = nx.relabel_nodes(graph_sub, index_remapping)
features_sub_list = []
y_sub = []
for i in nodes_sub:
    features_sub_list.append(features.getrow(i).toarray().reshape(fs,))
    y_sub.append(y[i])

features_sub_array = np.array(features_sub_list)
features_sub = coo_matrix(features_sub_array)
features_sub_array = []
features_sub_list = []
# Show subgraph
draw_graph(graph_sub, y_sub, 'gist_ncar', size_scalar=0.5)
# fit
reduction_dimensions = 64
model = FeatherNode(reduction_dimensions)
model.fit(graph_sub, features_sub)
X = model.get_embedding()
# UMAP reduction
reducer = umap.UMAP(n_neighbors = 10, min_dist = 0.5)
reducer.fit(X)
embedding = reducer.transform(X)
plot_umap(embedding, 'discrete', y_sub, 'Spectral')
# Cluster
n_clusters = 8
cluster_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
cluster_names = np.unique(cluster_labels)
cluster_labels = cluster_labels + 1-cluster_names[0]
plot_umap(embedding, 'discrete', cluster_labels, 'gist_ncar')
draw_graph(graph_sub, cluster_labels, 'gist_ncar')
# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
'{:.4f}'.format(auc)
#
# Graph embedding with FeatherGraph
reader = GraphSetReader("wikipedia")
graphs = reader.get_graphs()
y = reader.get_target()
# Fit
model = FeatherGraph()
model.fit(graphs)
X = model.get_embedding()
# evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
'AUC: {:.4f}'.format(auc)
