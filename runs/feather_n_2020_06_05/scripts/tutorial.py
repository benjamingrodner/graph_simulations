import networkx as nx
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

####################################################################################
# Functions
####################################################################################
def plot_umap(embedding, save='F', output_png_filename='', type='none', colors='', colormap='',  num_discrete_values=1):
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


def cluster_nodes(node_array, n_clusters):
    n_clusters = 6
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(node_array)
    cluster_names = np.unique(cluster_labels)
    cluster_labels = cluster_labels - cluster_names[0] + 1
    return(cluster_names, cluster_labels)


def draw_graph(graph, cluster_labels = 'F', save='F', output_png_filename=''):
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
    if save == 'T':
        plt.savefig(output_png_filename)
    else:
        plt.show()
        plt.close()
    return
    fig.set_size_inches(20,10)
    fig.sca(ax)
    pos = nx.spring_layout(graph,k = 1.0, iterations=300)
    degree = dict(nx.degree(graph))
    node_size = [v*20 + 60 for v in degree.values()]
    # nx.draw(graph, pos=nx.kamada_kawai_layout(graph), node_size = 50)
    if cluster_labels == 'F':
        nodes = nx.draw_networkx_nodes(graph, pos=pos, node_size = node_size, with_labels=True, cmap=plt.cm.gist_ncar)
        pos = ax.get_position().bounds
        cax = fig.add_axes([pos[0] + pos[2] + 0.02, pos[1], 0.02, pos[3]])
        cluster_names = np.unique(cluster_labels)
        num_clusters = cluster_names.shape[0]
        boundaries = np.arange(1, num_clusters + 2) - 0.5
        cbar = fig.colorbar(nodes, cax=cax, ticks=cluster_labels, boundaries = boundaries, orientation='vertical')
    else:
        nodes = nx.draw_networkx_nodes(graph, pos=pos, node_size = node_size, node_color=cluster_labels, with_labels=True, cmap=plt.cm.gist_ncar)
    nodes.set_edgecolor('gray')
    nx.draw_networkx_edges(graph, pos=pos, edge_color='gray')
    plt.savefig('{}/{}'.format(sim_dir, 'graph_q_0.5.png'))
    # plt.show()
    plt.close()



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
reader = GraphReader("twitch")
graph = reader.get_graph()
draw_graph(graph, 'F','F','')
y = reader.get_target()
# fit
model = FeatherNode()
model.fit(graph)
X = model.get_embedding()
plot_umap(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
'{:.4f}'.format(auc)
#
# Graph embedding with FeatherGraph
reader = GraphSetReader("reddit10k")
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
