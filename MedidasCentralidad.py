import igraph as igraph
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scipy

df = pd.read_csv("data/preprocessedTrainReduced.csv", encoding="utf-8")

graph = nx.DiGraph()
graph.add_nodes_from(set(list(df["idA"])+list(df["idB"])))
print(len(list(df["class"])))
for i in range(len(list(df["class"]))):
    graph.add_edge(df["idA"][i],df["idB"][i], weight=df["class"][i])

# tuples = [tuple(x) for x in df.values]
# graph= igraph.Graph.TupleList(tuples, directed = True, edge_attrs = ['class'])

print(graph)
plt.figure(figsize=(15, 15))
# nx.draw(graph, with_labels=True)
# plt.savefig("Graph.png", format="PNG")


deg_centrality = nx.degree_centrality(graph)
print("deg_centrality")

print(deg_centrality)

in_deg_centrality = nx.in_degree_centrality(graph)
print("in_deg_centrality")
print(in_deg_centrality)

out_deg_centrality = nx.out_degree_centrality(graph)
print("out_deg_centrality")
print(out_deg_centrality)

eigenvector_centrality=nx.eigenvector_centrality(graph)
print("eigenvector_centrality")
print(eigenvector_centrality)

between_centrality=nx.betweenness_centrality(graph)
print("between_centrality")
print(between_centrality)

# information_centrality=nx.information_centrality(graph)
# print("information_centrality")
# print(information_centrality)

# number_cliques=nx.number_of_cliques(graph)
# print("number_cliques")
# print(number_cliques)

constraint=nx.constraint(graph)
print("constraint")
print(constraint)

effective_size=nx.effective_size(graph)
print("effective_size")
print(effective_size)

clustering_coefficient=nx.average_clustering(graph)
print("clustering_coefficient")
print(clustering_coefficient)

hubs_centrality,authority_centrality = nx.hits(graph)
print("hubs_centrality")
print(hubs_centrality)
print("authority_centrality")
print(authority_centrality)

A_in_deg_centrality=[]
B_in_deg_centrality=[]

A_out_deg_centrality=[]
B_out_deg_centrality=[]

A_eigen_centrality=[]
B_eigen_centrality=[]

A_between_centrality=[]
B_between_centrality=[]

A_constraint=[]
B_constraint=[]

A_effective=[]
B_effective=[]

A_hubs_centrality=[]
B_hubs_centrality=[]

A_author_centrality=[]
B_author_centrality=[]
for idA,idB,weight in graph.edges(data=True):
    A_in_deg_centrality.append(in_deg_centrality[idA])
    B_in_deg_centrality.append(in_deg_centrality[idB])

    A_out_deg_centrality.append(out_deg_centrality[idA])
    B_out_deg_centrality.append(out_deg_centrality[idB])

    A_eigen_centrality.append(eigenvector_centrality[idA])
    B_eigen_centrality.append(eigenvector_centrality[idB])

    A_between_centrality.append(between_centrality[idA])
    B_between_centrality.append(between_centrality[idB])

    A_constraint.append(constraint[idA])
    B_constraint.append(constraint[idB])

    A_effective.append(effective_size[idA])
    B_effective.append(effective_size[idB])

    A_hubs_centrality.append(hubs_centrality[idA])
    B_hubs_centrality.append(hubs_centrality[idB])

    A_author_centrality.append(authority_centrality[idA])
    B_author_centrality.append(authority_centrality[idB])



print(len(A_in_deg_centrality))
print(len(B_in_deg_centrality))

print(len(A_out_deg_centrality))
print(len(B_out_deg_centrality))

print(len(A_eigen_centrality))
print(len(B_out_deg_centrality))

print(len(A_between_centrality))
print(len(B_between_centrality))

print(len(A_constraint))
print(len(B_constraint))

print(len(A_effective))
print(len(B_effective))

print(len(A_hubs_centrality))
print(len(B_hubs_centrality))

print(len(A_author_centrality))
print(len(B_author_centrality))

print(len(list(df["class"])))

preprocesseData = {"A_in_deg_centrality": A_in_deg_centrality,"B_in_deg_centrality": B_in_deg_centrality,
                   "A_out_deg_centrality": A_out_deg_centrality,"B_out_deg_centrality": B_out_deg_centrality,
                   "A_eigen_centrality": A_eigen_centrality,"B_eigen_centrality": B_eigen_centrality,
                   "A_between_centrality": A_between_centrality,"B_between_centrality": B_between_centrality,
                   "A_constraint_centrality": A_out_deg_centrality,"B_constraint_centrality": B_out_deg_centrality,
                   "A_effective_net_size": A_effective,"B_effective_net_size": B_effective,
                   "A_hubs_centrality": A_hubs_centrality,"B_hubs_centrality": B_hubs_centrality,
                   "A_author_centrality": A_author_centrality, "B_author_centrality": B_author_centrality,"class":list(df["class"])}
# print(len(list(in_deg_centrality.values())))
# print(len(list(out_deg_centrality.values())))
# print(len(list(eigenvector_centrality.values())))
# print(len(list(between_centrality.values())))
# print(len(list(constraint.values())))
# print(len(list(effective_size.values())))
# print(len(list(authority_centrality.values())))
# print(len(list(hubs_centrality.values())))

df = pd.DataFrame(preprocesseData)

df.to_csv(path_or_buf="data/caracteristicsTrainReduced.csv", sep=',', encoding="utf-8", index=False)


