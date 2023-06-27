import math
import time
import random
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit

# Packages used for G-Trie
import subprocess
import requests
import zipfile
import shutil
import tqdm
import os

from io import BytesIO # Used to temporarily store the ZIP downloaded file


from IPython.display import display, HTML

# Remove display after code execution for matplotlib
plt.ioff()


################################################################################################
# 
#                                       Auxiliar functions for G-Trie
#
################################################################################################
def _write_network(network, txt_name):
    '''
        Used for calculating Z-Scores
        Writes network in a txt file where every line is a edge (Node1 Node2)
       
    '''
    node_mapping = {}   # Converting every node into a number
    edges = set()
    with open(txt_name, 'w') as f:
        for edge in network.edges:
            if edge in edges: continue
            edges.add(edge)
            edges.add((edge[1], edge[0]))
            node_mapping.setdefault(edge[0], len(node_mapping)+1)
            node_mapping.setdefault(edge[1], len(node_mapping))
            f.write(f'{node_mapping[edge[0]]} {node_mapping[edge[1]]}\n')

def _install_g_trie(directory='./src/gtrieScanner'):
    # Installing gtrieScanner Tool if not already installed (it needs to be installed in a folder named "Ex7")
    if not os.path.isdir(directory):
        print(directory)
        print(os.getcwd())
        # URL to the GTrieScanner tool (zip file)
        gtrieScanner_url = 'https://www.dcc.fc.up.pt/~pribeiro/aulas/ns2223/homework/gtrieScanner_src_01.zip'

        # Download and extract gtrieScanner to folder
        response = requests.get(gtrieScanner_url)
        with BytesIO(response.content) as f:
            gtrie_zip = zipfile.ZipFile(f)
            gtrie_zip.extractall('/'.join(directory.split('/')[:-1]))
            gtrie_zip.close()
        response.close()

        # Installing gtrieScanner_src_01
        prev_dir = os.getcwd()
        os.chdir(f'{"/".join(directory.split("/")[:-1])}/gtrieScanner_src_01')
        subprocess.run(['make'])
        os.chdir(prev_dir)
        os.rename(f'{"/".join(directory.split("/")[:-1])}/gtrieScanner_src_01', directory)


# Defining function which will create a image for a given adjacency matrix
def save_graph_with_labels(adjacency_matrix, file_name=None, width='100px', padding_left='0', margin_left='0'):
    if file_name is None:
        file_name = adjacency_matrix
    # If network already exists, jurt return HTML text
    if f'{file_name}.png' in os.listdir('./data/imgs/'): 
        return f'<img src="./data/imgs/{file_name}.png" style="width:{width}; padding-left:{padding_left}; margin-left:{margin_left}">'
    num_rows = int(np.sqrt(len(adjacency_matrix)))
    adj_matrix = np.array([[int(x) for x in adjacency_matrix[i*num_rows: (i+1)*num_rows]] for i in range(len(adjacency_matrix)//num_rows)])
    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=3000, with_labels=False, width=1, node_color='#000000')
    # Save fig to be loaded by HTML afterwards
    plt.savefig(f'./data/imgs/{file_name}.png', format='png')
    plt.clf()
    return f'<img src="./data/imgs/{file_name}.png" style="width:{width}; padding-left:{padding_left}">'


################################################################################################
# 
#                                 Class to evaluate networks
#
################################################################################################

class EvaluateNetworks():
    def __init__(
            self
            ,network: nx.Graph
            ,ncols=5
            ,evaluations=[
                'degree_distribution',
                'avg_path_length',
                'diameter',
                'clustering_coefficient',
                'z_score',    # USE G-TRIES or whatever it is called, don't have it currently
                'degree_centrality',
                'betweenness_centrality',
                'shortest_path_distribution',
                'communities_number',
                'average_community_size'
            ]
            ,gtrie_directory='./src/gtrieScanner'
            ,**kwargs   # Kwargs for FIGURE object from matplotlib.pyplot.figure
        ):
        self.network = network
 
        self.evaluations = evaluations

        # Install G_TrieScanner
        self.gtrie_dir = gtrie_directory

        # Initiate Motif_size to learn if motif already has been executed
        self.motif_size = -1



    def betweenness_centrality(self, force_update=False, ax=None, prints=False, plots=False, betweenness_k=None, **kwargs):
        # If method has already been called, no need to calculate values again  (unless forceupdate is true)
        if not hasattr(self, '_betweenness_centrality') or force_update:
            # betweenness centrality
            self._betweenness_centrality = nx.centrality.betweenness_centrality(self.network, k=betweenness_k)
            # betweeness per node
            self.betweeness = (sorted(self._betweenness_centrality.items(), key=lambda item: item[1], reverse=True))

        if prints:
            print("    These are the top 10 nodes with highest betweenness centrality:\n" + '\n'.join([f"{node} -> {centrality_value}" for node, centrality_value in self.betweeness[:10]]))
            print("    These are the 10 nodes with lowest betweenness centrality:\n" + '\n'.join([f"{node} -> {centrality_value}" for node, centrality_value in self.betweeness[-10:]]))
        
        if plots:
            # plot histogram
            if ax is None:
                fig, ax = plt.subplots(1, 1)

            ax.hist(self._betweenness_centrality .values(), bins=25)
            # plt.xticks(ticks=[0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001])  # set the x axis ticks
            ax.set_title("betweenness Centrality Histogram ", fontdict={"size": 10}, loc="center")
            ax.set_xlabel("betweenness Centrality", fontdict={"size": 10})
            ax.set_ylabel("Counts", fontdict={"size": 10})

            display(ax.get_figure())
            plt.close(ax.get_figure())

        return {'Betweeness Centrality': np.mean(list(self._betweenness_centrality.values()))}
    
    
    
    def degree_distribution(self, force_update=False, axs=None, plots=False, **kwargs):
        if not hasattr(self, 'degree_frequencies') or force_update:
            # avg_degree
            self.avg_degree = np.mean([d for _, d in self.network.degree()])

            # Degree Distribution
            degree_frequencies = pd.DataFrame.from_dict(dict(self.network.degree), orient='index')
            degree_frequencies['N'] = 0
            degree_frequencies = degree_frequencies.groupby(by=[0]).count().sort_values(by=[0], ascending=True).reset_index()
            
            # If there is not enough data to create a regression, return empty
            if degree_frequencies.shape[0] < 3: return {'PowerLaw Exponent': 'NA', 'Average Degree': self.avg_degree}
            
            # Calculate power law estimated
            degree_frequencies['N_Cumulative'] = degree_frequencies['N'][::-1].cumsum()
            
            # Define the power law function
            def power_law(x, a, b):
                return a * x + b
            
            # Fit the power law using curve_fit
            params, _ = curve_fit(power_law, np.log(degree_frequencies[0]), np.log(degree_frequencies['N_Cumulative']))
            degree_frequencies['Expected'] = degree_frequencies.apply(lambda row: np.exp(power_law(np.log(row[0]), *params)), axis=1)

            self.degree_frequencies = degree_frequencies

        if plots:
            # Plot distributions
            if axs is None:
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

            axs[1].grid(True, which="both", linestyle='--', alpha=0.5)
            axs[1].tick_params(axis='both', which='major', labelsize=10)
            axs[1].tick_params(axis='both', which='minor', labelsize=8)


            # Normal scale
            degree_frequencies.plot(title='Degree Distribution', x=0, y='N', ax=axs[0], xlabel=None)


            degree_frequencies.plot(
                title='Degree Distribution (log-log scale)'
                ,ax=axs[1]
                ,loglog=True
                ,markersize=6
                ,markeredgecolor='steelblue'
                ,x=0
                ,y=['N', 'N_Cumulative', 'Expected']
                ,label=['Data', 'Cumulative', 'Power Law Cumulative']
                ,style=['bo', 'c.', 'r--']
        )

            display(axs[0].get_figure())
            plt.close(axs[0].get_figure())

        # Return Metrics
        return {'PowerLaw Exponent': params[0]-1, 'Average Degree': self.avg_degree}


    def get_shortest_paths(self, force_update=False, print_tqdm=True, **kwargs):
        if not hasattr(self, 'shortest_paths') or force_update:
            self.shortest_paths = dict()
            self._avg_path_length = 0
            self._diameter = 0

            _iter = nx.all_pairs_shortest_path_length(self.network)
            if print_tqdm: _iter = tqdm.tqdm(_iter, desc='    Finding Paths', total=len(self.network.nodes))
            for node, shortest_paths in _iter:
                self.shortest_paths[node] = shortest_paths
                for node2 in shortest_paths.keys():
                    self._avg_path_length += self.shortest_paths[node][node2]
                    self._diameter = max(self._diameter, self.shortest_paths[node][node2])
            
            self._avg_path_length = self._avg_path_length/(len(self.shortest_paths)**2)

        return self.shortest_paths


    def avg_path_length(self, force_update=False, **kwargs):
        if not hasattr(self, 'shortest_paths') or force_update:
            self.get_shortest_paths(force_update=force_update, **kwargs)

        return {'Average Path Length': self._avg_path_length}
        

    def diameter(self, force_update=False, **kwargs):
        # Diameter
        if not hasattr(self, '_diameter') or force_update:
            self.get_shortest_paths(force_update=force_update, **kwargs)    
        return {'Diameter': self._diameter}          


    def shortest_path_distribution(self, force_update=False, ax=None, plots=True, **kwargs):

        if not plots: return {}
        
        # If diameter already exists, we know the shortest path already exists as well
        if not hasattr(self, 'diameter') or force_update:
            self.avg_path_length(force_update=force_update)      

        # We know the diameter, so create an array
        # to store values from 0 up to (and including) diameter
        path_lengths = np.zeros(self._diameter + 1, dtype='int64')

        # Extract the frequency of shortest path lengths between two nodes
        for pls in self.shortest_paths.values():
            pl, cnts = np.unique(list(pls.values()), return_counts=True)
            path_lengths[pl] += cnts
        # Express frequency distribution as a percentage (ignoring path lengths of 0)
        freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()
                
        # Plot distributions
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # Plot the frequency distribution (ignoring path lengths of 0) as a percentage
        ax.bar(np.arange(1, self._diameter + 1), height=freq_percent)
        ax.set_title("Distribution of shortest path length on a random network", fontdict={"size": 10}, loc="center")
        ax.set_xlabel("Shortest Path Length", fontdict={"size": 10})
        ax.set_ylabel("Frequency (%)", fontdict={"size": 10})

        display(fig)
        plt.close(fig)
        return {}


    def degree_centrality(self, force_update=False, ax=None, prints=False, plots=False, **kwargs):

        if not hasattr(self, '_degree_centrality') or force_update:
            # degree centrality
            self._degree_centrality = nx.centrality.degree_centrality(self.network)            
            # degree centrality per node
            self.centrality = (sorted(self._degree_centrality.items(), key=lambda item: item[1], reverse=True))

        if prints:
            print("    These are the top 10 nodes with highest degree centrality:\n" + '\n'.join([f"{node} -> {centrality_value}" for node, centrality_value in self.centrality[:10]]))
            print("    These are the 10 nodes with lowest degree centrality:\n" + '\n'.join([f"{node} -> {centrality_value}" for node, centrality_value in self.centrality[-10:]]))

        if plots:
            # plot histogram
            if ax is None:
                fig, ax = plt.subplots(1, 1)
                
            # plot histogram
            ax.hist(self._degree_centrality.values(), bins=25)
            ax.set_title("Degree Centrality Histogram ", fontdict={"size": 10}, loc="center")
            ax.set_xlabel("Degree Centrality", fontdict={"size": 10})
            ax.set_ylabel("Counts", fontdict={"size": 10})
            
            display(ax.get_figure())
            plt.close(ax.get_figure())

        return {'Average Degree Centrality': np.mean(list(self._degree_centrality.values()))}

    
    def clustering_coefficient(self, force_update=False, ax=None, plots=False, **kwargs):
        # If method has already been called, no need to calculate values again  (unless forceupdate is true)
        if not hasattr(self, '_clustering_coefficient') or force_update:
            # betweenness centrality
            self._clustering_coefficient = nx.clustering(self.network)
        
        if plots:
            # plot histogram
            if ax is None:
                fig, ax = plt.subplots(1, 1)

            ax.hist(self._clustering_coefficient.values(), bins=25)
            # plt.xticks(ticks=[0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001])  # set the x axis ticks
            ax.set_title("Clustering Coefficient Histogram ", fontdict={"size": 10}, loc="center")
            ax.set_xlabel("Clustering Coefficient", fontdict={"size": 10})
            ax.set_ylabel("Counts", fontdict={"size": 10})
            
            display(ax.get_figure())
            plt.close(ax.get_figure())

        return {'Average Clustering Coefficient': np.mean(list(self._clustering_coefficient.values()))}




    def z_score(self, motif_size=4, force_update=False, remove_gtrie_dir=False, z_score_k=None, debug=False, **kwargs):
        # If method has already been called, no need to calculate values again  (unless forceupdate is true)
        if self.motif_size!=motif_size or force_update:
            self.motif_size = motif_size
            # Install G_Trie
            _install_g_trie(directory=self.gtrie_dir)

            # If there is a Sampling number, pick those number of edges
            if z_score_k is not None: 
                random_edges = random.sample(list(self.network.edges), z_score_k)
                G = nx.Graph()
                G.add_edges_from(random_edges)
                _write_network(G, '__NETWORK__.txt')

            else: _write_network(self.network, '__NETWORK__.txt')
            gtrie_command = f'{self.gtrie_dir}/gtrieScanner -s {motif_size} -m esu -g ./__NETWORK__.txt -f simple -o gtrie_output.txt -raw'

            if debug: print(gtrie_command)
            subprocess.run(
                gtrie_command.split(' ')
                ,stdout=subprocess.DEVNULL  # To not print anything in the console
                ,stderr=subprocess.DEVNULL  # To not print anything in the console
            )
            self.z_scores = pd.read_csv(
                'raw.txt'
                ,sep=','
                ,dtype={
                    'adjmatrix': 'str'
                    ,' occ_original': 'int'
                    ,' z_score': 'float'
                    ,' avg_random': 'float'
                    ,' stdev_random': 'float'
                }
            ).sort_values(by=' z_score', ascending=False)
            self.z_scores['motif'] = self.z_scores['adjmatrix'].apply(save_graph_with_labels)

            # Deleting files/directories created
            os.remove('__NETWORK__.txt')
            os.remove('gtrie_output.txt')
            os.remove('raw.txt')

            if remove_gtrie_dir:
                # removing directory
                shutil.rmtree(os.path.abspath(self.gtrie_dir))

        return {f'Most Relevant Subgraph (size={motif_size})': f"{self.z_scores.iloc[0]['motif']} with a Z-Score of {self.z_scores.iloc[0][' z_score']}"}
    
    def communities_number(self, force_update=False, **kwargs):
        if not hasattr(self, '_communities_number') or force_update:
            communities = nx.community.louvain_communities(self.network, seed=1234)
            number_of_communities = len(communities)
            self._communities_number = number_of_communities
        return {'Number of Communities': self._communities_number}           

    def average_community_size(self, force_update=False, **kwargs):
        if not hasattr(self, '_average_community_size') or force_update:
            communities = nx.community.louvain_communities(self.network, seed=1234)
            average_size = sum(len(community) for community in communities) / len(communities)
            self._average_community_size = average_size
        return {'Average Community Size': self._average_community_size}


    def motif_frequency(self, motif_size=4, **kwargs):
        if self.motif_size != motif_size:
            self.z_score(motif_size=motif_size, **kwargs)

        df = self.z_scores[['motif', ' occ_original']].copy().rename(columns={' occ_original': 'frequency'})
        display(HTML(df.to_html(index=False).replace('&lt;', '<').replace('&gt;', '>')))

        return {}
        


    def evaluate(self, plots=False, print_evals=True, warnings=True, **kwargs):
        evaluations = {}
        start = time.process_time()
        for evaluation in self.evaluations:
            try:
                if print_evals: print(f'  <{time.process_time()-start:05.02f} sec> Calculating: <{evaluation}>')
                evaluations.update(eval(f'self.{evaluation}(plots={plots}, **kwargs)'))
            except Exception as e:
                if warnings: print(f'  <{time.process_time()-start:05.02f} sec> Error evaluatinog <{evaluation}>. Message: {str(e)}')
        return evaluations
