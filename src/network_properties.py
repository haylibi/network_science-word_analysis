import math
import time
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
    with open(txt_name, 'w') as f:
        for edge in network.edges:
            f.write(f'{edge[0]} {edge[1]}\n')

def _install_g_trie(directory='./src/gtrieScanner'):
    # Installing gtrieScanner Tool if not already installed (it needs to be installed in a folder named "Ex7")
    if not os.path.isdir(directory):
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
        os.rename('gtrieScanner_src_01', directory.split("/")[-1])



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
                'betweeness_centrality',
                'shortest_path_distribution',
                'communities_number',
                'average_community_size'
            ]
            ,gtrie_directory='./src'
            ,**kwargs   # Kwargs for FIGURE object from matplotlib.pyplot.figure
        ):
        self.network = network
        self.figure = plt.figure(**kwargs)

        # Don't turn figure visible until everything has been plotted
        self.figure.set_visible(False)

        # Defining axes
        # Number of rows -> evaluations dividided by number of cols (ceiling, because if not disible one more row is needed)
        nrows = math.ceil(len(evaluations)/ncols)
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=plt.figure(figsize=(14, 10)))

        # If number of evaluations is divisilbe by number of columns, generate axes needed (very direct, we have a rectangle rows X cols)
        if len(evaluations)%ncols == 0:
            axs = [plt.subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
        # Otherwise, last row will be different
        else:
            # Axis until last row always have number of cols, so creating those
            axs = [plt.subplot(gs[i, j]) for i in range(nrows-1) for j in range(ncols)]
            
            # Last row, it will have a different number of columns depending on wether evaluations is divisible by NCols or not
            # Last axis will fill screen if not divisible 
            for j in range(len(evaluations)%ncols-1):
                axs.append(plt.subplot(gs[nrows-1, j]))
            axs.append(gs[nrows-1, j+1:])

        self.axs = axs
        self.evaluations = evaluations

        # Install G_TrieScanner
        self.gtrie_dir = gtrie_directory



    def betweeness_centrality(self, force_update=False, ax=None, prints=False, plots=False, **kwargs):
        # If method has already been called, no need to calculate values again  (unless forceupdate is true)
        if not hasattr(self, '_betweeness_centrality') or force_update:
            # betweenness centrality
            self._betweeness_centrality = nx.centrality.betweenness_centrality(self.network)
            # betweeness per node
            self.betweeness = (sorted(self._betweeness_centrality.items(), key=lambda item: item[1], reverse=True))

        if prints:
            print("    These are the top 10 nodes with highest betweenness centrality:\n" + '\n'.join([f"{node} -> {centrality_value}" for node, centrality_value in self.betweeness[:10]]))
            print("    These are the 10 nodes with lowest betweenness centrality:\n" + '\n'.join([f"{node} -> {centrality_value}" for node, centrality_value in self.betweeness[-10:]]))
        
        if plots:
            # plot histogram
            if ax is None:
                ax = self.figure.add_subplot()

            ax.hist(self._betweeness_centrality .values(), bins=25)
            # plt.xticks(ticks=[0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001])  # set the x axis ticks
            ax.set_title("betweenness Centrality Histogram ", fontdict={"size": 10}, loc="center")
            ax.set_xlabel("betweenness Centrality", fontdict={"size": 10})
            ax.set_ylabel("Counts", fontdict={"size": 10})

        return {'Betweeness Centrality': np.mean(list(self._betweeness_centrality.values()))}
    
    
    
    def degree_distribution(self, force_update=False, axs=None, plots=False, **kwargs):
        if not hasattr(self, 'degree_frequencies') or force_update:
            # avg_degree
            self.avg_degree = np.mean([d for _, d in self.network.degree()])

            # Degree Distribution
            degree_frequencies = pd.DataFrame.from_dict(dict(self.network.degree), orient='index')
            degree_frequencies['N'] = 0
            degree_frequencies = degree_frequencies.groupby(by=[0]).count().sort_values(by=[0], ascending=True).reset_index()
            
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
                axs = [self.figure.add_subplot(1, 2, 1), self.figure.add_subplot(1, 2, 2)]

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

        # Return Metrics
        return {'PowerLaw Exponent': params[0]-1, 'Average Degree': self.avg_degree}


    def get_shortest_paths(self, force_update=False, **kwargs):
        if not hasattr(self, 'shortest_paths') or force_update:
            self.shortest_paths = dict()
            self.avg_path_length = 0
            self._diameter = 0

            for node, shortest_paths in tqdm.tqdm(nx.all_pairs_shortest_path_length(self.network), desc='    Finding Paths', total=len(self.network.nodes)):
                self.shortest_paths[node] = shortest_paths
                for node2 in shortest_paths.keys():
                    self.avg_path_length += self.shortest_paths[node][node2]
                    self._diameter = max(self._diameter, self.shortest_paths[node][node2])
            
            self.avg_path_length = self.avg_path_length/(len(self.shortest_paths)**2)

        return self.shortest_paths


    def avg_path_length(self, force_update=False, **kwargs):
        if not hasattr(self, 'shortest_paths') or force_update:
            self.get_shortest_paths(force_update=force_update)

        return {'Average Path Length': self.avg_path_length}
        

    def diameter(self, force_update=False, **kwargs):
        # Diameter
        if not hasattr(self, '_diameter') or force_update:
            self.get_shortest_paths(force_update=force_update)    
        return {'Diameter': self._diameter}          


    def shortest_path_distribution(self, force_update=False, ax=None, **kwargs):

        # If diameter already exists, we know the shortest path already exists as well
        if not hasattr(self, 'diameter') or force_update:
            self.avg_path_length(force_update=force_update)      

        # We know the diameter, so create an array
        # to store values from 0 up to (and including) diameter
        path_lengths = np.zeros(self._diameter + 1, dtype=int)

        # Extract the frequency of shortest path lengths between two nodes
        for pls in self.shortest_paths.values():
            pl, cnts = np.unique(list(pls.values()), return_counts=True)
            path_lengths[pl] += cnts
        # Express frequency distribution as a percentage (ignoring path lengths of 0)
        freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()
                
        # Plot distributions
        if ax is None:
            ax = self.figure.add_subplot()

        # Plot the frequency distribution (ignoring path lengths of 0) as a percentage
        ax.bar(np.arange(1, self._diameter + 1), height=freq_percent)
        ax.set_title("Distribution of shortest path length on a random network", fontdict={"size": 10}, loc="center")
        ax.set_xlabel("Shortest Path Length", fontdict={"size": 10})
        ax.set_ylabel("Frequency (%)", fontdict={"size": 10})
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
                ax = self.figure.add_subplot()
                
            # plot histogram
            ax.hist(self._degree_centrality.values(), bins=25)
            ax.set_title("Degree Centrality Histogram ", fontdict={"size": 10}, loc="center")
            ax.set_xlabel("Degree Centrality", fontdict={"size": 10})
            ax.set_ylabel("Counts", fontdict={"size": 10})
        return {'Average Degree Centrality': np.mean(list(self._degree_centrality.values()))}

    
    def clustering_coefficient(self, force_update=False, ax=None, plots=False, **kwargs):
        # If method has already been called, no need to calculate values again  (unless forceupdate is true)
        if not hasattr(self, '_clustering_coefficient') or force_update:
            # betweenness centrality
            self._clustering_coefficient = nx.clustering(self.network)
        
        if plots:
            # plot histogram
            if ax is None:
                ax = self.figure.add_subplot()

            ax.hist(self._clustering_coefficient.values(), bins=25)
            # plt.xticks(ticks=[0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001])  # set the x axis ticks
            ax.set_title("Clustering Coefficient Histogram ", fontdict={"size": 10}, loc="center")
            ax.set_xlabel("Clustering Coefficient", fontdict={"size": 10})
            ax.set_ylabel("Counts", fontdict={"size": 10})

        return {'Average Clustering Coefficient': np.mean(list(self._clustering_coefficient.values()))}




    def z_score(self, motif_size=4, force_update=False, remove_gtrie_dir=False, **kwargs):
        # If method has already been called, no need to calculate values again  (unless forceupdate is true)
        if not hasattr(self, 'z_scores') or force_update:

            # Install G_Trie
            _install_g_trie(directory=self.gtrie_dir)

            _write_network(self.network, '__NETWORK__.txt')
            gtrie_command = f'{self.gtrie_dir}/gtrieScanner -s {motif_size} -m esu -g ./__NETWORK__.txt -f simple -o gtrie_output.txt -raw'

            subprocess.run(
                gtrie_command.split(' ')
                ,stdout=subprocess.DEVNULL  # To not print anything in the console
                ,stderr=subprocess.DEVNULL  # To not print anything in the console
            )
            self.z_scores = pd.read_csv('raw.csv', sep=',').sort_values(by=' z_score', ascending=False)

            # Deleting files/directories created
            os.remove('__NETWORK__.txt')
            os.remove('gtrie_output.txt')

            if remove_gtrie_dir:
                # removing directory
                shutil.rmtree(os.path.abspath(self.gtrie_dir))

        return {f'Most Relevant Subgraph (size={motif_size})': f"'{self.z_scores.iloc[0]['adjmatrix']}' with a Z-Score of {self.z_scores.iloc[0][' z_score']}"}
    
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

    def evaluate(self, plots=False, **kwargs):
        evaluations = {}
        start = time.process_time()
        for evaluation in self.evaluations:
            print(f'  <{time.process_time()-start:05.02f} sec> Executing: <{evaluation}>')
            evaluations.update(eval(f'self.{evaluation}(plots={plots}, **kwargs)'))
        plt.close(self.figure)
        return evaluations
