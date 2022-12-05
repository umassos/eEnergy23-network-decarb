# gasNet.py
# July 2022

# Gets OpenStreetMap data for chosen city/place, converts the data to a NetworkX graph using OSMnx , 
# then performs processing to create a simulated network of underground gas pipelines based on vehicle
# road topology.

# Original Paper Reference:
'''
Adam Lechowicz, Noman Bashir, John Wamburu, Mohammad Hajiesmaili, and Prashant Shenoy
Equitable Network-Aware Decarbonization of Residential Heating at City Scale
ACM International Conference on Future Energy Systems (e-Energy), 2023.
'''

from time import time
from xml.dom import NotFoundErr
import networkx as nx
import itertools
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import osmnx as ox
import osmnx.geometries as gx
import math

from dotenv import load_dotenv
load_dotenv()
import os
mainMapping = os.environ.get("mainMapping")
transFolder = os.environ.get("transformerDataFolder")
gasFolder = os.environ.get("gasDataFolder")

ox.settings.log_console=True
ox.settings.use_cache=True

class GasNet:
    def __init__(self):
        # download street network data from OSM and construct a MultiDiGraph model
        # EXAMPLE: place = {"city": "Amherst", "state": "Massachusetts", "country": "USA"}
        place = {"city": "Amherst", "state": "Massachusetts", "country": "USA"}
        
        cf = '["highway"~"residential|primary|secondary|tertiary|unclassified"]'
        G = ox.graph_from_place(place, network_type="drive", custom_filter=cf)

        # Find closest connection to transmission pipeline (generally the "gate station" for a municipality)
        #   Given a set of coordinates for the gate station, this code will find the closest OSM node, which
        #   will become the "source node" for gas flow in the simulated network.
        # (THESE MUST BE COORDINATES IN UTM FORMAT! use this converter: https://www.latlong.net/lat-long-utm.html)
        connX = 703603.11
        connY = 4695980.74
        self.mapping = {}
        self.G = ox.project_graph(G)
        self.source = ox.nearest_nodes(self.G, connX, connY)

        # get shortest path from source node (gas) to all other nodes:
        paths = nx.single_source_dijkstra_path(self.G, self.source, weight="length")

        # remove extra edges not used by any shortest paths
        keptedges = set()
        for nodePath in paths.values():
            length = len(nodePath)
            for i in range(length-1):
                u = nodePath[i]
                v = nodePath[i+1]
                # mark u,v as kept
                keptedges.add((u,v))

        originalEdges = set(self.G.edges)
        print("original number of edges: {}".format(len(originalEdges)))

        for (u, v, z) in originalEdges:
            if (u,v) in keptedges:
                continue
            else:
                self.G.remove_edge(u,v,z)
        print("simplified number of edges: {}".format(len(self.G.edges)))

        # delete nodes which have been disconnected from the main component
        connectedNodes = max(nx.weakly_connected_components(self.G), key=len)
        originalNodes = set(self.G.nodes)
        for node in originalNodes:
            if node not in connectedNodes:
                self.G.remove_node(node)

        # save a visualization of the original graph generated from OSM data.
        fig, ax = ox.plot_graph(
            self.G, bgcolor="k", node_color="#999999", figsize=(30, 15),  node_size=10, edge_linewidth=2, edge_color="#444444", save=True, filepath="graph.png"
        )

        # given data for gas and electric consumers (meters), assign them to the graph.
        df = pd.read_csv(mainMapping)
        pointsList = []
        for lat, long in zip(df["lat"], df["long"]):
            pointsList.append(Point(long, lat))

        # convert to GeoSeries (enables next step)
        points = gpd.GeoSeries(pointsList, crs='epsg:4326')

        # project to CRS (coordinate reference system) used by the projected graph (this is for accuracy)
        pointsProg = points.to_crs(self.G.graph['crs'])

        # compute nearest edge for each point, corresponding to a gas/electric consumer
        nearest = ox.nearest_edges(self.G, pointsProg.x, pointsProg.y, interpolate=None, return_dist=True)
        nearest_edges = nearest[0]
        distances = nearest[1]

        # initialize edge attributes
        for edge in self.G.edges:
            u,v,e = edge
            self.G.edges[(u,v,e)]['gasids'] = []
            self.G.edges[(u,v,e)]['numgasmeters'] = 0

        # for each edge, assign meter IDs based on the gas/electric consumers which are close to this edge
        #   each gas/electric consumer increases the virtual "length" of this edge, to account for the underground
        #   natural gas service line necessary to tap into the main pipeline.
        for i, edge in enumerate(nearest_edges):
            u,v,e = edge
            point = pointsProg[i]
            if df['gas_meter_id'][i] > 0:
                dist = distances[i]
                self.G.edges[(u,v,e)]['gasids'].append(df['gas_meter_id'][i])
                self.G.edges[(u,v,e)]['numgasmeters'] += 1
                self.G.edges[(u,v,e)]['length'] += dist 

                # for each edge, save the relations between a consumer's gas ID and the distribution transformer they get electric service from.
                if (u,v) not in self.mapping.keys():
                    self.mapping[(u,v)] = []
                self.mapping[(u,v)].append((int(df['gas_meter_id'][i]), df['transformer_id'][i]))
        
        # save a dictionary of total gas usage for each gas meter.
        self.usages = self.getTotalUse()


    # returns the total length of all edges in the network.
    #   will decrease as portions of the network are decommissioned.
    def totalNetworkLength(self):
        total = 0
        for edge in self.G.edges:
            u,v,e = edge
            total += self.G.edges[(u,v,e)]['length']
        return total

    # function that attempts to decommission one segment (one edge) of the simulated gas network.
    #   if deleting this segment disconnects some other segments from the source node, the process continues
    #   to delete those newly disconnected segments until the remaining network is one connected component
    #   containing the source node.
    def pruneEdge(self, u,v,e, save=False):
        # check that the graph has one connected component before pruning anything.
        if not nx.is_weakly_connected(self.G):
            print("ERROR: graph not connected before pruning edges")
            return 0,0
        copyG = self.G.copy()
        lengthPruned = 0.0  # stores the total length of edges that have been decommissioned
        gasPruned = []  # stores the ID of gas meters where gas service has been shut off.
        elecDict = {}   # elecDict stores the transformer IDs serving consumers where gas service has been shut off.

        # try deleting the edge:
        lengthPruned += copyG.edges[(u,v,e)]['length']
        gasPruned.extend(copyG.edges[(u,v,e)]['gasids'])
        copyG.remove_edge(u,v)
        if (u,v) in self.mapping.keys():
            for gasid, elecid in self.mapping[(u,v)]:
                if elecid not in elecDict.keys():
                    elecDict[elecid] = []
                elecDict[elecid].append(gasid)
        if copyG.degree[u] == 0:
            copyG.remove_node(u)
        if copyG.degree[v] == 0:
            copyG.remove_node(v)

        # check how many connected components there are now
        if nx.is_weakly_connected(copyG):
            # if 1 component, the process is successful. assign new graph and return lists
            if save == True:
                self.G = copyG
            return (gasPruned, lengthPruned), elecDict
        else:
            # if not, we have to take the edges in the component not including the source and decommision them as well 
            # (we cannot add edges)
            smallestCC = min(nx.weakly_connected_components(copyG), key=len)
            if self.source in smallestCC:
                smallestCC = max(nx.weakly_connected_components(copyG), key=len)
            
            # for each node that has newly been disconnected from the source node, delete all incident edges.
            for node in smallestCC:
                outEdges = set(copyG.out_edges(node))
                inEdges = set(copyG.in_edges(node))
                for edge in itertools.chain(outEdges, inEdges):
                    i,j = edge
                    lengthPruned += copyG.edges[(i,j,e)]['length']
                    gasPruned.extend(copyG.edges[(i,j,e)]['gasids'])
                    copyG.remove_edge(i,j)
                    if (i,j) in self.mapping.keys():
                        for gasid, elecid in self.mapping[(i,j)]:
                            if elecid not in elecDict.keys():
                                elecDict[elecid] = []
                            elecDict[elecid].append(gasid)
                copyG.remove_node(node)
            
            # check to make sure everything is connected now:
            if nx.is_weakly_connected(copyG):
                # if 1 component, the process is successful. assign new graph and return lists
                if save == True:
                    self.G = copyG
                return (gasPruned, lengthPruned), elecDict
            else:
                print("ERROR: pruning unsuccessful")

    # computes the total gas usage for each gas meter, only considering non-summer (non-heating) months.
    def getTotalUse(self):
        print("Calculating gas load for non-summer months, each gas meter")
        usages = {}
        dfr = pd.read_csv(mainMapping)

        # for each gas meter, sum over the non-summer gas usage and assign to "usages" dictionary.
        for gasID in list(dfr['gas_meter_id']):
            if math.isnan(gasID):
                continue
            try:
                # read df with usages for this specific gas meter
                df = pd.read_csv(gasFolder+str(int(gasID))+'.csv', usecols=["datetime", "power"], parse_dates=["datetime"])
                
                # compute season
                df["season"] = df["datetime"].apply(lambda x: "summer" if x.month in [6,7,8] else "other")
                otherTotal = df[df["season"] == "other"]["power"].sum()
                
                usages[int(gasID)] = otherTotal
            except FileNotFoundError:
                print("this should not happen")
        
        return usages
        