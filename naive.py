# naive.py
# August 2022

# Initializes a ''NaiveNet'' and ElecNet object for the given city, and attempts to decommission gas
# service to the highest gas consuming houses, subject to a budget constraint.  Each house has a 
# conversion cost and a value (potential CO2 reduction).  Once houses are chosen, the heating load
# previously supplied by gas is shifted to the grid, potentially incurring upgrade costs.

# Original Paper Reference:
'''
Adam Lechowicz, Noman Bashir, John Wamburu, Mohammad Hajiesmaili, and Prashant Shenoy
Equitable Network-Aware Decarbonization of Residential Heating at City Scale
ACM International Conference on Future Energy Systems (e-Energy), 2023.
'''

from time import time
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import pickle
import math
import elecNet
from sklearn.utils import shuffle

from dotenv import load_dotenv
load_dotenv()
import os
mainMapping = os.environ.get("mainMapping")
transFolder = os.environ.get("transformerDataFolder")
gasFolder = os.environ.get("gasDataFolder")
heatPumpCost = os.environ.get("heatPumpCost")

ox.settings.log_console=True
ox.settings.use_cache=True

# SET BUDGET FOR EXPERIMENT HERE (IN USD)
CONVERSION_BUDGET = 180000000

# set maintenance cost metric for full network
# (i.e. yearly maintenance cost for full city natural gas network is $9,000,000)
maintenanceCost = 9000000 

# filename for the image with mapped results
outputFilename = "graphNaiveDecommissioned.png"

# NaiveNet is a variant of GasNet which only allows to decommission houses one by one, not by neighborhood.
class NaiveNet:
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

        # get shortest path from source (gas) to all other nodes:
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
        
    # returns the total length of all edges in the network.
    #   will decrease as portions of the network are decommissioned.
    def totalNetworkLength(self):
        total = 0
        for edge in self.G.edges:
            u,v,e = edge
            total += self.G.edges[(u,v,e)]['length']
        return total

    # function that sequentially disconnects a list of houses from gas service.
    #   once houses are disconnected, this function deletes any edges (segments of gas pipeline)
    #   which no longer serve any meters.
    def deleteHouses(self, list):

        for edge in self.G.edges:
            u,v,e = edge
            for id in self.G.edges[(u,v,e)]['gasids']:
                if id in list: # delete any houses identified in the list of decommisioned houses.
                    self.G.edges[(u,v,e)]['gasids'].remove(id)
        
        unnecessary = []
        for edge in self.G.edges:
            u,v,e = edge
            # if this edge is not serving any houses, mark it as "unnecessary"
            if len(self.G.edges[(u,v,e)]['gasids']) < 1:
                unnecessary.append((u,v,e))
        
        # remove all unnecessary edges as long as they don't disconnect the graph
        copyG = self.G.copy()
        for (u,v,e) in unnecessary:
            copycopyG = copyG.copy()
            copycopyG.remove_edge(u,v)
            if nx.is_weakly_connected(copycopyG):
                copyG = copycopyG
        
        return copyG, unnecessary

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
        
nn = NaiveNet()
en = elecNet.ElecNet()
incomeMapping = pickle.load( open( "incomeMapping.pickle", "rb" ) )

# returns (kg, tons) of CO2 not emitted thanks to transition
# ccf represents natural gas usage in cubic feet.
def carbonReduction(ccf):
    return 5.51*ccf, 0.00551*ccf  

# gets the index of the next highest item in a list, given 
# a value that we need to accommodate
def getNextIndex(myList, load):
    ind = myList.index(min(myList, key=lambda x:abs(x-load)))
    if myList[ind] < load:
        ind += 1
    return ind

# computes the cost of converting one house from gas heating
# to ASHP, scaled based on median gas usage and median ASHP
# installation cost.
def getConversionCost(usage):
    median = 7800.0
    return heatPumpCost * (usage / median)

# computes the cost of upgrading newly overloaded transformers
# due to ASHP installations
def getOverloadCost(en, upgradedTrans):
    upgradedtIDs = upgradedTrans.keys()
    capacities = [15, 25, 37.5, 50, 75, 100, 150, 167, 225]
    cost = 0.0
    for tID in upgradedtIDs: # for each transformer to be upgraded
        existingkVA = en.transformers[tID]['kVA']
        peakLoad = upgradedTrans[tID]
        requiredkVA = peakLoad/0.95 # compute the absolute minimum required kVA rating

        if requiredkVA < existingkVA*2 and existingkVA < 75: # next-nearest pole-top upgrade
            base = 10225.0
            delta = 21300.0
            delScale = (getNextIndex(capacities, requiredkVA)/4.0)
            cost += (base + delScale*delta)
        elif requiredkVA > existingkVA*3 and existingkVA < 75: #pole-top to pad-mount conversion
            base = 79050.0
            delta = 74900.0
            delScale = (getNextIndex(capacities, requiredkVA)-4)/4.0
            cost += (base + delScale*delta)
        elif requiredkVA > existingkVA*2 and existingkVA < 75: #additional pole-top
            base = 10225.0
            delta = 21300.0
            addkVA = requiredkVA - existingkVA
            delScale = (getNextIndex(capacities, addkVA)/4.0)
            cost += (base + delScale*delta)
        elif requiredkVA > existingkVA*2 and existingkVA > 75: #additional pad-mount
            base = 74900.0
            delta = 74900.0
            addkVA = requiredkVA - existingkVA
            delScale = (getNextIndex(capacities, addkVA)-4)/4.0
            cost += (base + delScale*delta)
    return cost

dfr = pd.read_csv(mainMapping)

budget = CONVERSION_BUDGET * (2/3)

# get gas usage of each house
usages = nn.getTotalUse()

# get conversion cost for each house
conversionCost = [getConversionCost(usage) for usage in usages]

# save parameters for evaluation
original_overload, orig_percent = en.overloadingFactor(en.baseLoad)
metersPruned = []
costs = 0.0
saved = 0.0

# for each house, save the gas usage and the conversion cost.
dfr['GasUsage'] = usages
dfr['ConversionCost'] = conversionCost
dfr['Utility'] = np.array(usages) / np.array(conversionCost)

# sort values by gas usage, seeking to convert biggest gas consumers
dfr = dfr.sort_values(by=['GasUsage'], ascending=False)

print("Converting Houses...")

# add houses to a list "to be converted" until there is no more budget left.
for meter, cost, ccf in zip(dfr['gas_meter_id'], dfr['ConversionCost'], dfr['GasUsage']):
    if math.isnan(meter):
        continue
    metersPruned.append(meter)
    costs += cost
    saved += ccf
    if costs > budget:
        break

# for each gas meter ID which will be converted, save the corresponding transformer ID in gasDict.
gasDict = {}
for tID, gasID in zip(dfr['transformer_id'], dfr['gas_meter_id']):
    if math.isnan(gasID):
        continue
    if int(gasID) in metersPruned:
        if tID not in gasDict.keys():
            gasDict[tID] = []
        gasDict[tID].append(int(gasID))

# add energy load previously taken by gas to the corresponding transformer
newLoad, saved = en.addLegacyGasLoad(gasDict)

# compute number of newly overloaded transformers and get the cost
newOverload, percentage = en.overloadingFactor(newLoad)
upgradedTrans = {}
for key in newOverload.keys():
    if key not in original_overload.keys():
        upgradedTrans[key] = newOverload[key]
transCost = getOverloadCost(en, upgradedTrans)

costs += transCost

# print results to console
print("Total Cost: {}".format(costs))
print("Total Gas Saved: {}".format(saved))
print("Carbon Reduction: {}".format(5.51*saved))

# delete houses from underlying gas network
print("Deleting Houses...")
newG, unnEdges = nn.deleteHouses(metersPruned)

oldLength = nn.totalNetworkLength()

print("Number of Houses Converted: {}".format(len(metersPruned)))
print("Overloaded Transformers: {}".format(len(upgradedTrans.keys())))

newLength = 0
for edge in newG.edges:
    u,v,e = edge
    newLength += newG.edges[(u,v,e)]['length']

print("Length Deleted: {}".format(oldLength - newLength))

# based on the OSM graph, plot the decommissioned portions of the gas network.
# deleted edges are in RED, and extant edges are GRAY.
print("Plotting...")

# choose color based on whether the edge exists in the experimental graph.
colorDict = {}
edgeList = list(nn.G.edges)
for u,v,e in nn.G.edges:
    if not newG.has_edge(u,v,e):
        colorDict[(u,v,e)] = "#FF0001"
    else:
        colorDict[(u,v,e)] = "#444444"

ser = pd.Series(data=colorDict, index=edgeList)

# here we make a new node for each house disconnected from gas (just for visualization's sake)
pointsList = []
groups = []
df = pd.read_csv(mainMapping)
incomeMapping = pickle.load( open( "incomeMapping.pickle", "rb" ) )

# classify each house based on income tract (generated elsewhere)
for meterID, lat, long in zip(df["gas_meter_id"], df["lat"], df["long"]):
    if meterID in metersPruned:
        pointsList.append(Point(long, lat))
        groups.append(incomeMapping[meterID])

# convert to GeoSeries (enables next step)
points = gpd.GeoSeries(pointsList, crs='epsg:4326')

# project to CRS (coordinate reference system) used by the projected graph (this is for accuracy)
pointsProg = points.to_crs(nn.G.graph['crs'])
        
# plots the main map graph of the gas network
fig, ax = ox.plot_graph(
    nn.G, bgcolor="w", node_color="#444444", figsize=(30, 15), node_size=5, edge_linewidth=2, edge_color=ser, show=False, save=False, close=False
)

# adds "points" representing each house's income tract & location to the map graph plotted above.
houseAlpha = 0.5
houseSize = 25
for i, (point, group) in enumerate(zip(pointsProg, groups)):
    if group == 1:
        ax.scatter(point.x, point.y, c="#0173B2", s = houseSize, alpha = houseAlpha)
    elif group == 2:
        ax.scatter(point.x, point.y, c="#029E73", s = houseSize, alpha = houseAlpha)
    elif group == 3:
        ax.scatter(point.x, point.y, c="#D55E00", s = houseSize, alpha = houseAlpha)
    else:
        ax.scatter(point.x, point.y, c="#222222", s = houseSize, alpha = houseAlpha)

extent = ax.bbox.transformed(fig.dpi_scale_trans.inverted())
plt.draw()
plt.savefig("naive.png", bbox_inches=extent, dpi=200)