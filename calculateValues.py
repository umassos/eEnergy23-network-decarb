# calculateValues.py
# July 2022

# Initializes GasNet and ElecNet objects for the given city, and tries deleting (decommissioning)
# each edge in the simulated gas network.  Each edge acts as an "index" for a potential neighborhood
# to be converted to ASHP.  Each neighborhood has a cost (conversion cost) and a value (potential CO2
# reduction).  These results are written to a CSV file, indexed by the initial deleted edge.

# Original Paper Reference:
'''
Adam Lechowicz, Noman Bashir, John Wamburu, Mohammad Hajiesmaili, and Prashant Shenoy
Equitable Network-Aware Decarbonization of Residential Heating at City Scale
ACM International Conference on Future Energy Systems (e-Energy), 2023.
'''

import gasNet
import elecNet
import pandas as pd
import networkx as nx
from multiprocessing import Pool, Manager
import statistics

from dotenv import load_dotenv
load_dotenv()
import os
mainMapping = os.environ.get("mainMapping")
transFolder = os.environ.get("transformerDataFolder")
gasFolder = os.environ.get("gasDataFolder")
heatPumpCost = os.environ.get("heatPumpCost")
sourceNode = os.environ.get("sourceNode")

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

# computes the cost of converting houses from gas heating
# to ASHP, scaled based on median gas usage and median ASHP
# installation cost.
def getConversionCost(gn, gasPruned):
    usages = gn.usages
    median = statistics.median(gn.usages.values())
    totalPumps = 0
    for gasID in gasPruned: # for each converted house
        div = usages[int(gasID)] / median
        totalPumps += div
    return heatPumpCost * totalPumps

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

# initialize gasNet and elecNet
gn = gasNet.GasNet()
en = elecNet.ElecNet()

nodeList = list(gn.G.nodes)
manager = Manager()
data = manager.list()
edgeList = list(gn.G.edges)

# get baseline results for # of overloaded transformers and total length
# of natural gas network.
orig_overloaded, orig_overload_percentage = en.overloadingFactor(en.baseLoad)
original_length = gn.totalNetworkLength()

# set maintenance cost metric for full network
# (i.e. yearly maintenance cost for full city natural gas network is $9,000,000)
maintenanceCost = 9000000 

# try deleting (decommissioning) one segment of the gas network, and save
# the results.  If other edges must be decommissioned due to this first edge,
# include the results from those edges here as well.
# returns cost of ASHP conversion, transformer upgrade cost, maintenance savings, and carbon reduced.
def oneEdge(edge):
    u,v,e = edge
    if u == sourceNode: # if u is the source node (gate station), skip this edge
        return
    print("Pruning Edge ({}, {})...".format(u,v))

    # use gasNet function to attempt deleting this edge
    (gasPruned, lengthPruned), elecDict = gn.pruneEdge(u, v, e)

    # add load previously covered by natural gas to electric network
    newLoad, savedCCF = en.addLegacyGasLoad(elecDict)

    # get cost of ASHP conversions
    conversionCost = getConversionCost(gn, gasPruned)

    # compute estimated savings due to shutting down parts of network
    newLength = original_length - lengthPruned
    y = (newLength * maintenanceCost) / (original_length)
    savings = maintenanceCost - y

    # compute which transformers are newly overloaded, and upgrade cost
    new_overloaded, percentage = en.overloadingFactor(newLoad)
    addedOverload = len(new_overloaded.keys()) - len(orig_overloaded.keys())
    upgradedTrans = {}
    for key in new_overloaded.keys():
        if key not in orig_overloaded.keys():
            upgradedTrans[key] = new_overloaded[key]
    overloadCost = getOverloadCost(en, upgradedTrans)

    # convert CCF gas usage into corresponding carbon reduction
    carbonReduced = carbonReduction(savedCCF)[0]

    totalCost = conversionCost + overloadCost - savings

    # "utility" is the cost-to-benefit ratio
    utility = carbonReduced/totalCost

    data.append({"edge_u": u, "edge_v": v, "conversion_cost": conversionCost, "maintenance_savings": savings, "newly_overloaded_transformers": addedOverload, "overloaded_transformers_cost": overloadCost, "total_cost": totalCost, "carbon_reduced": carbonReduced, "utility": utility, "gas_meters_pruned": gasPruned, "upgraded_transformers": list(upgradedTrans)})

# split into 6 child processes to speed up the process -- each process tests the outcome
# of decommissioning one edge at a time.
with Pool(6) as p:
    p.map(oneEdge, edgeList)
    p.close()
    p.join()

dataList = list(data)

# write results to csv file.
df = pd.DataFrame(dataList)
df = df.sort_values(by=['utility'], ascending=False)
df= df[df['total_cost'] > 0]
df.to_csv('results.csv', index=False, encoding='utf-8')
