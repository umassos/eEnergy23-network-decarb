# elecNet.py
# July 2022

# Given a mapping of gas/electric consumers, encapsulates a list of distribution transformers and information about them,
# including capacity in kVA and kWh, load profiles over the entire experimental period, and allows to compute quantities
# such as the number of overloaded transformers and portion of time spent overloaded.  Also handles the conversion of
# a natural gas heating load into an equivalent electric heat pump load for the same heat energy in BTU.

# Original Paper Reference:
'''
Adam Lechowicz, Noman Bashir, John Wamburu, Mohammad Hajiesmaili, and Prashant Shenoy
Equitable Network-Aware Decarbonization of Residential Heating at City Scale
ACM International Conference on Future Energy Systems (e-Energy), 2023.
'''

from time import time
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()
import os
mainMapping = os.environ.get("mainMapping")
transFolder = os.environ.get("transformerDataFolder")
gasFolder = os.environ.get("gasDataFolder")

class ElecNet:
    def __init__(self):
        self.transformers = {}
        # create transformer objects based on the mapping to different meters:
        df = pd.read_csv(mainMapping)
        df_trans = df['transformer_id']

        # for each transformer, save information about it (rating, which meters it is connected to)
        for i, tID in enumerate(df_trans):
            if tID not in self.transformers.keys():
                self.transformers[tID] = {'kVA': df['transformer_kVA'][i], 'meters': []}
            self.transformers[tID]['meters'].append(df['device_id'][i])
        
        self.baseLoad = self.getCompleteTransformerLoad()

    # encapsulates the total electric load on each distribution transformer in the data.
    #   each transformer is represented by a list giving the load at five minute intervals throughout the experiment's time period
    def getCompleteTransformerLoad(self):
        networkloads = {}

        # for each transformer
        for tID in self.transformers.keys():
            try:
                # load file for this specific transformer
                load = pd.read_csv(transFolder+str(tID)+'.csv')
                self.transformers[tID]['kW'] = load['transformer_kW'][0]
                networkloads[tID] = list(load['power'])
            except FileNotFoundError:
                print("this should not happen -- trans ID: {}".format(tID))
        return networkloads
    
    # based on a list of loads, computes how many and how often the distribution transformers in the network are overloaded
    def overloadingFactor(self, networkloads):
        overloadedtIDs = {}
        totalTime = 0.0
        overloadedTime = 0.0
        load = pd.read_csv(os.listdir(transFolder)[0])
        timesteps = len(load['power'])
        
        # for each transformer
        for tID in self.transformers.keys():
            try:
                capacity = self.transformers[tID]['kW']
                load = networkloads[tID]
                # if load is greater than 125% of capacity, we consider that time slot to be overloaded.
                overloadingEvents = len([x for x in load if x > (1.25 * capacity)])  
                totalTime += timesteps
                overloadedTime += overloadingEvents

                if overloadingEvents > 0:
                    overloadedtIDs[tID] = max(load)     # save transformer to dict if it is overloaded at any time
            except FileNotFoundError:
                print("this should not happen -- trans ID: {}".format(tID))

        return overloadedtIDs, (overloadedTime/totalTime)

    # save a new load profile for distribution transformers in the network
    def setNewLoad(self, newLoad):
        self.baseLoad = newLoad
        return
    
    # given a list of transformers and corresponding gas meters which have been decommissioned from gas service,
    # add the corresponding heating-related load to the correct transformer serving that house, representing the
    # increase of electrical load due to electric heat pump adoption.
    def addLegacyGasLoad(self, elecDict, save=False):
        newLoad = self.baseLoad.copy()
        ccfSaved = 0
        load = pd.read_csv(os.listdir(transFolder)[0])

        # for each impacted transformer
        for tID in elecDict.keys():
            # for each downstream electric consumer with gas service that has been shut off
            for gasID in elecDict[tID]:
                try:
                    i = 0
                    # read data file for this consumer's gas usage
                    df = pd.read_csv(gasFolder+str(int(gasID))+'.csv', usecols=["datetime", "power"], parse_dates=["datetime"])
                    
                    # compute carbon savings due to not using natural gas
                    ccfSaved += df["power"].sum()

                    # compute season
                    df["season"] = df["datetime"].apply(lambda x: "summer" if x.month in [6,7,8] else "other")
                    
                    # Compute summer average
                    summerAvg = df[df["season"] == "summer"]["power"].mean()
                    # Drop season column
                    df = df.drop(columns=["season"])
                    # Rename column to represent natural gas usage
                    df = df.rename(columns={"power": "ccf"})
                    # Compute heating load (remove usage from other gas appliances)
                    df["heating_ccf"] = df["ccf"] - summerAvg
                    # Remove any negatives
                    df["heating_ccf"] = df["heating_ccf"].apply(lambda x: max([x,0]))

                    currentList = newLoad[tID]
                    length = len(currentList)
                    newList = []

                    # compute the corresponding kWh needed to heat the same amount.
                    for gasUsed in df['heating_ccf']:
                        # compute BTU based on efficiency (average) of furnace and conversion
                        gasBTU = gasUsed * 0.875 * 103700

                        # compute kWh based on BTU
                        gaskWh = gasBTU * 0.000293071

                        # compute electric kWh (8.5 HSPF) (this is where COP comes into play)
                        ehpkWh = gaskWh / 2.5

                        nextHour = i + 12
                        while i < nextHour and i < length:
                            newList.append(currentList[i] + ehpkWh)
                            i += 1
                    newLoad[tID] = newList
                except FileNotFoundError:
                    print("this should not happen")
        if save == True:
            self.baseLoad = newLoad
        return newLoad, ccfSaved
