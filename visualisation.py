# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:40:38 2025

@author: dekoning
"""

### Script that visualizes the results
# import packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import statistics
import scipy
import matplotlib_latex_bridge as mlb

# figures to latex style
mlb.setup_page(**mlb.formats.article_letterpaper_10pt_singlecolumn)
mlb.figure_columnwidth() 
mlb.set_font_sizes(18)

# Human experiments data
# hider, searcher, iteration, search strategy, hider strategy, result (0 drone captured, 1 hider found), location of result
with open('Experiment1.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    humandata = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location']) 
with open('Experiment2.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location']) 
    df = df.drop(df[df['Searcher'] == 'Human'].index)  # Remove participant1 vs participant 3 data
    df = df.drop(df[df['Hider'] == 'Human'].index)  # Remove participant1 vs participant 3 data
    humandata = pd.concat([humandata, df])
with open('Experiment3.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location']) 
    humandata = pd.concat([humandata, df])
with open('Experiment4.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location']) 
    humandata = pd.concat([humandata, df])
with open('Experiment5.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location']) 
    humandata = pd.concat([humandata, df])


### data without human experiments
# hider, searcher, iteration, search strategy, hider strategy, result (0 drone captured, 1 hider found), location of result, Ps, Ph, V
with open('ExperimentnotHuman1.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    nothumandata = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location', 'Ps', 'Ph', 'V']) 
with open('ExperimentnotHuman2.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location', 'Ps', 'Ph', 'V']) 
    nothumandata = pd.concat([nothumandata, df])
with open('ExperimentnotHuman3.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location', 'Ps', 'Ph', 'V']) 
    nothumandata = pd.concat([nothumandata, df])
with open('ExperimentnotHuman4.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location', 'Ps', 'Ph', 'V']) 
    nothumandata = pd.concat([nothumandata, df])
with open('ExperimentnotHuman5.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location', 'Ps', 'Ph', 'V']) 
    nothumandata = pd.concat([nothumandata, df])
    
    
### Robotic Simulator data
with open('ExperimentSim.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    simdata = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location']) 
with open('ExperimentSimPart3.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    df = pd.DataFrame(mynewlist, columns = ['Hider', 'Searcher', 'Iteration', 'Search strat', 'Hide strat', 'Result', 'Location']) 
    simdata = pd.concat([simdata, df])

simdata.loc[simdata['Hider'].str.contains('Participant'), 'Hider'] = 'Human'  # convert participants to human
simdata.loc[simdata['Searcher'].str.contains('Participant'), 'Searcher'] = 'Human'  # convert participants to human
simdata.loc[simdata['Hider'].str.contains('robot'), 'Hider'] = 'belief-based'  
simdata.loc[simdata['Searcher'].str.contains('robot'), 'Searcher'] = 'belief-based' 

####################
# optimal strategies:
opth = [0.545454,0.181818,0.272727]
opts1 = [1/6]*6 #-> learning 
opts2 = [0.272727, 0.272727, 0.1363635, 0.1363635, 0.090909, 0.090909] #-> paper
opts3 = [0.54585,0.09061,0,0,0,0.36354] #-> payoff matrix solver
opts4 = [0.61843,0,0,0,0.05432,0.32725] #-> payoff matrix solver
opts = [opts1, opts2, opts3, opts4]

# preprocess data 
bigdata = pd.concat([nothumandata, humandata])
bigdata.loc[bigdata['Searcher'].str.contains('Participant'), 'Searcher'] = 'Human'  # convert participants to human
bigdata.loc[bigdata['Hider'].str.contains('Participant'), 'Hider'] = 'Human'  # convert participants to human

# table, distri plots, KL-divergences
kldivs = []
kldivh = []
avg_hiders = []
uncertainty = []

loopdata = bigdata  # data from experiments. replace with simdata for simulator data plots
for hider in loopdata['Hider'].unique():
    for searcher in loopdata['Searcher'].unique():
        print([hider, searcher])
        data = loopdata[(loopdata['Searcher'] == searcher) & (loopdata['Hider'] == hider)]     
        
        block_avg_hiders = []
        block_hcaught = []
        block_hfound = []
        block_scaught = []
        block_sfound = []
        block_scount = []
        block_hcount = []
        for n in range(int(len(data)/200)):
            block = data.iloc[n*200:(n+1)*200]
            block_avg_hiders.append(sum(block['Result'])/len(block['Result']))
            
            hcaught = [0,0,0]
            hfound = [0,0,0]
            scaught = [0,0,0,0,0,0]
            sfound = [0,0,0,0,0,0]
            for index, row in block.iterrows():
                if row['Result'] == 0:
                    hcaught[row['Hide strat']] += 1
                    scaught[row['Search strat']] += 1
                if row['Result'] == 1:
                    hfound[row['Hide strat']] += 1
                    sfound[row['Search strat']] += 1
            block_hcaught.append(hcaught)
            block_hfound.append(hfound)
            block_scaught.append(scaught)
            block_sfound.append(sfound)
            block_scount.append([sum(x) for x in zip(scaught, sfound)])
            block_hcount.append([sum(x) for x in zip(hcaught, hfound)])
            
        avg_hiders.append(round(sum(block_avg_hiders)/len(block_avg_hiders),2))
        uncertainty.append(round(statistics.pstdev(block_avg_hiders),2))
        
        avg_hcaught = [sum(i)/len(i) for i in list(map(list, zip(*block_hcaught)))]
        std_hcaught = [statistics.pstdev(i) for i in list(map(list, zip(*block_hcaught)))]
        avg_hfound = [sum(i)/len(i) for i in list(map(list, zip(*block_hfound)))]
        std_hfound = [statistics.pstdev(i) for i in list(map(list, zip(*block_hfound)))]
        avg_scount = [sum(i)/len(i) for i in list(map(list, zip(*block_scount)))]
        std_scount = [statistics.pstdev(i) for i in list(map(list, zip(*block_scount)))]
        avg_hcount = [sum(i)/len(i) for i in list(map(list, zip(*block_hcount)))]
        std_hcount = [statistics.pstdev(i) for i in list(map(list, zip(*block_hcount)))]
        
        # calculate kullback-leibler divergence
        kldivh.append(sum(scipy.special.rel_entr(opth, [i/200 for i in avg_hcount])))
        kldivsi = []
        for optsi in opts:
            kldivsi.append(sum(scipy.special.rel_entr(optsi, [i/200 for i in avg_scount])))
        kldivs.append(kldivsi)
        

        fig = plt.subplots(figsize =(9, 6))
        p1 = plt.bar(range(3), [i/200 for i in avg_hcaught], color='blue') #, color = 'orange')  # change for github plots
        p2 = plt.bar(range(3), [i/200 for i in avg_hfound], bottom = [i/200 for i in avg_hcaught], color='blue') 
        plt.ylabel('Probability', fontsize=18)
        plt.xticks(range(3), ('1', '2', '3'))
        plt.xlabel('Hider strategies', fontsize=18)
        # plt.legend((p1[0], p2[0]), ('Searchers destroyed', 'Hiders found')) # uncomment for github plots
        # plt.title('HIDER strategies of {} vs searcher {}.'.format(hider, searcher)) # uncomment  for github plots
        
        avg_scaught = [sum(i)/len(i) for i in list(map(list, zip(*block_scaught)))]
        std_scaught = [statistics.pstdev(i) for i in list(map(list, zip(*block_scaught)))]
        avg_sfound = [sum(i)/len(i) for i in list(map(list, zip(*block_sfound)))]
        std_sfound = [statistics.pstdev(i) for i in list(map(list, zip(*block_sfound)))]
        
        fig = plt.subplots(figsize =(9, 6))      
        p1 = plt.bar(range(6), [i/200 for i in avg_scaught], color = 'blue') #, color = 'orange')  # change for github plots
        p2 = plt.bar(range(6), [i/200 for i in avg_sfound], bottom = [i/200 for i in avg_scaught], color = 'blue') 
        plt.ylabel('Probability', fontsize = 18)
        plt.xticks(range(6), ('1-2-3', '1-3-2', '2-1-3', '2-3-1', '3-1-2', '3-2-1'))
        plt.xlabel('Searcher strategies', fontsize=18)
        # plt.legend((p1[0], p2[0]), ('Searchers destroyed', 'Hiders found')) # uncomment for github plots
        # plt.title('SEARCHER strategies of {} vs searcher {}.'.format(searcher, hider)) # uncomment for github plots
        if searcher == 'robot' and hider == 'robot':
            plt.ylim((0,0.22))
        
        
avg_hiders = np.array(avg_hiders).reshape(3,4)
uncertainty = np.array(uncertainty).reshape(3,4)

f, ax = plt.subplots(figsize=(9, 6))
ax = sns.heatmap(avg_hiders, 
                 annot= avg_hiders.astype(str)+'±'+uncertainty.astype(str),
                 linewidth=.5, 
                 fmt='', 
                 center = 0.40909090, 
                 cmap='coolwarm', 
                 cbar_kws={'label': 'Average number of hiders found'})
ax.figure.axes[-1].yaxis.label.set_size(18)
ax.set_xlabel('Searcher agents', fontsize=22)
ax.set_xticklabels(['Naive risky', 'Naive safe', 'Belief-based', 'Human'])
ax.set_ylabel('Hider agents', fontsize=22)
ax.set_yticklabels(['Naive', 'Belief-based', 'Human'])
plt.show()        

##########################################

# Convergence plot
bb = bigdata[(bigdata['Searcher'] == 'robot') & (bigdata['Hider'] == 'robot')]
V = []

f, ax = plt.subplots(figsize=(9, 6))
for n in range(int(len(bb)/200)):
    bb_block = bb.iloc[n*200:(n+1)*200]  
    V.append(bb_block['V'])
      
Vavg = [sum(i)/len(i) for i in list(map(list, zip(*V)))]
Vstd = [statistics.pstdev(i) for i in list(map(list, zip(*V)))]
plt.plot(range(200), Vavg, color='b')
plt.fill_between(range(200), [i+j for i,j in zip(Vavg, Vstd)], [i-j for i,j in zip(Vavg, Vstd)], color='b', alpha=0.3)
y_ticks = np.append(ax.get_yticks(), 0.41)
ax.set_yticks(y_ticks)
ax.set_xlabel("Number of iterations", fontsize = 18)
ax.set_ylabel("Expected number of hiders found", fontsize = 18)
plt.axhline(y=0.41, color='r', linestyle='--', label="Optimal value", linewidth = 1.5)
plt.xlim((0,200))
plt.legend()



