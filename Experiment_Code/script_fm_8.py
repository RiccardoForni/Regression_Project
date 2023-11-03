import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import requests
import statsmodels.api as sm
"""import scipy as sp"""

"""
PUNTO 1
"""


#Import risk-free interest rates

data = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EURIBOR_3_M")
Interest_Rate_Monthly_Euribor=data[["BD INTEREST RATES - EURIBOR RATE "+
                                    "- 3 MONTH NADJ"]]/12

data = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="BUND")
Interest_Rate_Monthly_Bund=data[["RF GERMANY GVT BMK " +
                                 "BID YLD 3M - RED. YIELD"]]/12

#Import Market returns

data = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600")
mkt = pd.DataFrame()
mkt["Market_Return"] = 100*(np.log(data[data.columns[1]]) - 
                            np.log(data[data.columns[1]].shift(1)))

"""
PUNTO 2
"""

#Creating a new dataframe and adding the time column

ret = pd.DataFrame()
l = list(data.columns)
l.pop(0)
l.insert(0,"Time")
data.columns = l
ret['Time'] = data.Time

#Import stocks returns

data = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")

for i in range(1, data.shape[1]):
    
    nmcol = data.columns[i]
    ret[nmcol] = 100*(np.log(data[nmcol]) - np.log(data[nmcol].shift(1)))

#Adding the risk-free and market returns

ret['risk_free_euribor'] = Interest_Rate_Monthly_Euribor[
    Interest_Rate_Monthly_Euribor.columns[0]]
del Interest_Rate_Monthly_Euribor

ret['risk_free_bund'] = Interest_Rate_Monthly_Bund[
    Interest_Rate_Monthly_Bund.columns[0]]
del Interest_Rate_Monthly_Bund

ret['Market_Returns'] = mkt['Market_Return']
del mkt

#REMOVE TOTAL INDEX RETURN FROM COLUMN NAMES AND RENAME DATAFRAME COLUMNS

colnames = list(data.columns[1:])

for i in range(len(colnames)):
    
    colnames[i].replace(".","") 

    
    for j in range(0, len(colnames[i])):
               
        if colnames[i][j] == '-':
            
            if colnames[i][j+1] == ' ':
            
                colnames[i] = colnames[i][0:j-1]
                
                break

stock_names = colnames.copy()

colnames.insert(0, 'Time')
colnames.append('Risk-Free_euribor')
colnames.append('Risk-Free_bund')
colnames.append('Market_Return')


ret.columns = colnames

del data

#Dropping the first row

ret = ret.drop([0])

#COMPUTING THE EXCESS RETURNS (Euribor)

ex_ret_euribor = pd.DataFrame(columns = colnames)

for i in colnames:
    
    if i in stock_names:
        ex_ret_euribor[i] = ret[i] - ret['Risk-Free_euribor']
        
    else:
        ex_ret_euribor[i] = ret[i]


#PLOTTING SCATTER PLOTS (Excess returns Euribor) and saving them


cwd = os.getcwd()
f_euribor = cwd + "\EURIBOR" 

if not os.path.exists(f_euribor):
    os.mkdir(f_euribor)

for i in colnames[1:-3]:
    
    plt.figure()

    plt.scatter(ex_ret_euribor["Market_Return"],ex_ret_euribor[i])
    plt.title("{n} stock excess return vs market return (Euribor)".format(n = i))
    
    title = i + "_Euribor"
    plt.savefig(f_euribor + "\{n}.png".format(n = title))
    plt.close()
    
    
#COMPUTING THE EXCESS RETURNS (Bund)

ex_ret_bund = pd.DataFrame(columns = colnames)


for i in colnames:
    
    if i in stock_names:
        ex_ret_bund[i] = ret[i] - ret['Risk-Free_bund']
        
    else:
        ex_ret_bund[i] = ret[i]


#PLOTTING SCATTER PLOTS (Excess returns Bund) and saving them

cwd = os.getcwd()
f_euribor = cwd + "\Bund" 

if not os.path.exists(f_euribor):
    os.mkdir(f_euribor)

for i in colnames[1:-3]:
    
    plt.figure()


    plt.scatter(ex_ret_bund["Market_Return"],ex_ret_bund[i])
    plt.title("{n} stock excess return vs market return (Bund)".format(n = i))
    
    title = i + "_Bund"
    plt.savefig(f_euribor + "\{n}.png".format(n = title))
    plt.close()
    
    del title

"""
PUNTO 3
"""


l = ['alpha', 'p-value_alpha', 'beta', 'p-value_beta']
reg_results_bund = pd.DataFrame(index = stock_names, columns = l)



X = np . column_stack (( np . ones_like ( ex_ret_euribor["Market_Return"] ) , ex_ret_euribor["Market_Return"]))

for i in stock_names:
    
    Res1 = sm . OLS ( ex_ret_euribor[i] ,X). fit ()
    Res1.summary()
    param = Res1.params
    pval = Res1.pvalues
    l_val = [param[0], pval[0], param[1], pval[1]]
    reg_results_bund.loc[i] = l_val   

reg_results_bund = reg_results_bund.sort_values('p-value_alpha', ascending = False)

fig, ax = plt.subplots()

ax.bar(stock_names, reg_results_bund['p-value_alpha'])
x_pos = range(len(stock_names))
plt.xticks(x_pos, stock_names, rotation=90)


plt.show()

reg_results_bund = pd.DataFrame(index = stock_names, columns = l)

for i in stock_names:
    
    Res1 = sm . OLS ( ex_ret_bund[i] ,X). fit ()
    Res1.summary()
    param = Res1.params
    pval = Res1.pvalues
    l_val = [param[0], pval[0], param[1], pval[1]]
    reg_results_bund.loc[i] = l_val   

reg_results_bund = reg_results_bund.sort_values('p-value_alpha', ascending = False)

fig, ax = plt.subplots()

ax.bar(stock_names, reg_results_bund['p-value_alpha'])
x_pos = range(len(stock_names))
plt.xticks(x_pos, stock_names, rotation=90)

plt.show()

d = np.array(reg_results_bund)
type(d[1])

f = np.array([1,1])
type(f)
