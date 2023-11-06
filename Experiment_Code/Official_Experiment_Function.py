from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import statsmodels . api as sm
import pandas as pd
import os
import numpy as np

def plotbar(P):
    
    cwd = os.getcwd()
    folder = cwd + "/3_p_value_plots/"

    if not os.path.exists(folder):
        os.mkdir(folder)
    
    variable = P.name
    P = pd.DataFrame(data = P, columns = [variable])
    mean = P.loc['Mean', variable]
    P['stock_names'] = P.index

    def bar_highlight(value, one_value, 
                      five_value,
                      ten_value,
                      mean):
        if value <= one_value:
            return 'red'
        elif value <= five_value:
            return 'orange'
        elif value <= ten_value:
            return 'yellow'
        elif value == mean:
            return 'black'
        else:
            return 'grey'
    fig, ax = plt.subplots()
    
    one_value = 0.01
    five_value = 0.05
    ten_value = 0.1
    
    P['colors'] = P[variable].apply(bar_highlight, args = (one_value, 
                          five_value,
                          ten_value,
                          mean))

    bars = plt.bar(P['stock_names'], P[variable], color=P['colors'])
    x_pos = range(P['stock_names'].shape[0])
    plt.xticks(x_pos, P['stock_names'], rotation=90)
    

    plt.savefig(folder+"/"+variable+".png")
    plt.show()
    
    plt.close()


def plotCAPM(rStock,Market,stock_names,OLSRes,String):
    cwd = os.getcwd()
    folder = cwd + "/2_testCAPM/"

    if not os.path.exists(folder):
        os.mkdir(folder)

    for e in stock_names:

        plt.figure()
        plt.plot(Market, OLSRes.loc[e, 'beta: Market']*Market+OLSRes.loc[e, 'Alpha'])
        plt.scatter(Market,rStock[e])
        plt . xlabel ('Eurostoxx')
        plt . ylabel (e)
        plt.savefig("2_testCAPM/CAPM-"+e+String+".png")
        plt.close()

"""
- setx = market return
- set y = stock returns
- title = title of the graph
- xlabel = name of the x-axis
- ylabel = name of the y-axis
- sigla = unique name (either ER for euribor or ERB for bunds) to differentiate
          between plots of excess returns of stocks computed with Euribor
          or with the bund yield.
- Subset = dataframe with the data of total index returns
- string_to_save = name of the folder in which the graph will be saved in the 
                   working directory
"""

def plotscatter(setx,sety,title,xlabel,ylabel,sigla,string_to_save):
    cwd = os.getcwd()
    folder = cwd + "/"+string_to_save

    if not os.path.exists(folder):
        os.mkdir(folder)
        
    l = pd.DataFrame(index = sety.columns, columns= ['Plot'])
        
    for e in sety.columns:

        fig = plt.figure()
        
        ax1 = fig.add_subplot()
        
        ax1.scatter(setx,sety[e])
        plt.title(title)
        plt . xlabel (xlabel)
        plt . ylabel (e+ylabel)
        plt.savefig(folder+"/"+sigla+"-"+e+".png")
        l.loc[e,'Plot'] = fig
        plt.close()
    return l

def OLS(y, *x):

    try:
        
        intercept = pd.Series(data = np.ones_like(x[0]), name = "intercept",
                              index = x[0].index)
        
    except:
        
        intercept = pd.DataFrame(data = np.ones(y.shape[0] ), 
                              columns = ["intercept"],
                              index = y.index)
    
    try:
        X = pd.DataFrame([intercept, *x]).T

    except:
        X = pd.concat([intercept,*x],axis = 1)


    exog_names = list(X.columns)
    
    l = ['Alpha', 'p-value_alpha']
    
    for i in range(1, len(exog_names)):
        
        l.append("beta: " + exog_names[i])
        l.append("p-value_beta: "+ exog_names[i])
    
    l.append("R-Squared")

    try:
        endog_names = list(y.columns)
        result = pd.DataFrame(index = endog_names, columns = l)
        
    except:
        endog_names = [y.name]
        result = pd.DataFrame(index = endog_names, columns = l)
    
    reg = [] 
    
    for i in endog_names:
        
        try:  
            Res1 = sm . OLS ( y[i] ,X). fit ()
            Res1.summary()
            
        except:
            Res1 = sm . OLS (y ,X). fit ()

        r2 = Res1.rsquared
        param = Res1.params
        pval = Res1.pvalues
        reg.append(Res1)
        
        l_val = []
    
        for j in range(len(param)):
            
            l_val.extend([param[j],pval[j]])
        
        l_val.append(r2)
    
        result.loc[i] = l_val    
    
    return result, reg

def comparison_scatter(df_stocks, df_portfolios, market,
                       title,xlabel,ylabel,
                       sigla,string_to_save):
    cwd = os.getcwd()
    folder = cwd + "/"+string_to_save

    if not os.path.exists(folder):
        os.mkdir(folder)
    fig = plt.figure()
    ax1 = fig.add_subplot()
    
    for i in df_stocks.columns:
        ax1.scatter(market, df_stocks.loc[:,i], 
                    c = 'silver', alpha = 0.5)
    plt.title(title)
    plt . xlabel (xlabel)
    plt . ylabel (ylabel)

    
    ax1.scatter(market, df_portfolios, c = 'black')
    
    plt.savefig(folder+"/"+sigla+"-"+i+".png")
    plt.show()


def comparison_scatter_2(df_stocks, df_portfolios, market,
                       CAPM_Port,
                       title,xlabel,ylabel,
                       sigla,string_to_save):
    cwd = os.getcwd()
    folder = cwd + "/"+string_to_save

    if not os.path.exists(folder):
        os.mkdir(folder)
    fig = plt.figure()
    
    ax1 = fig.add_subplot()
    
    for i in df_stocks.columns:
        ax1.scatter(market, df_stocks.loc[:,i], 
                    c = 'silver', alpha = 0.5)
    plt.title(title)
    plt . xlabel (xlabel)
    plt . ylabel (ylabel)

    
    ax1.scatter(market, df_portfolios, c = 'black')
    ax1.plot(market, CAPM_Port.loc['Portfolio - EW','beta: Market']*
             market+CAPM_Port.loc['Portfolio - EW','Alpha'])
    
    plt.savefig(folder+"/"+sigla+"-"+i+".png")
    plt.show()
        
def m_scatter(CAPM_summary, df_factors, df_stocks,
              sheet,
              string_to_save):
    cwd = os.getcwd()
    folder = cwd + "/2_scatter_comparison/"

    if not os.path.exists(folder):
        os.mkdir(folder)
    
    
    x = list(CAPM_summary.index)
    y = x[:3] + x[-1:-4:-1]



    figure, axis = plt.subplots(2, 3) 

    for i in range(3):

        axis[0,i].scatter(df_factors['Market'],
                                df_stocks.loc[:,y[i]])

    for i in range(3,6):
        axis[1,i-3].scatter(df_factors['Market'],
                                df_stocks.loc[:,y[i]]) 
    
    plt.savefig(folder+"/"+string_to_save+".png")

    plt.show()

def RESET_test(l):
    df = pd.DataFrame(columns= ['F-Value', 'p-value'])
    results = []
    for i in range(len(l)):
        l_val = []
        x = l[i]
        x.fittedvalues = np.array(x.fittedvalues)
        f = smd.linear_reset(res = x , power = 3, test_type = "fitted", use_f = True)
        l_val.append(f.fvalue)
        l_val.append(f.pvalue)
        df.loc[l[i].model.endog_names,:] = l_val
        results.append(f)
    return df
