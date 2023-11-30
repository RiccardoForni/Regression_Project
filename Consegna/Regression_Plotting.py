from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

global allow_clean
allow_clean = False
if 'SPY_PYTHONPATH' in os.environ:
    allow_clean = True
    """
    Check Last line for clean variables
    """

"""Pre_config END"""
        
def folder_definer(folder):
        cwd = os.getcwd()
        PATH = cwd + "/"+folder+"/"

        if not os.path.exists(PATH):
            os.mkdir(PATH)
        
        return PATH
def controlla_permesso(f):
    def wrapper(self, *args, **kwargs):
        if self.allow_execution:
            return f(self, *args, **kwargs)
        else:
            return None
    return wrapper

class Plotting:
    def __init__(self, allow_execution):
        self.allow_execution = allow_execution
    @controlla_permesso
    def plotbar(self,P,SavePath, one_value = 0.01, five_value = 0.05, 
                ten_value = 0.1):
        """/3_p_value_plots/"""
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
                return 'gold'
            if value == mean:
                return 'black'
            else:
                return 'grey'
        fig, ax = plt.subplots()   
    
        P['colors'] = P[variable].apply(bar_highlight, args = (one_value, 
                            five_value,
                            ten_value,
                            mean))

        bars = plt.bar(P['stock_names'], P[variable], color=P['colors'])
        x_pos = range(P['stock_names'].shape[0])
        plt.xticks(x_pos, P['stock_names'], rotation=90)
        
        variable = variable.replace(":","_")
        plt.savefig(folder_definer(SavePath)+"/"+variable+".png")
        if allow_clean:
            plt.show()
        
        plt.close()

    @controlla_permesso
    def plotCAPM(self,rStock,Market,stock_names,OLSRes,SavePath):
        """/2_testCAPM/"""
        for e in stock_names:

            plt.figure()
            plt.plot(Market, OLSRes.loc[e, 'beta: Market']*Market+OLSRes.loc[e, 'Alpha'])
            plt.scatter(Market,rStock[e])
            plt . xlabel ('Eurostoxx')
            plt . ylabel (e)
            plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+"-"+e+SavePath[2]+".png")
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
    @controlla_permesso
    def plotscatter(self,setx,sety,title,xlabel,ylabel,sigla,SavePath):  

        l = pd.DataFrame(index = sety.columns, columns= ['Plot'])
            
        for e in sety.columns:
            
            ax1 = plt.figure().add_subplot()
            
            ax1.scatter(setx,sety[e])
            plt.title(title)
            plt . xlabel (xlabel)
            plt . ylabel (e+ylabel)
            plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+"-"+e+"_"+sigla+".png")
            l.loc[e,'Plot'] = plt.figure()
            plt.close("all")
            
        return l
    @controlla_permesso
    def comparison_scatter(self,df_stocks, df_portfolios, market,
                        title,xlabel,ylabel,
                        SavePath,CAPM_Port=None):
        ax1 = plt.figure().add_subplot()
        
        for i in df_stocks.columns:
            ax1.scatter(market, df_stocks.loc[:,i], 
                        c = 'silver', alpha = 0.5)
        plt.title(title)
        plt . xlabel (xlabel)
        plt . ylabel (ylabel)
        
        ax1.scatter(market, df_portfolios, c = 'black')

        if CAPM_Port is not None:
            ax1.plot(market, CAPM_Port.loc['Portfolio - EW','beta: Market']*
                    market+CAPM_Port.loc['Portfolio - EW','Alpha'])
            
        plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+"-"+i+".png")
        if allow_clean:
            plt.show()
        plt.close()

    @controlla_permesso
    def m_scatter(self,CAPM_summary, df_factors, df_stocks,
                SavePath):
        """
        /2_scatter_comparison/
        """
        
        x = list(CAPM_summary.index)
        y = x[:3] + x[-1:-4:-1]

        figure, axis = plt.subplots(2, 3) 

        for i in range(3):

            axis[0,i].scatter(df_factors['Market'],
                                    df_stocks.loc[:,y[i]])

        for i in range(3,6):
            axis[1,i-3].scatter(df_factors['Market'],
                                    df_stocks.loc[:,y[i]]) 
        
        plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+".png")

        if allow_clean:
            plt.show()
        plt.close()
    @controlla_permesso  
    def comparison_barplot(self,FF_summary, CAPM_summary):
        index = 1
        diction = {}
        for e,i in zip(range(1,FF_summary.shape[1]+1),list(FF_summary.columns)):
            diction.update({e: i})

        print('The possible comparisons are: {}\n'.format(diction.items()))
        index = int(input('Which one would you like to compare?(0 to stop)\n'))
        if index == 0 or index>(FF_summary.shape[1]+1):
            print("ENDED")
            return
        name=diction[index]
        
        index = np.arange(FF_summary.index.shape[0])
        bar_width = 0.35
        
        fig, ax = plt.subplots()
        summer = ax.bar(index, FF_summary.loc[:,name], bar_width,
                        label="Fama_French")
        
        winter = ax.bar(index + bar_width, CAPM_summary.loc[:,name], bar_width,
                        label="CAPM")
        
        ax.set_xlabel('Company')
        ax.set_ylabel('Value')
        ax.set_title('Comparison between CAPM and Fama-French model: {}'.format(name))
        
        x_pos = range(FF_summary.index.shape[0])
        plt.xticks(x_pos, FF_summary.index, rotation=90)
        
        ax.set_xticklabels(FF_summary.index)
        ax.legend()
        
        if allow_clean:
            plt.show()
        plt.close()
    
    @controlla_permesso
    def plot_looking(self,time_series,df_factors,df_portfolios):
        """
        Check why the only relevant variable appears to be the market
        """

        plt.figure()

        plt.plot(time_series, df_factors.loc[:, 'Market'], label = 'Market')
        plt.plot(time_series, df_portfolios.iloc[:,0], label = 'Portfolio')
        plt.legend()
        plt.show()
        plt.close()
    @controlla_permesso    
    def shish(p_val_df):

        for i in p_val_df.columns:
        
            plt.figure()
            plt.plot(p_val_df.loc[:, i])
            plt.axhline(y= 0.05, color = 'red')
            plt.title(i)

            plt.show()
            plt.close()

    @controlla_permesso
    def shish1(df):
        for i in df.columns:
        
            plt.plot(df.loc[:, i])
            plt.show()
            plt.close()

    @controlla_permesso    
    def plotting_CAPM_7(self,list_to_plot,d3,df_bd_CAPM_2,l_conf):

           
            for m in l_conf:

                for i in list_to_plot:
                    
                    l = df_bd_CAPM_2.loc[i]
                
                        
                    plt.figure()
                    
                    plt.plot(d3[i].index, d3[i][m+ '_UBound'], label = 'Upper bound')
                    
                    if m != 'Alpha':
                    
                        plt.plot(d3[i].index, d3[i]['beta: ' + m], label = 'Beta value')
                        
                    else:
                        
                        plt.plot(d3[i].index, d3[i][m], label = 'Alpha value')
                    
                    plt.plot(d3[i].index, d3[i][m+'_LBound'], label = 'Lower bound')
                    
                    if l.ndim > 1:
                        
                        for j in l['Date']:
                            
                            plt.axvline(j, color = 'red', linestyle = '--',
                                        label = 'Break date')
                    
                    else:
                        
                        for j in l:
                
                            plt.axvline(j, color = 'red', linestyle = '--',
                                        label = 'Break date')
                            
                    if m != 'Alpha':        
                        
                        plt.title(i +": {}".format('Value of beta '+ m))
                        
                    else:
                        
                        plt.title(i +": {}".format('Value of alpha'))
                    
                    plt.legend()
                    
                    plt.show()

            """
            Plot of values that do not have a confidence interval
            """

            l_roll = ['R-Squared', 'bic']

            for m in l_roll:

                for i in list_to_plot:
                    
                    l = df_bd_CAPM_2.loc[i]
                
                        
                    plt.figure()
                    
                    plt.plot(d3[i].index, d3[i][m], label = m)
                        
                
                    
                    if l.ndim > 1:
                        
                        for j in l['Date']:
                            
                            plt.axvline(j, color = 'red', linestyle = '--',
                                        label = 'Break date')
                    
                    else:
                        
                        for j in l:
                
                            plt.axvline(j, color = 'red', linestyle = '--',
                                        label = 'Break date')
                            
                        
                    plt.title(i +": {}".format('Value of '+m))
                    
                    plt.legend()
                    
                    plt.show()
