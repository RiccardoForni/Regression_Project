from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns

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
                ten_value = 0.1, obj = ''):
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
        plt.title(obj)
        
        variable = variable.replace(":","_")
        plt.savefig(folder_definer(SavePath)+"/"+variable+".png")
        if allow_clean:
            plt.show()
        
        plt.close()
        
    @controlla_permesso
    def plotbar_DW(self,P,SavePath, Lbound = 0.01, Ubound= 0.05, 
                   obj = '', conf = 0.05,
                   pos_autocorr = True):
        """/3_p_value_plots/"""
        variable = P.name
        P = pd.DataFrame(data = P, columns = [variable])
        mean = P.loc['Mean', variable]
        P['stock_names'] = P.index

        def bar_highlight(value, Lbound, Ubound, mean):
            
            if pos_autocorr:
                #there is statistical evidence that the error terms are positively autocorrelated
                if value <= Lbound:
                    return 'red'
                #there is no statistical evidence that the error terms are positively autocorrelated
                elif value >= Ubound:
                    return 'grey'
    
                if value == mean:
                    return 'black'
                #the test is inconclusive
                else:
                    return 'blue'
                
            else: 
                
                #there is statistical evidence that the error terms are negatively autocorrelated
                if (4 - value) <= Lbound:
                    return 'red'
                #there is no statistical evidence that the error terms are negatively autocorrelated
                elif (4-value) >= Ubound:
                    return 'grey'
                
                if value == mean:
                    return 'black'
                #the test is inconclusive
                else:
                    return 'blue'
                
        fig, ax = plt.subplots()   
    
        P['colors'] = P[variable].apply(bar_highlight, args = (Lbound, Ubound, mean))

        bars = plt.bar(P['stock_names'], P[variable], color=P['colors'])
        x_pos = range(P['stock_names'].shape[0])
        plt.xticks(x_pos, P['stock_names'], rotation=90)
        if pos_autocorr:
            plt.title("{name}: H0 = Absence of positive autocorrelation, confidence level = {cf}".format(
                name = obj, cf = conf))
            
        else:
            plt.title("{name}: H0 = Absence of negative autocorrelation, confidence level = {cf}".format(
                name = obj, cf = conf))
        
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
    def comparison_barplot(self,FF_summary, CAPM_summary,
                           label1 = 'CAPM',
                           label2 = 'Fama-French',
                           legend_pos = 'best'):
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
                        label= label2)
        
        winter = ax.bar(index + bar_width, CAPM_summary.loc[:,name], bar_width,
                        label = label1)
        
        ax.set_xlabel('Company')
        ax.set_ylabel('Value')
        ax.set_title('Comparison between {CAPM} and {FF}: {n}'.format(n = name, CAPM = label1,
                                                                                  FF = label2))
        
        x_pos = range(FF_summary.index.shape[0])
        plt.xticks(x_pos, FF_summary.index, rotation=90)
        
        ax.set_xticklabels(FF_summary.index)
        ax.legend(loc = legend_pos)
        
        if allow_clean:
            plt.show()
        plt.close()
        
    @controlla_permesso
    def factor_plot(self,GETS_ad_hoc_summary,df_factors):
        
        l = [i for i in GETS_ad_hoc_summary.columns if i[0:4] == 'beta']
        
        df_to_plot = pd.DataFrame(columns = l, index = GETS_ad_hoc_summary.index)
        
        for i in l:
            
            k = np.array(GETS_ad_hoc_summary[i])
            
            for j in range(len(k)):
        
                if np.isnan(k[j]):
                    
                    a = 0 
                    #a = np.dtype('int64').type(a)
                    k[j] = a
                
                else:
                    
                    a = 1
                    #a = np.dtype('int64').type(a)
                    k[j] = a
                    
            df_to_plot[i] = k
        
        df_to_plot = df_to_plot.astype(int)
        df_to_plot.columns = df_factors.columns
        

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_to_plot, cmap='Blues', fmt='d', cbar=False,
                    linewidths=0.5, linecolor='black')
        
        plt.xlabel('Factors')
        plt.ylabel('Stocks')
        
        plt.show()
        
        
        
    
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
    def fama_french_plotting(self,df, model):
        # Convert datetime objects to numerical values for plotting
        x_values = date2num(df['Date'])

        plt.figure()

        # Set up the bar plot
        plt.bar(x_values, height = 1, width = 25)


        plt.xticks(df['Date'], df['Date'])
        plt.xticks(rotation=90, ha='right')
        plt.title('Break date distribution ({})'.format(model))

        # Format the x-axis as dates


        # Display the plot
        plt.show()
    @controlla_permesso
    def chow_test_plotting(self,p_val_df, model):
        for i in p_val_df.columns:
        
            plt.figure()
            plt.plot(p_val_df.loc[:, i])
            plt.axhline(y= 0.05, color = 'red')
            plt.title(i+ " ({})".format(model))

            plt.show()




    @controlla_permesso    
    def plotting_CAPM_7(self, list_to_plot,d3,df_bd_CAPM_2,l_conf, end):

           
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
                        
                        plt.title(i +": {par} ({mo} months rolling window)".format(par = 'Value of beta '+ m, mo = end + 1))
                        
                    else:
                        
                        plt.title(i +": {par} ({mo} months rolling window)".format(par = 'Value of alpha', mo = end + 1))
                    
                    plt.legend()
                    
                    plt.show()

            """
            Plot of values that do not have a confidence interval
            """

            l_roll = ['R-Squared']

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
                            
                        
                    plt.title(i +": {par} ({mo} months rolling window)".format(par = 'Value of '+m, mo =  end + 1))
                    
                    plt.legend()
                    
                    plt.show()
