from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import statsmodels . api as sm
import statsmodels.stats.diagnostic as smd
import statsmodels.stats.stattools as smt
import pandas as pd
import os
import numpy as np
import scipy as sp

def folder_definer(folder):
    cwd = os.getcwd()
    PATH = cwd + "/"+folder+"/"

    if not os.path.exists(PATH):
        os.mkdir(PATH)
    
    return PATH
    
    
def plotbar(P,SavePath, one_value = 0.01, five_value = 0.05, 
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
    plt.show()
    
    plt.close()


def plotCAPM(rStock,Market,stock_names,OLSRes,SavePath):
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

def plotscatter(setx,sety,title,xlabel,ylabel,sigla,SavePath):  

    l = pd.DataFrame(index = sety.columns, columns= ['Plot'])
        
    for e in sety.columns:
        
        ax1 = plt.figure().add_subplot()
        
        ax1.scatter(setx,sety[e])
        plt.title(title)
        plt . xlabel (xlabel)
        plt . ylabel (e+ylabel)
        plt.savefig(folder_definer(SavePath[0])+"/"+SavePath[1]+"-"+e+"_"+sigla+".png")
        l.loc[e,'Plot'] = plt.figure()
        plt.close()
        
    return l

def f_test_retrieval(l):
    

    df = pd.DataFrame(columns = ['F-Test_Value','F-Test_p-value'])

    for i in range(len(l)):
        
        name = l[i].model.endog_names
        df.loc[name, 'F-Test_Value'] = l[i].fvalue
        df.loc[name, 'F-Test_p-value'] = l[i].f_pvalue
            
    return df

def f_test_retrieval_2(l):
    
    critical_alpha = l[l.iloc[:,1] < 0.05].iloc[:,1]

    l = CAPM_list

    df = pd.DataFrame(columns = [critical_alpha.name, 'F-Test_p-value'],
                      index = critical_alpha.index)
    
    df.loc[:,critical_alpha.name] = critical_alpha
    r = list(critical_alpha.index)

    for i in range(len(l)):
        
        name = l[i].model.endog_names

        if name in r:
            
            df.loc[name, 'F-Test_p-value'] = l[i].f_pvalue
            
    return df

def OLS(y, *x):

    intercept = pd.DataFrame(data = np.ones(y.shape[0] ), 
                              columns = ["intercept"],
                              index = y.index)
    
    X = pd.concat([intercept,*x],axis = 1)

    exog_names = list(X.columns)
    
    l = ['Alpha', 'p-value_alpha']
    
    for i in range(1, len(exog_names)):
        
        l.append("beta: " + exog_names[i])
        l.append("p-value_beta: "+ exog_names[i])
    
    l.append("R-Squared")
    l.append('bic')
    l.append('aic')

    endog_names = list(y.columns)
    result = pd.DataFrame(index = endog_names, columns = l)
        
    reg = [] 
    
    for i in endog_names:
        
        Res1 = sm . OLS ( y[i] ,X). fit ()
        Res1.summary()
    
        r2 = Res1.rsquared
        bic = Res1.bic
        aic = Res1.aic
        param = Res1.params
        pval = Res1.pvalues
        reg.append(Res1)
        
        l_val = []
    
        for j in range(len(param)):
            
            l_val.extend([param[j],pval[j]])
        
        l_val.append(r2)
        l_val.append(bic)
        l_val.append(aic)
    
        result.loc[i] = l_val    
    
    return result, reg

def comparison_scatter(df_stocks, df_portfolios, market,
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
    plt.show()

def m_scatter(CAPM_summary, df_factors, df_stocks,
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

def h_test(l):
    
    df = pd.DataFrame(columns= ['F-Value', 'p-value'])
    
    for i in range(len(l)):
        
        l_val = []
        residuals = l[i].resid
        exogen = l[i].model.exog
        f = smd.het_white(residuals, exogen)
        l_val.append(f[2])
        l_val.append(f[3])
        df.loc[l[i].model.endog_names,:] = l_val
        
    return df

def Durbin_Watson_test(l):
    
    df = pd.DataFrame(columns= ["Test-statistic"])
    
    for i in range(len(l)):
        
        l_val = []
        
        residuals = l[i].resid.copy()
        residuals = np.array(residuals)
        
        f = smt.durbin_watson(residuals)
        l_val.append(f)
        
        df.loc[l[i].model.endog_names,:] = l_val
        
    return df

def Breusch_Godfrey_test(l):
    
    df = pd.DataFrame(columns= ['F-Value', 'p-value'])
    
    for i in range(len(l)):
        
        l_val = []

        f = smd.acorr_breusch_godfrey(l[i], nlags = 3)
        
        l_val.append(f[2])
        l_val.append(f[3])
        
        df.loc[l[i].model.endog_names,:] = l_val
        
    return df

def comparison_barplot(FF_summary, CAPM_summary):
    
    name = 'empty'
    

    
    print('The possible comparisons are: {}\n'.format(list(FF_summary.columns)))
    print("(Insert 0 to stop)\n")
    name = input('Which one would you like to compare?\n')

    
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
    
    plt.show()

def GETS_BIC(FF_summary, df_factors, df_stocks):

    elim = []
    
    summary = FF_summary.copy()
    df_fac = df_factors.copy()
    
    bic = summary.loc['Mean', 'bic']
    bic_list = [bic]
    
    results = []
    
    while True:
    
        p_v = [i for i in list(summary.columns) if ('p-value' in i)] 
        
        del p_v[0:2] 
        names = [j[-3:] for j in p_v ] 
        
        temp_df = pd.DataFrame(index = names, columns = ['p-values'])  
        
        p_values = []
        
        for i in p_v: 
            
            p_values.append(summary.loc['Mean', i])
        
        if p_values:         
            
            temp_df.iloc[:,0] = p_values
            temp_df = temp_df.sort_values('p-values', ascending = False)
            
            if temp_df.iloc[0,0] > 0.05:
               
                elim.append(temp_df.index[0])
                df_fac = df_fac.drop(elim[-1], axis = 1)
                
                summary, FF_list_2 = OLS(df_stocks,df_fac)
                summary.loc['Mean'] = summary.mean()
                results.append(summary)
                
                if summary.loc['Mean', 'bic'] < bic:
                    bic = summary.loc['Mean', 'bic']
                    bic_list.append(bic)
                    
                else:
                    bic = summary.loc['Mean', 'bic']
                    bic_list.append(bic)
                    results.pop()
                    break
                
            else:
                break
        
        else:
            break
        
    return results[-1],bic_list

def GETS_AIC(FF_summary, df_factors, df_stocks):

    elim = []
    
    summary = FF_summary.copy()
    df_fac = df_factors.copy()
    
    aic = summary.loc['Mean', 'aic']
    aic_list = [aic]
    
    results = []
    
    while True:
    
        p_v = [i for i in list(summary.columns) if ('p-value' in i)] 
        
        del p_v[0:2] 
        names = [j[-3:] for j in p_v ] 
        
        temp_df = pd.DataFrame(index = names, columns = ['p-values'])  
        
        p_values = []
        
        for i in p_v: 
            
            p_values.append(summary.loc['Mean', i])
        
        if p_values:         
            
            temp_df.iloc[:,0] = p_values
            temp_df = temp_df.sort_values('p-values', ascending = False)
            
            if temp_df.iloc[0,0] > 0.05:
               
                elim.append(temp_df.index[0])
                df_fac = df_fac.drop(elim[-1], axis = 1)
                
                summary, FF_list_2 = OLS(df_stocks,df_fac)
                summary.loc['Mean'] = summary.mean()
                results.append(summary)
                
                if summary.loc['Mean', 'aic'] < aic:
                    aic = summary.loc['Mean', 'aic']
                    aic_list.append(aic)
                    
                else:
                    aic = summary.loc['Mean', 'aic']
                    aic_list.append(aic)
                    results.pop()
                    break
                
            else:
                break
        
        else:
            break
        
    return results[-1],aic_list


def GETS_BIC_worst(FF_summary, df_factors, df_stocks):

    elim = []
    
    summary = FF_summary.copy()
    df_fac = df_factors.copy()
    
    bic = max(summary.loc[:, 'bic'])
    bic_list = [bic]
    
    results = []
    
    while True:
    
        p_v = [i for i in list(summary.columns) if ('p-value' in i)] 
        
        del p_v[0:2] 
        names = [j[-3:] for j in p_v ] 
        
        temp_df = pd.DataFrame(index = names, columns = ['p-values'])  
        
        p_values = []
        
        for i in p_v: 
            
            p_values.append(summary.loc['Mean', i])
        
        if p_values:         
            
            temp_df.iloc[:,0] = p_values
            temp_df = temp_df.sort_values('p-values', ascending = False)
            
            if temp_df.iloc[0,0] > 0.05:
               
                elim.append(temp_df.index[0])
                df_fac = df_fac.drop(elim[-1], axis = 1)
                
                summary, FF_list_2 = OLS(df_stocks,df_fac)
                summary.loc['Mean'] = summary.mean()
                results.append(summary)

                
                if max(summary.loc[:, 'bic']) < bic:
                    bic = max(summary.loc[:, 'bic'])
                    bic_list.append(bic)
                    
                else:
                    bic = max(summary.loc[:, 'bic'])
                    bic_list.append(bic)
                    results.pop()
                    break
                
            else:
                break
        
        else:
            break
        
    return results[-1],bic_list

def GETS_BIC_p(FF_summary, df_factors, df_stocks):

    elim = []
    
    summary = FF_summary.copy()
    df_fac = df_factors.copy()
    
    bic = summary.loc[summary.index[0], 'bic']
    bic_list = [bic]
    
    results = []
    
    while True:
    
        p_v = [i for i in list(summary.columns) if ('p-value' in i)] 
        
        del p_v[0:2] 
        names = [j[-3:] for j in p_v ] 
        
        temp_df = pd.DataFrame(index = names, columns = ['p-values'])  
        
        p_values = []
        
        for i in p_v: 
            
            p_values.append(summary.loc[summary.index[0], i])
        
        if p_values:         
            
            temp_df.iloc[:,0] = p_values
            temp_df = temp_df.sort_values('p-values', ascending = False)
            
            if temp_df.iloc[0,0] > 0.05:
               
                elim.append(temp_df.index[0])
                df_fac = df_fac.drop(elim[-1], axis = 1)
                
                summary, FF_list_2 = OLS(df_stocks,df_fac)

                results.append(summary)
                
                if summary.loc[summary.index[0], 'bic'] < bic:
                    bic = summary.loc[summary.index[0], 'bic']
                    bic_list.append(bic)
                    
                else:
                    bic = summary.loc[summary.index[0], 'bic']
                    bic_list.append(bic)
                    results.pop()
                    break
                
            else:
                break
        
        else:
            break
        
    return results[-1],bic_list

def ad_hoc_GETS(FF_summary, df_factors, df_stocks):
    
    df = pd.DataFrame(index = FF_summary.index, columns = FF_summary.columns)
    final_models = []
    
    for i in df.index:
        
        print(i)
        
        if i == 'Mean':
            
            continue
        
        #Create an object to be compatible with the function GETS_BIC_p
        df_2 = pd.DataFrame( columns = FF_summary.columns)
        df_2.loc[len(df_2)] = FF_summary.loc[i,:].values
        
        df_3 = pd.DataFrame(index = df_stocks.index)
        df_3[i] = df_stocks.loc[:,i].values
        
        res, bic = GETS_BIC_p(df_2, df_factors,df_3)
        final_models.append(res)

        
    return final_models
        
        


def CHOW_TEST(df_stocks, df_factors):
    sub = 0.1
    prop = int(sub*df_stocks.shape[0])
    
    end = df_stocks.shape[0] - prop
    
    index = df_stocks.index[(prop-1):]
    index = index[:-(prop)]
    
    p_val_df = pd.DataFrame(columns = df_stocks.columns, index = index)
    
    while prop <= end:
    
        prop_df = df_stocks.iloc[:prop,:]
        
        prop_factors = df_factors.iloc[:prop, :]
        prop_factors = prop_factors.loc[:,'Market']
        
        
        compl_df = df_stocks.iloc[prop:,:]
        
        compl_factors = df_factors.iloc[prop:, :]
        compl_factors = compl_factors.loc[:,'Market']
        
        
        prop_summary, prop_reg = OLS(prop_df, prop_factors)
        
        compl_summary, compl_reg = OLS(compl_df, compl_factors)
        
        total_summary, total_reg = OLS(df_stocks, df_factors.loc[:,'Market'])
        
    
        
        
        for i in range(len(prop_reg)):
            
            if (prop_reg[i].model.endog_names == 
                compl_reg[i].model.endog_names) and (prop_reg[i].model.endog_names == 
                                                     total_reg[i].model.endog_names):
            
                RSS_prop = prop_reg[i].ssr
                RSS_compl = compl_reg[i].ssr
                RSS_tot = total_reg[i].ssr
                
                F = ((RSS_tot - RSS_prop - RSS_compl)/2)/ ((RSS_prop + RSS_compl)/(df_stocks.shape[0] - 2*2))
                
                p_value = 1 - sp.stats.f.cdf(F, 2, df_stocks.shape[0] - 2*2)
                
                p_val_df.loc[prop_df.index[-1], prop_reg[i].model.endog_names] = p_value
            
            else:
                print("PROBLEM")
        
        prop += 1
    
    return p_val_df
