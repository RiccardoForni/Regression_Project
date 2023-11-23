import statsmodels . api as sm
import statsmodels.stats.diagnostic as smd
import statsmodels.stats.stattools as smt
import pandas as pd
import numpy as np
import scipy as sp


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


"""
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
"""


def GETS_ABIC(FF_summary, df_factors, df_stocks,param):
    loc=None
    str=None
    match param:
        case 'a':
            loc='Mean'
            str='aic'
        case 'b':
            loc=FF_summary.index[0]
            str='bic'
        case 'c':
            loc='Mean'
            str='bic'

    elim = []
    
    summary = FF_summary.copy()
    df_fac = df_factors.copy()
    
    ic = summary.loc[loc, str]
    ic_list = [ic]
    
    results = []
    
    while True:
    
        p_v = [i for i in list(summary.columns) if ('p-value' in i)] 
        
        del p_v[0:2] 
        names = [j[-3:] for j in p_v ] 
        
        temp_df = pd.DataFrame(index = names, columns = ['p-values'])  
        
        p_values = []
        
        for i in p_v: 
            
            p_values.append(summary.loc[loc, i])
        
        if p_values:         
            
            temp_df.iloc[:,0] = p_values
            temp_df = temp_df.sort_values('p-values', ascending = False)
            
            if temp_df.iloc[0,0] > 0.05:
               
                elim.append(temp_df.index[0])
                df_fac = df_fac.drop(elim[-1], axis = 1)
                
                summary, FF_list_2 = OLS(df_stocks,df_fac)
                if param == 'a':
                    summary.loc[loc] = summary.mean()
                results.append(summary)
                
                if summary.loc[loc, str] < ic:
                    ic = summary.loc[loc,str]
                    ic_list.append(ic)
                    
                else:
                    ic = summary.loc[loc, str]
                    ic_list.append(ic)
                    results.pop()
                    break
                
            else:
                break
        
        else:
            break
        
    return results[-1],ic_list


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
        
        res, bic = GETS_ABIC(df_2, df_factors,df_3,'b')
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

