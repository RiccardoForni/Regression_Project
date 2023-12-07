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


    df = pd.DataFrame(columns = [critical_alpha.name, 'F-Test_p-value'],
                      index = critical_alpha.index)
    
    df.loc[:,critical_alpha.name] = critical_alpha
    r = list(critical_alpha.index)

    for i in range(len(l)):
        
        name = l[i].model.endog_names

        if name in r:
            
            df.loc[name, 'F-Test_p-value'] = l[i].f_pvalue
            
    return df


def OLS(y, *x, hac =False, conf_int = [False]):

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
    
    if conf_int[0] == True:
        for i in conf_int[1]:        
            l.append(i+ '_LBound')
            l.append(i +'_UBound')

    endog_names = list(y.columns)
    result = pd.DataFrame(index = endog_names, columns = l)
        
    reg = [] 
    
    for i in endog_names:
        
        Res1 = sm . OLS ( y[i] ,X). fit ()
        Res1.summary()
        
        if hac == True:

            #Checking for heteroskedasticity
            residuals = Res1.resid
            exogen = Res1.model.exog
            het = smd.het_white(residuals, exogen)
            ind = smd.acorr_breusch_godfrey(Res1, nlags = 1)
            
            if (het[3] < 0.05) and (ind[3] <0.05):
                Res1 = sm . OLS ( y[i] ,X). fit (cov_type ='HAC',cov_kwds= {'maxlags':1})
                #print('HAAAAAAAAAAAC: {}'.format(i))
                
            elif het[3] < 0.05:
                
                Res1 = sm . OLS ( y[i] ,X). fit (cov_type ='HC3')
                #print('HETEROOOOOOO: {}'.format(i) )
                
            elif ind[3] < 0.05:
                Res1 = sm . OLS ( y[i] ,X). fit (cov_type ='HAC',cov_kwds= {'maxlags':1})
                #print('SERIAL CORRELAAAAATION: {}'.format(i))
                
    
        r2 = Res1.rsquared
        bic = Res1.bic
        aic = Res1.aic
        param = Res1.params
        pval = Res1.pvalues
        reg.append(Res1)
        
        if conf_int[0] == True:
            
            intervals = Res1.conf_int(alpha = 0.05)
        
        l_val = []
    
        for j in range(len(param)):
            
            l_val.extend([param[j],pval[j]])
        
        l_val.append(r2)
        l_val.append(bic)
        l_val.append(aic)
        
        if conf_int[0] == True:
            
            for j in range(intervals.shape[0]):
            
                l_val.append(intervals.iloc[j,0])
                l_val.append(intervals.iloc[j,1])
        
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

def Breusch_Godfrey_test(l, n=1):
    
    df = pd.DataFrame(columns= ['F-Value', 'p-value'])
    
    for i in range(len(l)):
        
        l_val = []

        f = smd.acorr_breusch_godfrey(l[i], nlags = n)
        
        l_val.append(f[2])
        l_val.append(f[3])
        
        df.loc[l[i].model.endog_names,:] = l_val
        
    return df

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
                if param == 'a' or param == "c":
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



def GETS_BIC_p(FF_summary, df_factors, df_stocks):

    elim = []
    
    summary = FF_summary.copy()
    df_fac = df_factors.copy()
    
    bic = summary.loc[summary.index[0], 'bic']
    bic_list = [bic]
    
    results = []
    
    while True:
    
        p_v = [i for i in list(summary.columns) if ('p-value' in i)] 
        
        del p_v[0] 
        names = [j[-3:] for j in p_v[1:] ] 
        """
        ----------------------------------------- HARD-CODED
        """    
        names.insert(0, 'Market')
        
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
                
                summary, FF_list_2 = OLS(df_stocks,df_fac, hac = True)

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
    sub = 0.2
    prop = int(sub*df_stocks.shape[0])
    
    if prop <= 20:
        prop = 21
    
    
    end = df_stocks.shape[0] - prop
    
    index = df_stocks.index[(prop-1):]
    index = index[:-(prop)]
    
    p_val_df = pd.DataFrame(columns = df_stocks.columns, index = index)
    
    hac_check = True
    
    while prop <= end:
    
        prop_df = df_stocks.iloc[:prop,:]
        compl_df = df_stocks.iloc[prop:,:]
        
        
        if df_factors.ndim == 1: 
            
            df_factors = df_factors.to_frame()
            hac_check = False
        
        
        prop_factors = df_factors.iloc[:prop, :]
        prop_factors = prop_factors.loc[:,'Market']       
       
        compl_factors = df_factors.iloc[prop:, :]
        compl_factors = compl_factors.loc[:,'Market']
        
        
        prop_summary, prop_reg = OLS(prop_df, prop_factors, hac = hac_check)
        
        compl_summary, compl_reg = OLS(compl_df, compl_factors, hac = hac_check)
        
        total_summary, total_reg = OLS(df_stocks, df_factors.loc[:,'Market'], hac = hac_check)
        
    
        
        
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


def CHOW_TEST_FF(df_stocks, df_factors):
    sub = 0.2
    prop = int(sub*df_stocks.shape[0])
    
    if prop <= 20:
        prop = 21
    
    end = df_stocks.shape[0] - prop
    
    index = df_stocks.index[(prop-1):]
    index = index[:-(prop)]
    
    p_val_df = pd.DataFrame(columns = df_stocks.columns, index = index)
    
    while prop <= end:
    
        prop_df = df_stocks.iloc[:prop,:]
        
        prop_factors = df_factors.iloc[:prop, :]

        
        
        compl_df = df_stocks.iloc[prop:,:]
        
        compl_factors = df_factors.iloc[prop:, :]

        
        
        prop_summary, prop_reg = OLS(prop_df, prop_factors, hac = True)
        
        compl_summary, compl_reg = OLS(compl_df, compl_factors, hac = True)
        
        total_summary, total_reg = OLS(df_stocks, df_factors, hac = True)
        
        k = len(prop_factors.columns) + 1
        
    
        
        
        for i in range(len(prop_reg)):
            
            if (prop_reg[i].model.endog_names == 
                compl_reg[i].model.endog_names) and (prop_reg[i].model.endog_names == 
                                                     total_reg[i].model.endog_names):
            
                RSS_prop = prop_reg[i].ssr
                RSS_compl = compl_reg[i].ssr
                RSS_tot = total_reg[i].ssr
                
                F = ((RSS_tot - RSS_prop - RSS_compl)/k)/ ((RSS_prop + RSS_compl)/(df_stocks.shape[0] - 2*k))
                
                p_value = 1 - sp.stats.f.cdf(F, 2, df_stocks.shape[0] - 2*2)
                
                p_val_df.loc[prop_df.index[-1], prop_reg[i].model.endog_names] = p_value
            
            else:
                print("PROBLEM")
        
        prop += 1
    
    return p_val_df


def CAPM_break_dates(p_val_df, CAPM_summary,
                     df_stocks, df_factors):
    min_pval = []
    index_min = []
    
    break_dates_df = pd.DataFrame(index = p_val_df.columns, columns = ['min_pval', 'date'])
    
    for i in p_val_df.columns:
        
        min_pval = min(p_val_df.loc[:,i])
        index_min = p_val_df.loc[:,i].idxmin()
        break_dates_df.loc[i,:] = [min_pval, index_min]
        
        
    d2 = {}
    
    l_col = []
    l_col = l_col + list(CAPM_summary.columns) + ['beg_date', 'end_date']
    
    for i in df_stocks.columns:
        
        d2[i] = pd.DataFrame(columns = l_col)
        
    
    no_break_stocks = []
    
    no_break_stocks = break_dates_df[break_dates_df['min_pval'] > 0.05].index.to_list()
    
    break_dates_df = break_dates_df[break_dates_df['min_pval'] < 0.05]
    
    
    
    for i in no_break_stocks:
        
        d2[i] = pd.concat([d2[i],CAPM_summary.loc[i].to_frame().T], 
                            ignore_index = True)
        d2[i]['beg_date'] = df_stocks.index[0]
        d2[i]['end_date'] = df_stocks.index[-1]
    
    
    for name in break_dates_df.index: 
        break_date = 1
        
        d2[name] = pd.DataFrame(columns = l_col)
        start_date = df_stocks.index[0]
        
        
        
    
        break_date = break_dates_df.loc[name, 'date']
        
        i=0 
        
        while break_date != 0:
            reg_summary, reg_list = OLS(df_stocks.loc[start_date:break_date, name].to_frame(),
                                           df_factors.loc[start_date:break_date, 'Market'], 
                                           hac = True)
            """
            Storing the model before the break 
            """   
            
            final_res = OLS(df_stocks.loc[start_date:break_date, name].to_frame(),
                                       df_factors.loc[start_date:break_date, 'Market'],hac = True)
            
            
            d2[name] = pd.concat([d2[name], final_res[0]], ignore_index = True)
            
            d2[name].iloc[i,-2] = df_stocks.loc[start_date:break_date].index[0]
            d2[name].iloc[i,-1] = df_stocks.loc[start_date:break_date].index[-1]
            
            """
            DO WE HAVE TO CHECK FOR BREAKS EVEN BEFORE THE BREAK???????????????????????????
            """
            #p_val_df_2 = CHOW_TEST(df_stocks.loc[:break_date, name].to_frame(),
                                         #df_factors.loc[:break_date, 'Market'])
            
            """
            ----------------------------------------------------------------------------------------
            ----------------------------------------------------------------------------------------
            """
    
            
            """
            Checking for further breaks
            """
            
            p_val_df_2 = CHOW_TEST(df_stocks.loc[break_date:, name].to_frame(),
                                         df_factors.loc[break_date:, 'Market'])
                
            if p_val_df_2.empty:
                
                i = i + 1
                reg_summary, reg_list = OLS(df_stocks.loc[break_date:, name].to_frame(),
                                               df_factors.loc[break_date:, 'Market'], hac = True)
                
    
                d2[name] = pd.concat([d2[name], reg_summary], ignore_index = True)
                
                d2[name].iloc[i,-2] = df_stocks.loc[break_date:].index[0]
                d2[name].iloc[i, -1] = df_stocks.loc[break_date:].index[-1]
                break_date = 0
                
            
            elif (min(p_val_df_2[name]) > 0.05) :
            
                i = i + 1
                reg_summary, reg_list = OLS(df_stocks.loc[break_date:, name].to_frame(),
                                               df_factors.loc[break_date:, 'Market'], hac = True)
                
            
            
                
                
                d2[name] = pd.concat([d2[name], reg_summary], ignore_index = True)
                
                d2[name].iloc[i,-2] = df_stocks.loc[break_date:].index[0]
                d2[name].iloc[i, -1] = df_stocks.loc[break_date:].index[-1]
                break_date = 0
                
                
            else:
                start_date = break_date
                break_date = p_val_df_2[name].idxmin()
            
            
            i = i+1
    
    return d2


def break_dates_optimization(p_val_df_FF, FF_summary, df_stocks, df_factors):
        """
        PROCEDURE TO FIND BREAK DATES
        """
        
        """
        First we create a dataframe that, for each stock, assign the min_pval of the
        Chow test and the corresponding date that corresponds to that minimum value
        """
        


        min_pval = []
        index_min = []
        
        break_dates_df_FF = pd.DataFrame(index = p_val_df_FF.columns, columns = ['min_pval', 'date'])
        
        for i in p_val_df_FF.columns:
            
            min_pval = min(p_val_df_FF.loc[:,i])
            index_min = p_val_df_FF.loc[:,i].idxmin()
            break_dates_df_FF.loc[i,:] = [min_pval, index_min]
        
        
        """
        -------------------------------------------------------------------------------
        Estimating optimized models for which no structural break was detected
        -------------------------------------------------------------------------------
        """
        
        """
        We create a dictionary in which we will store the results of the models 
        estimated and optimized in each interval according to their break dates.
        These results will be stored in dataframes.
        If a stock doesn't have any break there will be a dataframe composed of a single
        row.
        If a stock show the presence of one or more breaks, it will have a number of rows
        equal to the number of breaks detected plus one.
        """
        d = {}
        
        l_col = []
        l_col = l_col + list(FF_summary.columns) + ['beg_date', 'end_date']
        
        for i in df_stocks.columns:
            
            d[i] = pd.DataFrame(columns = l_col)
        
        #Finding the stocks for which the FF model with all the factors didn't show breaks
        list_to_GETS = break_dates_df_FF[break_dates_df_FF['min_pval'] > 0.05].index.to_list()
        
        """
        Removing those stocks from the dataframe in which we stored the date and p-value
        of critical dates
        """
        break_dates_df_FF = break_dates_df_FF[break_dates_df_FF['min_pval'] < 0.05]
        
        """
        Optimizing by removing irrelevant variables for the models of the stocks that 
        didn't show any breaks
        """
        final_res = ad_hoc_GETS(FF_summary.loc[list_to_GETS], 
                                df_factors, df_stocks[list_to_GETS])
        
        
        """
        Storing the results of these models in a dataframe 
        """
        
        final_res_df= pd.DataFrame( columns = l_col)
        
        for i in range(len(final_res)):
            final_res_df = pd.concat([final_res_df, final_res[i]], axis = 0)
            
        final_res_df['beg_date'] = df_stocks.index[0]
        final_res_df['end_date'] = df_stocks.index[-1]
        
        """
        Storing these dataframes in the dictionary
        """    
        
        for i in range(len(final_res)):   
            
            name = final_res[i].index[0]
            d[name] = pd.concat([d[name],final_res_df.loc[name].to_frame().T], 
                                ignore_index = True)
            
        """
        Stocks that have breaks
        """
        
        
        for name in break_dates_df_FF.index: 
            break_date = 1
            
            d[name] = pd.DataFrame(columns = l_col)
            start_date = df_stocks.index[0]
            
            
            
        
            break_date = break_dates_df_FF.loc[name, 'date']
            
            i=0 
            
            while break_date != 0:
                reg_summary, reg_list = OLS(df_stocks.loc[start_date:break_date, name].to_frame(),
                                            df_factors[start_date:break_date], hac = True)
                """
                Storing the model before the break after removing irrelevant variables
                """   
                
                final_res = ad_hoc_GETS(reg_summary, 
                                        df_factors[start_date:break_date], 
                                        df_stocks.loc[start_date:break_date, name].to_frame())
                d[name] = pd.concat([d[name], final_res[0]], ignore_index = True)
                
                d[name].iloc[i,-2] = df_stocks.loc[start_date:break_date].index[0]
                d[name].iloc[i,-1] = df_stocks.loc[start_date:break_date].index[-1]
                
                
                
                
                
                
                """
                DO WE HAVE TO CHECK FOR BREAKS EVEN BEFORE THE BREAK???????????????????????????
                """
                p_val_df_2 = CHOW_TEST_FF(df_stocks.loc[:break_date, name].to_frame(),
                                            df_factors[:break_date])
                
                """
                Checking for further breaks
                """
                
                p_val_df_2 = CHOW_TEST_FF(df_stocks.loc[break_date:, name].to_frame(),
                                            df_factors[break_date:])
                    
                if p_val_df_2.empty:
                    
                    i = i + 1
                    reg_summary, reg_list = OLS(df_stocks.loc[break_date:, name].to_frame(),
                                                df_factors[break_date:], hac = True)
                    
                
                
                    final_res = ad_hoc_GETS(reg_summary, 
                                            df_factors[break_date:], 
                                            df_stocks.loc[break_date:, name].to_frame())
                    
                    d[name] = pd.concat([d[name], final_res[0]], ignore_index = True)
                    
                    d[name].iloc[i,-2] = df_stocks.loc[break_date:].index[0]
                    d[name].iloc[i, -1] = df_stocks.loc[break_date:].index[-1]
                    break_date = 0
                    
                
                elif (min(p_val_df_2[name]) > 0.05) :
                
                    i = i + 1
                    reg_summary, reg_list = OLS(df_stocks.loc[break_date:, name].to_frame(),
                                                df_factors[break_date:], hac = True)
                    
                
                
                    final_res = ad_hoc_GETS(reg_summary, 
                                            df_factors[break_date:], 
                                            df_stocks.loc[break_date:, name].to_frame())
                    
                    d[name] = pd.concat([d[name], final_res[0]], ignore_index = True)
                    
                    d[name].iloc[i,-2] = df_stocks.loc[break_date:].index[0]
                    d[name].iloc[i, -1] = df_stocks.loc[break_date:].index[-1]
                    break_date = 0
                    
                    
                else:
                    start_date = break_date
                    break_date = p_val_df_2[name].idxmin()
                
                
                i = i+1
                
        return d