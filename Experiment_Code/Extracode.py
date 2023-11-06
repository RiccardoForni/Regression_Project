"""
i=0
for e in np.array(Subset_Clear).T:
    str=Subset_Clear.columns[i].replace(" - TOT RETURN IND","")
    plt.figure()
    plt.plot(t , (e[1:n]/e[1]))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt . xlabel ('Time - Monthly - 30-09-2013 - 30-09-2023 ')
    plt . ylabel (str+' Monthly Total Ret')
    plt.savefig("img/Total_Return/TR-"+str+".png")
    if i == 15 :
        plt.close('all')
    i+=1
plt.close('all')
"""

"""
beta = []
alpha = []
Euro1D = ArrayEuroS.squeeze()
for Stock in result :
    b_,a_= np.polyfit(Euro1D[1:n],Stock[1:n], 1)
    beta.append(b_)
    print(b_)
    alpha.append(a_)
    
beta = np.array(beta).squeeze()
alpha = np.array(alpha).squeeze()
"""

"""
i = 0
for e in result:

    str=Subset_Clear.columns[i].replace(" - TOT RETURN IND","")
    plt.figure()
    plt.plot(ArrayEuroS[1:n], beta[i]*ArrayEuroS[1:n]+alpha[i])
    plt.scatter(ArrayEuroS[1:n],e[1:n])
    plt . xlabel ('Eurostoxx')
    plt . ylabel (str)
    plt.savefig("img/testCAPM/CAPM-"+str+".png")
    i+=1

plt.close('all')
"""

"""
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
    """
