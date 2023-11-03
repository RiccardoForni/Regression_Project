from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import statsmodels . api as sm
import pandas as pd
import os
import numpy as np
def plotbar(P,string):
    for e in P:
        plt.bar(e[0],e[1])
        plt.xticks(rotation=90)
    plt.savefig("TEST"+string+".png")

def plotCAPM(Stocks,Market,OLSResult,Subset,String):
    cwd = os.getcwd()
    folder = cwd + "/img/testCAPM/"

    if not os.path.exists(folder):
        os.mkdir(folder)
    myint=iter(Subset.columns)
    for e,OLSRes in zip(Stocks,OLSResult):
        str=next(myint).strip()
        plt.figure()
        plt.plot(Market, OLSRes.iloc[1][0]*Market+OLSRes.iloc[0][1])
        plt.scatter(Market,e)
        plt . xlabel ('Eurostoxx')
        plt . ylabel (str)
        plt.savefig("img/testCAPM/CAPM-"+str+String+".png")
        plt.close()


def plotscatter(setx,sety,title,xlabel,ylabel,sigla,Subset,string_to_save):
    cwd = os.getcwd()
    folder = cwd + "/"+string_to_save

    if not os.path.exists(folder):
        os.mkdir(folder)
        
    myint=iter(Subset.columns)
    for e in sety:
        str=next(myint)
        plt.figure()
        plt.scatter(setx,e)
        plt.title(title)
        plt . xlabel (xlabel)
        plt . ylabel (str+ylabel)
        plt.savefig(folder+"/"+sigla+"-"+str+".png")
        plt.close()
      


def OLS(Stock_Risk_Free,Market,printSummary=False):
    Res= []
    X = np . column_stack (( np . ones_like ( Market ) , Market ))
    try:
        Stock_Risk_Free.shape[1]
        for e,i in zip(Stock_Risk_Free,range(0,len(Stock_Risk_Free))):
            df = sm . OLS ( e[1:] , X[1:]  ). fit ()
            Res.append(pd.read_html(df.summary().tables[1].as_html(),header=0,index_col=0)[0])
            if printSummary:
               with open('summary'+str(i)+'.txt', 'w') as fh:
                    fh.write(df.summary().as_html())
    except:
        Res.append(pd.read_html(sm . OLS (  Stock_Risk_Free[1:], X[1:]  ). fit ().summary().tables[1].as_html(),header=0,index_col=0)[0])
        
    
    return Res

def ReorderByOLSParam(Stocks,Subset,Row_interess,Coloum_interess):
    """
    Function return stock reorder by OLS result
    Row_interess -> choose row of OLS Summary between
    Const=0, X1=1

    Coloum_interess-> choose coloum of OLS summary between
    coef=0      std err=1    t=2     P>|t|=3     0.025=4   0.975=5  
    """ 
    P = {}
    myint=iter(Subset.columns)
    for e in Stocks:
            P[next(myint)]=e.iloc[Row_interess][Coloum_interess]
    return sorted(P.items(), key=lambda x:x[1])
