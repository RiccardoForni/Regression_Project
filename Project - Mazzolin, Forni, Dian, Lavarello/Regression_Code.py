import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import statsmodels . api as sm
import os

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
      


def OLS_Pvalue(Stock_Risk_Free,Market,Subset):
    Res = []
    P = {}
    X = np . column_stack (( np . ones_like ( Market ) , Market ))
    myint=iter(Subset.columns)
    for e in Stock_Risk_Free:
        Res.append(sm . OLS ( e[1:] , X[1:]  ). fit ())
        P[next(myint)]=Res[-1].pvalues[0]
        """
        with open('summary'+str(i)+'.txt', 'w') as fh:
            fh.write(Res1[-1].summary().as_text())
        
        i+=1
        """
     
    return Res,sorted(P.items(), key=lambda x:x[1])



t=pd . date_range ( start ='15-09-2013 ',end ='15-09-2023 ', freq ='M') #Date series

"""
Retrieve Subset equity and Interest rate
"""
EuroStoxx = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600")
Subset_Stock_Selected= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")
Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EURIBOR_3_M")
Interest_BUND = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="BUND")



Subset_Stock_Selected.columns = Subset_Stock_Selected.columns.str.replace("- TOT RETURN IND","")
EuroStoxx = EuroStoxx.loc[:, EuroStoxx.columns != 'Name']#Delete column of date
Subset_Stock_Selected=Subset_Stock_Selected.loc[:, Subset_Stock_Selected.columns != 'Name'] #Delete column of date

RFREE=np.array(Interest[["BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"]]/12)
BUNDRISK=np.array(Interest_BUND[['RF GERMANY GVT BMK BID YLD 3M - RED. YIELD']]/12)
"""
Clear from Date
and
calculate Log operation 
"""
Market =  np.array(100 *( np . log ( EuroStoxx ) -np . log ( EuroStoxx . shift (1) )) )
Equities = np.array(100 *( np . log ( Subset_Stock_Selected ) -np . log ( Subset_Stock_Selected . shift (1) ))) #Log operation
"""
Calculate for each equity: Equity-interest rate

The result is a matrix [Number of equity][Value calculated]
"""
rStock = np.subtract(Equities,RFREE).T #Subtract operation filled inside an array 
rStock_Bond =np.subtract(Equities,BUNDRISK).T

rMarket =np.subtract(Market,RFREE)
rMarket_Bond =np.subtract(Market,BUNDRISK)

plotscatter(Market,rStock,"Excess Returns vs Eurostoxx - 3M Euribor",
            "Time - Monthly - 30-09-2013 - 30-09-2023",
            "","ER",Subset_Stock_Selected,
            "Excess_return"
            )

plotscatter(Market,rStock_Bond,"Excess Returns vs Eurostoxx - 3M BUND",
            "Time - Monthly - 30-09-2013 - 30-09-2023",
            "","ERB",Subset_Stock_Selected,
            "Excess_return"
            )


Res,D_sort = OLS_Pvalue(rStock,rMarket,Subset_Stock_Selected)
Res_bund,D_sort= OLS_Pvalue(rStock_Bond,rMarket,Subset_Stock_Selected)
for e in D_sort:
    plt.bar(e[0],e[1])
plt.savefig("TEST.png")
