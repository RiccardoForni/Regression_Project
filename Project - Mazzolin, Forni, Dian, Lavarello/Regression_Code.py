import numpy as np
import pandas as pd
import Regre_Function as rf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


t=pd . date_range ( start ='15-09-2013 ',end ='15-09-2023 ', freq ='M') #Date series

"""
Retrieve Subset equity and Interest rate
"""
sheet=-1
col=""
while sheet < 0 or sheet>1:
    sheet=int(input("Which Risk Free rate you want? press 0 for Euribor or 1 for Bund "))

match sheet:
    case 0:
         sheet = "EURIBOR_3_M"
         col="BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"
    case 1:
        sheet = "BUND"
        col="RF GERMANY GVT BMK BID YLD 3M - RED. YIELD"
        

EuroStoxx = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600")
Subset_Stock_Selected= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")
Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name=sheet)


Subset_Stock_Selected.columns = Subset_Stock_Selected.columns.str.replace("- TOT RETURN IND","")
EuroStoxx = EuroStoxx.loc[:, EuroStoxx.columns != 'Name']#Delete column of date
Subset_Stock_Selected=Subset_Stock_Selected.loc[:, Subset_Stock_Selected.columns != 'Name'] #Delete column of date

RFREE=np.array(Interest[[col]]/12)
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
rMarket =np.subtract(Market,RFREE)

rf.plotscatter(Market,rStock,"Excess Returns vs Eurostoxx -"+sheet,
            "Time - Monthly - 30-09-2013 - 30-09-2023",
            "",sheet,Subset_Stock_Selected,
            "Excess_return"
            )

Res_Euribor= rf.OLS(rStock,rMarket,False)
P_sort=rf.ReorderByOLSParam(Res_Euribor,Subset_Stock_Selected,0,3)
rf.plotbar(P_sort,sheet)
rf.plotCAPM(rStock,rMarket,Res_Euribor,Subset_Stock_Selected,sheet)
Excess_equi_valued=sum(rStock)/len(rStock)
Excess_OLS=rf.OLS(Excess_equi_valued,rMarket)
