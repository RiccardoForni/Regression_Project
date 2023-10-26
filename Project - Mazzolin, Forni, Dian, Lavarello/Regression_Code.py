import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
"""import scipy as sp"""
import seaborn as sns

t=pd . date_range ( start ='15-09-2013 ',end ='15-09-2023 ', freq ='M')
data = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EURIBOR_3_M")
Interest_Rate_Monthly=data[["BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"]]/3
n=np . size ( Interest_Rate_Monthly )
plt.figure()
plt . plot (t , Interest_Rate_Monthly [1: n])
plt.savefig("test.jpg")