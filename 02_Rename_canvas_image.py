
import os
import shutil
import pandas as pd
path_for_excel = "GVSS_Vision_Data/Annotation.xlsx"
P = sorted(os.listdir("Canvas_dataset"))

os.makedirs("Canvas_renamed_resized", exist_ok=True)
P = [i for i in P if i[-1]=="g"]

dfd = pd.read_excel(path_for_excel) 
excel = dfd['House_Price']
price = [str(i) for i in excel]



new_name = [P[i][:-4]+'_'+price[i]+'.png' for i in range(499)]

for i in range(499):
    shutil.copy("Canvas_dataset/"+P[i],"Canvas_renamed_resized/"+new_name[i])




