
## Removing **( and )** from the obtained Edges.csv file
import pandas as pd
import re
T = pd.read_csv('Edges.csv',header=None)
temp1=[]
temp2=[]
for i in T[0]:
    i=re.sub("[(]","",i)
    temp1.append(int(i))
for j in T[1]:
    j=re.sub("[)]","",j)
    temp2.append(int(j))
T[0]=temp1
T[1]=temp2
T.head()
T.to_csv('Edges.txt',header=None, index=None, sep=' ')
