import pandas as pd

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
print(lenses)
All_lensesLabel = ['age','prescript','astigmatic','tearRate','class']

data = pd.DataFrame(lenses,columns=All_lensesLabel)
print(data)
lenses_pd = data.iloc[:,0:4]
print(lenses_pd)
