import pandas as pd
from pandas import Series, DataFrame
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

titanic_df = pd.read_csv('train.csv')

print(titanic_df.info())

##################################################
#### WHO IS THE PASSENGERS? #####################

sns.factorplot('Sex', data=titanic_df, kind="count")  ##factorplot gives info about column
#plt.show()

sns.factorplot('Sex', data=titanic_df, hue='Pclass', kind='count')
#plt.show()

sns.factorplot('Pclass', data=titanic_df, hue='Sex', kind='count')
#plt.show()

def male_female_child(passenger):
    age,sex = passenger

    if age<16:
        return 'child'
    else:
        return sex

titanic_df['person']=titanic_df[['Age','Sex']].apply(male_female_child, axis=1) #created new column

sns.factorplot('Pclass', data=titanic_df, hue='person', kind='count')
#plt.show()

titanic_df['Age'].hist(bins=70)
#plt.show()

print(titanic_df['Age'].mean())
print(titanic_df['person'].value_counts())

fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)  #FaceGrid allows mult. plots on one figure
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()



fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)  #FaceGrid allows mult. plots on one figure
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()



fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)  #FaceGrid allows mult. plots on one figure
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
#plt.show()

deck = titanic_df['Cabin'].dropna()

levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin', data=cabin_df, palette='winter_d', kind='count')
#plt.show()

cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.factorplot('Cabin', data=cabin_df, palette='summer', kind='count')
#plt.show()

###############################################################################
####### WHERE DID PASSENGERS COME FROM? ###########################
sns.factorplot('Embarked',data=titanic_df,hue='Pclass', kind='count')
#plt.show()


#############################################################################
###### WHO WAS ALONE AND WHO WAS WITH FAMILY? ############################

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch    #define alone people on the ship(siblings + parents)

titanic_df['Alone'].loc[titanic_df['Alone'] > 0 ] = 'With Family'  
titanic_df['Alone'].loc[titanic_df['Alone'] == 0 ] = 'Alone'

sns.factorplot('Alone', data=titanic_df, palette='Blues', kind='count')
#plt.show()


######################################################################################
####### Which Factors Helped the Survived? #################################

titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})

sns.factorplot('Survivor', data=titanic_df, hue='Pclass',kind='count')
#plt.show()

sns.factorplot('Pclass','Survived', data=titanic_df) #ratio of survived according to their classes
#plt.show()

sns.factorplot('Pclass','Survived', hue='person',data=titanic_df) #ratio of survived according to sex
#plt.show()

generations = [10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df, palette='winter',x_bins=generations) #ratio of survived according to class and age
#plt.show()

sns.lmplot('Age','Survived',hue='Sex',data=titanic_df, palette='winter',x_bins=generations) #ratio of survived according to sex and age
plt.show()
