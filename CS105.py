#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[110]:


df2021 = pd.read_csv('2021.csv')
df2021.columns = [c.replace(' ','_') for c in df2021.columns] 
df2021.info()


# In[111]:


# renaming columns for convenience 
df2021 = df2021.rename(columns={'Regional_indicator':'Region',
                                'Country_name':'Country',
                                'Ladder_score':'Happiness_Score',
                                'Explained_by:_Log_GDP_per_capita':'Effect_Economy',
                                'Explained_by:_Healthy_life_expectancy':'Effect_Health',
                                'Explained_by:_Perceptions_of_corruption':'Effect_Trust',
                                'Explained_by:_Social_support':'Effect_Family'})


# In[112]:


# dropping unused columns and columns with no values
df2021 = df2021.drop(columns =['upperwhisker','lowerwhisker','Standard_error_of_ladder_score'])
df2021 = df2021.drop(df2021.index[df2021['Country'] == 'Afghanistan'], inplace=False)

df2021 = df2021.dropna()


# In[113]:


# heatmap of factors affecting happiness score
corr = df2021.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot = True)


# In[114]:


# Sorted correlations descending
corr['Happiness_Score'].sort_values(ascending=False)


# In[115]:


plt.subplots(2,1,figsize=(12,14))
plt.subplot(211)
x = df2021.sort_values('Happiness_Score', ascending=True).tail(15)
plt.barh(y='Country', width='Happiness_Score', data=x, color='skyblue')
plt.xlim(xmin=7.0, xmax=8)
plt.xlabel('Happiness Score')
plt.ylabel('Country')
plt.title('15 Countries with the Highest Happiness -2021')
plt.subplot(212)
x = df2021.sort_values('Happiness_Score', ascending=False).tail(15)
plt.barh(y='Country', width='Happiness_Score', data=x, color='lightpink')
plt.xlim(xmin=2.8, xmax=4.5)
plt.xlabel('Happiness Score')
plt.ylabel('Country')
plt.title('15 Countries with the Lowest Happiness -2021')
plt.show()


# In[70]:


# boxplot Happiness Score against Region
ax=df2021.boxplot(column='Happiness_Score',by='Region',vert=False, showfliers=False)
plt.suptitle(' ')
ax.set_xlabel('Happiness Score')
ax.set_ylabel('Region')
plt.title('Happiness Score Intervals by Region')
plt.show()


# In[138]:


plt.figure(figsize=(15,5)); 
    
plt.scatter(df2021['Happiness_Score'], df2021['Effect_Economy'], color='red');
plt.xlabel("Happiness Score");
plt.ylabel("Economy");
plt.suptitle("Economy",fontsize=18)
plt.show()


# In[140]:


plt.figure(figsize=(16,5));
plt.scatter(df2021['Happiness_Score'], df2021['Effect_Family'], color='b');
plt.xlabel("Happiness Score");
plt.ylabel("Family");
plt.suptitle("FAMILY",fontsize=18)
plt.show()


# In[144]:


plt.figure(figsize=(16,8));
plt.scatter(df2021['Happiness_Score'], df2021['Effect_Health'], color='purple');
plt.xlabel("Happiness Score");
plt.ylabel("Health");
plt.suptitle("Health",fontsize=18)
plt.show()


# In[145]:


plt.figure(figsize=(16,8));
plt.scatter(df2021['Happiness_Score'], df2021['Effect_Trust'], color='orange');
plt.xlabel("Happiness Score");
plt.ylabel("Trust");
plt.suptitle("Trust",fontsize=18)

plt.show()


# In[76]:


result_1 = smf.ols(formula="Happiness_Score ~ Effect_Economy", data=df2021).fit()
result_1.summary()


# In[77]:


result_2 = smf.ols(formula="Happiness_Score ~ Effect_Family", data=df2021).fit()
result_2.summary()


# In[78]:


result_3 = smf.ols(formula="Happiness_Score ~ Effect_Health", data=df2021).fit()
result_3.summary()


# In[79]:


result_3 = smf.ols(formula="Happiness_Score ~ Effect_Trust", data=df2021).fit()
result_3.summary()


# In[87]:


region_colors = {'Middle East and North Africa':'turquoise',
                 'Latin America and Caribbean':'coral',
                 'East Asia':'darkmagenta',
                 'Sub-Saharan Africa':'springgreen',
                 'Southeast Asia':'mediumseagreen',
                 'Western Europe':'pink',
                 'North America and ANZ':'yellow',
                 'Commonwealth of Independent States':'pink',
                 'Central and Eastern Europe':'darkcyan',
                 'South Asia':'fuchsia'};


type(region_colors)
colors = []
for i in df2021['Region']:
    
    colors.append(region_colors[i])


# In[102]:


plt.clf()
plt.figure(figsize=(20,10))
plt.scatter(df2021['Effect_Economy'], df2021['Effect_Health'], s=(df2021['Happiness_Score']**4), alpha=0.5, c=colors)
plt.grid(True)

plt.xlabel("Economy")
plt.ylabel("Health")

plt.suptitle("Health Economy with sizes as Happiness score by Region", fontsize=18)

plt.show()


# In[90]:


sns.scatterplot(data=df2021,x='Happiness_Score',y='Effect_Economy',hue='Region')
plt.legend(bbox_to_anchor=(1.75, 0.5), loc='center right', borderaxespad=0)
plt.suptitle('Economy and Happiness by Region')


# In[91]:


sns.scatterplot(data=df2021,x='Happiness_Score',y='Effect_Health',hue='Region')
plt.legend(bbox_to_anchor=(1.75, 0.5), loc='center right', borderaxespad=0)
plt.suptitle('Health and Happiness by Region')


# In[123]:


sns.scatterplot(data=df2021,x='Happiness_Score',y='Effect_Family',hue='Region')
plt.legend(bbox_to_anchor=(1.75, 0.5), loc='center right', borderaxespad=0)
plt.suptitle('Family and Happiness by Region')


# In[124]:


sns.scatterplot(data=df2021,x='Happiness_Score',y='Effect_Trust',hue='Region')
plt.legend(bbox_to_anchor=(1.75, 0.5), loc='center right', borderaxespad=0)
plt.suptitle('Trust and Happiness by Region')


# In[ ]:




