print('ANALYZING HOUR WISE CRIME DATA:')
print('\nImporting required libraries.')
print('Importing Dataset.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import calendar
import seaborn as sns
from pandas import read_csv

hour = []
for x in range(0, 25):
	hour.append(str(x))

crimes = read_csv('Data/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)
crimes.Date = pd.to_datetime(crimes.Date, format = '%m/%d/%Y %I:%M:%S %p')
crimes.index = pd.DatetimeIndex(crimes.Date)
crimes.insert(3, 'Hour', 'NULL')
crimes['Hour'] = crimes.index.hour

print('\nIMPLEMENTING K-MEAN CLUSTERING ALGORITHM')
print('We are clustering the different hours of a day based on the number of crimes commited in the particular hour.')
print('\nPreprocessing data.')
Hour_count = pd.DataFrame(crimes.groupby('Hour').size().sort_values(ascending=False).rename('Counts').reset_index())

print('\nFitting the data in KMeans algorithm.')
kmean = KMeans(n_clusters = 3)
kmean.fit(Hour_count)
pr = kmean.predict(Hour_count)
print(pr)
centers = kmean.cluster_centers_
print('\nPlotting graph...')

#fig1, ax1 = plt.subplots(figsize=(12, 14))
sns.set(style="darkgrid")
plt.figure(figsize=(8,6))
plt.scatter(Hour_count['Hour'], Hour_count['Counts'], c=pr, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Hours')
plt.ylabel('Number of Crimes')
plt.title('K-Means Clustering ')
s = Hour_count['Hour'].sort_values(ascending=True)
plt.xticks(s, hour)
x = input('Press ENTER to view graph.')
plt.show()

cluster_map = pd.DataFrame()
cluster_map['data_index'] = Hour_count['Hour']
cluster_map['cluster'] = kmean.labels_
cluster_map.to_csv('Data/cluster.csv')

l0 = []
l1 = []
l2 = []
li1 = cluster_map['cluster'].tolist()
li2 = cluster_map['data_index'].tolist()

for i in range(24):
	if li1[i] == 0:
		l0.append(li2[i])

	if li1[i] == 1:
		l1.append(li2[i])

	if li1[i] == 2:
		l2.append(li2[i])

print(l0)
print(l1)
print(l2)

crimes.insert(3, 'Value', 'NULL')
for i in l0:
	crimes.loc[crimes['Hour'] == i, ['Value']] = 'a'

for i in l1:
	crimes.loc[crimes['Hour'] == i, ['Value']] = 'b'

for i in l2:
	crimes.loc[crimes['Hour'] == i, ['Value']] = 'c'

cr_a = crimes.loc[crimes['Value'] == 'a', ['Primary Type', 'Arrest', 'Domestic', 'Location Description', 'Location']]
cr_b = crimes.loc[crimes['Value'] == 'b', ['Primary Type', 'Arrest', 'Domestic', 'Location Description', 'Location']]
cr_c = crimes.loc[crimes['Value'] == 'c', ['Primary Type', 'Arrest', 'Domestic', 'Location Description', 'Location']]

a_pt = pd.DataFrame(cr_a.groupby('Location Description').size().sort_values(ascending=False).rename('count').reset_index())
a_pt = a_pt.ix[:5,:]

b_pt = pd.DataFrame(cr_b.groupby('Location Description').size().sort_values(ascending=False).rename('count').reset_index())
b_pt = b_pt.ix[:5,:]

c_pt = pd.DataFrame(cr_c.groupby('Location Description').size().sort_values(ascending=False).rename('count').reset_index())
c_pt = c_pt.ix[:5,:]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), sharex = True)
fig.suptitle('Top 5 locations', fontsize = 15)

print(a_pt)
print(b_pt)
print(c_pt)

sns.set_color_codes("dark")
sns.barplot(x="Location Description", y="count", data=a_pt, label="Location", palette = 'Dark2', ax = ax1)
ax1.set(ylabel="Crimes", xlabel="Location")
ax1.axhline(0, color="k", clip_on=False)
ti1 = 'Hours: '+ str(l0)
ax1.set_title(ti1)

sns.set_color_codes("dark")
sns.barplot(x="Location Description", y="count", data=b_pt, label="Location", palette = 'Dark2', ax = ax2)
ax2.set(ylabel="Crimes", xlabel="Location")
ax2.axhline(0, color="k", clip_on=False)
ti2 = 'Hours: '+ str(l1)
ax2.set_title(ti2)


sns.set_color_codes("dark")
sns.barplot(x="Location Description", y="count", data=c_pt, label="Location", palette = 'Dark2', ax = ax3)
ax3.set(ylabel="Crimes", xlabel="Location")
ax3.axhline(0, color="k", clip_on=False)
ti3 = 'Hours: '+str(l2)
ax3.set_title(ti3)

sns.despine(bottom=True)
plt.setp(plt.setp(ax1.get_xticklabels(), rotation=45))
plt.setp(ax2.get_xticklabels(), rotation=45)
plt.setp(ax3.get_xticklabels(), rotation=45)
x = input('press ENTER to view the graph.')
plt.show()

#3rd plot
labels = ['False', 'True']
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), sharex = True)
fig2.suptitle('Percentage of Arrests', fontsize = 15)

arrest_count = pd.DataFrame(cr_a.groupby('Arrest').size().sort_values(ascending=False).rename('count').reset_index())
sizes = arrest_count['count'].tolist()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90) 
ax1.set_title(ti1)

arrest_count = pd.DataFrame(cr_b.groupby('Arrest').size().sort_values(ascending=False).rename('count').reset_index())
sizes = arrest_count['count'].tolist()
ax2.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax2.set_title(ti2)

arrest_count = pd.DataFrame(cr_c.groupby('Arrest').size().sort_values(ascending=False).rename('count').reset_index())
sizes = arrest_count['count'].tolist()
ax3.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax3.set_title(ti3)

x = input('press ENTER to view pie chart.')
plt.setp(fig2.axes, yticks=[])
plt.show()

x = input('Press any key to exit.')
