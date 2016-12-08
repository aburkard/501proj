# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:52:25 2016

@author: caryt
"""



#Cary Lou
#ANLY501
#Project part 2
#11/2/2016
import pandas  as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sklearn
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import fim

from datetime import datetime
from scipy.stats.stats import pearsonr

from pprint import pprint

def main(argv):
    explorechidata('../data/ChicagoCrime2016W3newfeatures.tsv','../data/chidata2016cleaned.tsv','../data/dataexploration.txt','resources/ChicagoDataCheckVals.csv')
    plotchidata('../data/chidata2016cleaned.tsv', '../data/plotoutcomes.txt')
    clusterchidata('../data/chidata2016cleaned.tsv','../data/clusteroutcomes.txt')
    assocruleschidata('../data/chidata2016cleaned.tsv','../data/assocoutcomes.txt')
    predictchidata('../data/chidata2016cleaned.tsv','../data/predictoutcomes.txt')

def explorechidata(data, cleandata, out, valcheck):
    #For use as function
    with open(data, 'r') as file:
    #For use in interactive exploration
    #with open("ChicagoCrime2016W3newfeatures.tsv", 'r') as file:
        myData = pd.read_table(file , sep='\t', encoding='latin1')
        myData = myData.drop('Unnamed: 0',axis=1)
        #WRITE RESULTS OUT TO FILE:
        with open(out, 'w') as outfile:

            #Create binned variable and additional date and time variables for exploration
            #Bin numeric var-- bin latitude variable to get a sense of how crimes vary by hwo north/south they are in the city
                    #use equiwidth bins to show variation by uniform geographic unit
            myData['latitude10eqwibins'] = pd.cut(myData['latitude'], 10)
            #explore frequency by bin-- interestingly, there seem to be more crimes reported in the mid-south and mid-north latitudes of the city
            #could be interesting to explore this by type of crime
            print(myData.groupby('latitude10eqwibins').size())
#            outfile.write("Frequency counts for new binned categorical variable of latitude (north-south):/n")
#            outfile.write(str(myData.groupby('latitude10eqwibins').size()))
#            outfile.write("\n")
            #not hugely different splits for more serious index crimes... hmmm, look up the high crime areas on map-- prob just high pop denstiy there too..
            print(myData.groupby(['latitude10eqwibins','indexCrime']).size())

            #Create additional variables used in analysis
            #Create numeric version of string date/time variable
            def date_time_num(ts):
                return datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.000')
            myData['date_time_numeric'] = myData['date'].apply(date_time_num)
            #pull hour
            def time_func(ts):
                    return ts.hour + ts.minute/60
            myData['time_crime'] = myData['date_time_numeric'].apply(time_func)
            #pull day of week
            def week_day_func(ts):
                    return ts.isoweekday()
            myData['day_of_week'] = myData['date_time_numeric'].apply(week_day_func)
            #pull time withing day of week
            def week_day_time_func(ts):
                    return ts.isoweekday() + ts.hour/24 + ts.minute/1440
            myData['day_of_week_time'] = myData['date_time_numeric'].apply(week_day_time_func)


            #descriptive stats
            print(myData.info())
            outfile.write("The variables, data types, and non-missing values are:\n")
            outfile.write(str(myData.info()))
            outfile.write("The number of records in the dataset is:" + str(len(myData['date'])) +"\n")
            outfile.write("\n")
            x= myData.info()
            outfile.write(str(x))
            #Quant vars
            print(myData.describe())
            print(myData[1:10])
            outfile.write("The summary statistics of the data are:\n")
            outfile.write(str(myData.describe()))
            outfile.write("\n")


            #Counts of categorical vars
            print(myData.groupby('beat').size())
            print(myData.groupby('community_area').size())
            print(myData.groupby('district').size())
            print(myData.groupby('ward').size())
            print(myData.groupby('arrest').size())
            print(myData.groupby('description').size())
            print(myData.groupby('fbi_code').size())
            print(myData.groupby('iucr').size())
            print(myData.groupby('location_description').size())
            print(myData.groupby('location_type').size())
            print(myData.groupby('primary_type').size())
            print(myData.groupby('domesticArrest').size())
            print(myData.groupby('indexCrime').size())
            print(myData.groupby('recentCrime').size())
            print(myData.groupby('day_of_week').size())
            print(myData.groupby('latitude10eqwibins').size())

            outfile.write("The medians are:\n")
            outfile.write(str(myData['latitude'].median())+"\n")
            outfile.write(str(myData['longitude'].median())+"\n")
            outfile.write(str(myData['location_1'].median())+"\n")
            outfile.write(str(myData['location_2'].median())+"\n")
            outfile.write(str(myData['x_coord'].median())+"\n")
            outfile.write(str(myData['y_coord'].median())+"\n")

            outfile.write("The frequency counts of categorical variables are:\n")
            outfile.write(str(myData['beat'].mode())+"\n")
            outfile.write(str(myData['community_area'].mode())+"\n")
            outfile.write(str(myData['ward'].mode())+"\n")
            outfile.write(str(myData['arrest'].mode())+"\n")
            outfile.write(str(myData['description'].mode())+"\n")
            outfile.write(str(myData['fbi_code'].mode())+"\n")
            outfile.write(str(myData['iucr'].mode())+"\n")
            outfile.write(str(myData['location_description'].mode())+"\n")
            outfile.write(str(myData['location_type'].mode())+"\n")
            outfile.write(str(myData['primary_type'].mode())+"\n")
            outfile.write(str(myData['domesticArrest'].mode())+"\n")
            outfile.write(str(myData['indexCrime'].mode())+"\n")
            outfile.write(str(myData['recentCrime'].mode())+"\n")
            outfile.write(str(myData['day_of_week'].mode())+"\n")
            outfile.write(str(myData['latitude10eqwibins'].mode())+"\n")

            outfile.write(str(myData.groupby('community_area').size()))
            outfile.write(str(myData.groupby('district').size()))
            outfile.write(str(myData.groupby('ward').size()))
            outfile.write(str(myData.groupby('arrest').size()))
            outfile.write(str(myData.groupby('description').size()))
            outfile.write(str(myData.groupby('fbi_code').size()))
            outfile.write(str(myData.groupby('iucr').size()))
            outfile.write(str(myData.groupby('location_description').size()))
            outfile.write(str(myData.groupby('location_type').size()))
            outfile.write(str(myData.groupby('primary_type').size()))
            outfile.write(str(myData.groupby('domesticArrest').size()))
            outfile.write(str(myData.groupby('indexCrime').size()))
            outfile.write(str(myData.groupby('recentCrime').size()))
            outfile.write(str(myData.groupby('day_of_week').size()))
            outfile.write(str(myData.groupby('latitude10eqwibins').size()))
            outfile.write("\n")

            outfile.write(str(myData.groupby('beat').size()))
            outfile.write(str(myData.groupby('community_area').size()))
            outfile.write(str(myData.groupby('district').size()))
            outfile.write(str(myData.groupby('ward').size()))
            outfile.write(str(myData.groupby('arrest').size()))
            outfile.write(str(myData.groupby('description').size()))
            outfile.write(str(myData.groupby('fbi_code').size()))
            outfile.write(str(myData.groupby('iucr').size()))
            outfile.write(str(myData.groupby('location_description').size()))
            outfile.write(str(myData.groupby('location_type').size()))
            outfile.write(str(myData.groupby('primary_type').size()))
            outfile.write(str(myData.groupby('domesticArrest').size()))
            outfile.write(str(myData.groupby('indexCrime').size()))
            outfile.write(str(myData.groupby('recentCrime').size()))
            outfile.write(str(myData.groupby('day_of_week').size()))
            outfile.write(str(myData.groupby('latitude10eqwibins').size()))
            outfile.write("\n")
            ##DATA QUALITY
            #Outliers
                #prob don't need to remove-- explain why in write-up; all interesting to analyze

            #Missing values and noise values cleaning when possible:
                #mostly location-- use mean/mode of same block to fill in
            #for function version, pull in check value list to replace noise values:
            checkvals = pd.read_csv(valcheck)
            #for interactive testing
            #checkvals = pd.read_csv('ChicagoDataCheckVals.csv')
            checkframe1 = myData['beat'].isin(checkvals['beat'])
            checkframe2 = myData['iucr'].isin(checkvals['iucr'])

            # Counts of missing and noise values for variables that are to be fixed
            record_count = 0

            location_description_missing = 0
            longitude_missing = 0
            latitude_missing = 0
            location_type_missing = 0
            location_1_missing = 0
            location_2_missing = 0
            x_coord_missing = 0
            y_coord_missing = 0

            beatnoisevals = 0
            iucrnoisevals = 0
            #replace missing or noise values
            for i in range(len(myData)):
                record_count += 1
                if  str(myData.ix[i,'location_description']) == "nan" :
                    location_description_missing += 1
                    subset = myData.loc[myData['block'] == myData.ix[i,'block'] ]
                    subset = myData.loc[myData['description'] == myData.ix[i,'description']]
                    #print(subset['location_description'].mode())
                    loc_descmode = subset['location_description'].mode()
                    #print(loc_descmode)
                    if loc_descmode.empty :
                        x = np.nan
                    else: x = loc_descmode.iloc[0]
                    myData.ix[i,'location_description'] = x
                if np.isnan(myData.ix[i,'longitude']) :
                    longitude_missing += 1
                    latitude_missing += 1
                    location_type_missing += 1
                    location_1_missing += 1
                    location_2_missing += 1
                    x_coord_missing += 1
                    y_coord_missing += 1
                    subset = myData.loc[myData['block'] == myData.ix[i,'block']]
                    myData.ix[i,'longitude'] = subset['longitude'].mean()
                    myData.ix[i,'latitude'] = subset['latitude'].mean()
                    myData.ix[i,'location_type'] = "Point"
                    myData.ix[i,'location_1'] = subset['location_1'].mean()
                    myData.ix[i,'location_2'] = subset['location_2'].mean()
                    myData.ix[i,'x_coord'] = subset['x_coord'].mean()
                    myData.ix[i,'y_coord'] = subset['y_coord'].mean()

                if checkframe1[i] == False:
                    beatnoisevals += 1
                    subset = myData.loc[myData['block'] == myData.ix[i,'block']]
                    #print( subset['beat'].mode() )
                    beatmode = subset['beat'].mode()
                    #print(beatmode)
                    if beatmode.empty :
                        x = np.nan
                    else: x = beatmode.iloc[0]
                    myData.ix[i,'beat'] = x
                if checkframe2[i] == False:
                    iucrnoisevals += 1
                    subset = myData.loc[myData['description'] == myData.ix[i,'description']]
                    #print( subset['iucr'].mode() )
                    iucrmode = subset['iucr'].mode()
                    #print(loc_descmode)
                    if iucrmode.empty :
                        x = np.nan
                    else: x = iucrmode.iloc[0]
                    myData.ix[i,'iucr'] = x

            outfile.write("The variables, data types, and non-missing values after cleaning are:\n")
            outfile.write(str(myData.info()))
            outfile.write("\n")
            outfile.write("Compared to "+ str(location_description_missing)+ " missing values for location_description out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(longitude_missing)+ " missing values for longitude out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(latitude_missing)+ " missing values for latitude out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(location_type_missing)+ " missing values for location_type out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(location_1_missing)+ " missing values for location_1 out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(location_2_missing)+ " missing values for location_2 out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(x_coord_missing)+ " missing values for x_coord out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(y_coord_missing)+ " missing values for y_coord out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(beatnoisevals)+ " noise values for beat out of "+str(record_count)+" total records.\n")
            outfile.write("Compared to "+ str(iucrnoisevals)+ " noise values for iucr out of "+str(record_count)+" total records.\n")
            outfile.write("\n")


            print(myData.info())
            #Quant vars
            print(myData.describe())
            print(myData[1:10])
        #export cleaned data
            myData.to_csv(cleandata,sep='\t')

        outfile.closed
    file.closed


def plotchidata(data, out):
    #For use as function
    with open(data, 'r') as file:
    #For use in interactive exploration
    #with open("chidata2016cleaned.tsv", 'r') as file:
        myData = pd.read_table(file , sep='\t', encoding='latin1')
        myData = myData.drop('Unnamed: 0',axis=1)
        #myData.info()
        #WRITE RESULTS OUT TO FILE:
        with open(out, 'w') as outfile:
            print(myData.info())
            #Quant vars
            print(myData.describe())
            print(myData[1:10])

            #histograms
            #hist of latitude
            plt.figure(figsize=(15, 15))
            myData['latitude'].hist()
            #pl.suptitle("Histogram latitude")
            #plt.show()
            plt.savefig('../data/images/histlat.png')
            plt.clf()
            plt.close()
            #hist of longitude
            plt.figure(figsize=(15, 15))
            myData['longitude'].hist()
            #pl.suptitle("Histogram longitude")
            #plt.show()
            plt.savefig('../data/images/histlong.png')
            plt.clf()
            plt.close()
            #Perhaps pull out times and show hist
            plt.figure(figsize=(15, 15))
            myData['time_crime'].hist()
            #pl.suptitle("Histogram time of crime")
            #plt.show()
            plt.savefig('../data/images/histtime.png')
            plt.clf()
            plt.close()
            #Perhaps pull out day of week and show dist
            plt.figure(figsize=(15, 15))
            myData['day_of_week'].hist()
            #pl.suptitle("Histogram Day of Week")
            #plt.show()
            plt.savefig('../data/images/histweekday.png')
            plt.clf()
            plt.close()


            #correlation w/ p-values-- longitude, latitude, and time of crime
            selection = myData[['longitude','latitude','time_crime','day_of_week']].dropna()
            print(pearsonr(selection['longitude'],selection['latitude']))
            print(pearsonr(selection['latitude'],selection['time_crime']))
            print(pearsonr(selection['longitude'],selection['time_crime']))
            print(pearsonr(selection['day_of_week'],selection['time_crime']))
            print(pearsonr(selection['day_of_week'],selection['latitude']))
            print(pearsonr(selection['day_of_week'],selection['longitude']))
            outfile.write("Corrleation for longitude and latitude:\n")
            outfile.write(str(pearsonr(selection['longitude'],selection['latitude'])))
            outfile.write("\n")
            outfile.write("Corrleation for longitude and time of crime:\n")
            outfile.write(str(pearsonr(selection['longitude'],selection['time_crime'])))
            outfile.write("\n")
            outfile.write("Corrleation for latitude and time of crime:\n")
            outfile.write(str(pearsonr(selection['latitude'],selection['time_crime'])))
            outfile.write("\n")
            outfile.write("Corrleation for day and time of crime:\n")
            outfile.write(str(pearsonr(selection['day_of_week'],selection['time_crime'])))
            outfile.write("\n")
            outfile.write("Corrleation for day of crime and longitude:\n")
            outfile.write(str(pearsonr(selection['day_of_week'],selection['longitude'])))
            outfile.write("\n")
            outfile.write("Corrleation for day of crime and latitude:\n")
            outfile.write(str(pearsonr(selection['day_of_week'],selection['latitude'])))
            outfile.write("\n")
            #Scatters
            #longtiude by latitude
            plt.figure(figsize=(15, 15))
            plt.scatter(myData['longitude'], myData['latitude'],s=0.1)
            #plt.show()
            plt.savefig('../data/images/../data/images/longlat.png')
            plt.clf()
            plt.close()


            plt.figure(figsize=(15, 15))
            plt.scatter(myData['latitude'], myData['day_of_week_time'],s=0.01)
            plt.savefig('../data/images/../data/images/latdaytime.png')
            plt.clf()
            plt.close()

            plt.figure(figsize=(15, 15))
            plt.scatter(myData['longitude'], myData['time_crime'],s=0.01)
            plt.savefig('../data/images/longtime.png')
            plt.clf()
            plt.close()

            #w/ density
            # Calculate the point density
    #        longlat = [myData['longitude'].dropna(), myData['latitude'].dropna()]
    #        xy = np.vstack(longlat)
    #        z = gaussian_kde(xy)(xy)
    #
    #        fig, ax = plt.subplots()
    #        ax.scatter(myData['longitude'], myData['latitude'], c=z, s=1, edgecolor='')
    #        plt.show()

        outfile.closed

    file.closed

def clusterchidata(data, out):
    #For use as function
    with open(data, 'r') as file:
    #For use in interactive exploration
    #with open("chidata2016cleaned.tsv", 'r') as file:
        myData = pd.read_table(file , sep='\t', encoding='latin1')
        myData = myData.drop('Unnamed: 0',axis=1)
        print(myData.info())
        #Quant vars
        print(myData.describe())
        print(myData[1:10])

        #Results file
        with open(out, 'w') as outfile:
            # Remove missing data
            myDataNoNA = myData.dropna()
            myDataNoNA = myDataNoNA.reset_index(drop=True)
            print(len(myDataNoNA.index))

            ## Fix MyData
            #print(myData)
            myData2=pd.concat([myDataNoNA['longitude'], myDataNoNA['latitude'], myDataNoNA['arrest'], myDataNoNA['domestic'],
                              myDataNoNA['time_crime'],  myDataNoNA['day_of_week'],myDataNoNA['indexCrime'] ],
                             axis=1, keys=['longitude', 'latitude', 'arrest', 'domestic', 'time_crime', 'day_of_week', 'indexCrime' ])
            print(myData2[1:10])
            x = myData2.values #returns a numpy array
            print(x[1:10])
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            print(x_scaled[1:10])
            normalizedDataFrame = pd.DataFrame(x_scaled)
            print(normalizedDataFrame[:10])

            #KMEANS

            # Create clusters with different Ks
            #3 clusters
            k = 3
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(normalizedDataFrame)

            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            pprint(labels[1:10])
            pprint(centroids)

            outfile.write("Centroids for k=3: \n")
            outfile.write(str(centroids))
            outfile.write("\n")

            prediction = kmeans.predict(normalizedDataFrame)
            pprint(prediction)

            # See how it fits data on different dimensions
            print(pd.crosstab(labels, myDataNoNA['longitude']))
            print(pd.crosstab(labels, myDataNoNA['latitude']))
            print(pd.crosstab(labels, myDataNoNA['arrest']))
            print(pd.crosstab(labels, myDataNoNA['domestic']))
            print(pd.crosstab(labels, myDataNoNA['time_crime']))
            print(pd.crosstab(labels, myDataNoNA['day_of_week']))
            print(pd.crosstab(labels, myDataNoNA['indexCrime']))

            outfile.write(str(pd.crosstab(labels, myDataNoNA['longitude'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['longitude'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['arrest'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['domestic'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['time_crime'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['day_of_week'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['indexCrime'])))
            outfile.write("\n")

            #####
            # PCA
            # Let's convert our high dimensional data to 2 dimensions
            # using PCA
            pca2D = decomposition.PCA(2)

            # Turn the data into two columns with PCA
            plot_columns = pca2D.fit_transform(normalizedDataFrame)

            # Plot using a scatter plot and shade by cluster label
            plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
            plt.savefig('../data/images/kmeans3pca2.png')
            plt.clf()
            plt.close()

            # Plot geographically using a scatter plot and shade by cluster
            plt.scatter(x=myDataNoNA['longitude'], y=myDataNoNA['latitude'], c=labels, s= 1, lw=0)
            plt.savefig('../data/images/kmeans3latlong.png')
            plt.clf()
            plt.close()

            #8 clusters
            # Create clusters with different Ks
            k = 8
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(normalizedDataFrame)

            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            pprint(labels[1:10])
            pprint(centroids)

            outfile.write("Centroids for k=8: \n")
            outfile.write(str(centroids))
            outfile.write("\n")

            prediction = kmeans.predict(normalizedDataFrame)
            pprint(prediction)

            # See how it fits data on different dimensions
            print(pd.crosstab(labels, myDataNoNA['longitude']))
            print(pd.crosstab(labels, myDataNoNA['latitude']))
            print(pd.crosstab(labels, myDataNoNA['arrest']))
            print(pd.crosstab(labels, myDataNoNA['domestic']))
            print(pd.crosstab(labels, myDataNoNA['time_crime']))
            print(pd.crosstab(labels, myDataNoNA['day_of_week']))
            print(pd.crosstab(labels, myDataNoNA['indexCrime']))

            outfile.write(str(pd.crosstab(labels, myDataNoNA['longitude'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['longitude'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['arrest'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['domestic'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['time_crime'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['day_of_week'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['indexCrime'])))
            outfile.write("\n")

            #####
            # PCA
            # Plot using a scatter plot and shade by cluster label
            plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
            plt.savefig('../data/images/kmeans8pca2.png')
            plt.clf()
            plt.close()

            # Plot using a scatter plot and shade by cluster label geographically
            plt.scatter(x=myDataNoNA['longitude'], y=myDataNoNA['latitude'], c=labels, s= 1, lw=0)
            plt.savefig('../data/images/kmeans8latlong.png')
            plt.clf()
            plt.close()

#            ##CLUSTER USING ONLY GEOGRAPHIC DATA
#            ## Fix MyData
#            #print(myData)
#            myData2=pd.concat([myDataNoNA['longitude'], myDataNoNA['latitude']],
#                             axis=1, keys=['longitude', 'latitude' ])
#            print(myData2[1:10])
#            x = myData2.values #returns a numpy array
#            print(x[1:10])
#            min_max_scaler = preprocessing.MinMaxScaler()
#            x_scaled = min_max_scaler.fit_transform(x)
#            print(x_scaled[1:10])
#            normalizedDataFrame = pd.DataFrame(x_scaled)
#            print(normalizedDataFrame[:10])
#
#            #KMEANS
#
#            # Create clusters with different Ks
#            #3 clusters
#            k = 3
#            kmeans = KMeans(n_clusters=k)
#            kmeans.fit(normalizedDataFrame)
#
#            labels = kmeans.labels_
#            centroids = kmeans.cluster_centers_
#            pprint(labels[1:10])
#            pprint(centroids)
#
#            prediction = kmeans.predict(normalizedDataFrame)
#            pprint(prediction)
#
#            # See how it fits data on different dimensions
#            print(pd.crosstab(labels, myDataNoNA['longitude']))
#            print(pd.crosstab(labels, myDataNoNA['latitude']))
#            # Plot using a scatter plot and shade by cluster label geographically
#            plt.scatter(x=myDataNoNA['longitude'], y=myDataNoNA['latitude'], c=labels, s= 1, lw=0)
#            plt.show()
#
#            #12 clusters
#            k = 12
#            kmeans = KMeans(n_clusters=k)
#            kmeans.fit(normalizedDataFrame)
#
#            labels = kmeans.labels_
#            centroids = kmeans.cluster_centers_
#            pprint(labels[1:10])
#            pprint(centroids)
#
#            prediction = kmeans.predict(normalizedDataFrame)
#            pprint(prediction)
#
#            # See how it fits data on different dimensions
#            print(pd.crosstab(labels, myDataNoNA['longitude']))
#            print(pd.crosstab(labels, myDataNoNA['latitude']))
#            # Plot using a scatter plot and shade by cluster label geographically
#            plt.scatter(x=myDataNoNA['longitude'], y=myDataNoNA['latitude'], c=labels, s= 1, lw=0)
#            plt.show()


    #Other Algorithms
            ## Fix MyData-- smaller sample to deal with memory limitations
            myDataNoNA = myDataNoNA[myDataNoNA['recentCrime'] == 1]
            myDataNoNA = myDataNoNA[1:10000]
            print(len(myDataNoNA))
            myData2=pd.concat([myDataNoNA['longitude'], myDataNoNA['latitude'], myDataNoNA['arrest'], myDataNoNA['domestic'],
                              myDataNoNA['time_crime'],  myDataNoNA['day_of_week'],myDataNoNA['indexCrime'] ],
                             axis=1, keys=['longitude', 'latitude', 'arrest', 'domestic', 'time_crime', 'day_of_week', 'indexCrime' ])
            print(myData2[1:10])
            x = myData2.values #returns a numpy array
            print(x[1:10])
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            print(x_scaled[1:10])
            normalizedDataFrame = pd.DataFrame(x_scaled)
            print(normalizedDataFrame[:10])

            #DBSCAN
            dbscan = DBSCAN()
            dbscan.fit(normalizedDataFrame)

            labels = dbscan.labels_.tolist()
            pprint(labels[1:10])
            outfile.write("DBSCAN clustering: \n")



            # See how it fits data on different dimensions
            print(pd.crosstab(labels, myDataNoNA['longitude']))
            print(pd.crosstab(labels, myDataNoNA['latitude']))
            print(pd.crosstab(labels, myDataNoNA['arrest']))
            print(pd.crosstab(labels, myDataNoNA['domestic']))
            print(pd.crosstab(labels, myDataNoNA['time_crime']))
            print(pd.crosstab(labels, myDataNoNA['day_of_week']))
            print(pd.crosstab(labels, myDataNoNA['indexCrime']))

            outfile.write(str(pd.crosstab(labels, myDataNoNA['longitude'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['longitude'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['arrest'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['domestic'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['time_crime'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['day_of_week'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['indexCrime'])))
            outfile.write("\n")

            #####
            # PCA
            # Let's convert our high dimensional data to 2 dimensions
            # using PCA
            pca2D = decomposition.PCA(2)

            # Turn the data into two columns with PCA
            plot_columns = pca2D.fit_transform(normalizedDataFrame)

            # Plot using a scatter plot and shade by cluster label
            plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
            plt.savefig('../data/images/dbscanpca2.png')
            plt.clf()
            plt.close()

            # Plot using a scatter plot and shade by cluster label geographically
            plt.scatter(x=myDataNoNA['longitude'], y=myDataNoNA['latitude'], c=labels, s= 1, lw=0)
            plt.savefig('../data/images/dbscanlatlong.png')
            plt.clf()
            plt.close()

            #WARD
            ward = AgglomerativeClustering(n_clusters=8, linkage='ward')
            ward.fit(normalizedDataFrame)

            labels = ward.labels_.tolist()
            pprint(labels[1:10])
            outfile.write("WARD clustering: \n")

            # See how it fits data on different dimensions
            print(pd.crosstab(labels, myDataNoNA['longitude']))
            print(pd.crosstab(labels, myDataNoNA['latitude']))
            print(pd.crosstab(labels, myDataNoNA['arrest']))
            print(pd.crosstab(labels, myDataNoNA['domestic']))
            print(pd.crosstab(labels, myDataNoNA['time_crime']))
            print(pd.crosstab(labels, myDataNoNA['day_of_week']))
            print(pd.crosstab(labels, myDataNoNA['indexCrime']))

            outfile.write(str(pd.crosstab(labels, myDataNoNA['longitude'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['longitude'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['arrest'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['domestic'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['time_crime'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['day_of_week'])))
            outfile.write("\n")
            outfile.write(str(pd.crosstab(labels, myDataNoNA['indexCrime'])))
            outfile.write("\n")

            #####
            # PCA
            # Plot using a scatter plot and shade by cluster label
            plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
            plt.savefig('../data/images/wardpca2.png')
            plt.clf()
            plt.close()

            # Plot using a scatter plot and shade by cluster label geographically
            plt.scatter(x=myDataNoNA['longitude'], y=myDataNoNA['latitude'], c=labels, s= 1, lw=0)
            plt.savefig('../data/images/wardlatlong.png')
            plt.clf()
            plt.close()

        outfile.closed
    file.closed

def assocruleschidata(data, out):

    #For use as function
    with open(data, 'r') as file:
    #For use in interactive exploration
    #with open("chidata2016cleaned.tsv", 'r') as file:
        myData = pd.read_table(file , sep='\t', encoding='latin1')
        myData = myData.drop('Unnamed: 0',axis=1)
        print(myData.info())
        #Quant vars
        print(myData.describe())

        #Results file
        with open(out, 'w') as outfile:
            # Remove missing data
            myDataNoNA = myData.dropna()
            myDataNoNA = myDataNoNA.reset_index(drop=True)
            myDataNoNA = myDataNoNA[myDataNoNA['recentCrime']==1]
            print(len(myDataNoNA.index))
            myDataPredict = myDataNoNA[['indexCrime','arrest','domestic','day_of_week','primary_type']]
            print(myDataPredict.info())
            myDataPredict['indexCrime'].replace(to_replace=1, value="index crime", inplace=True)
            myDataPredict['indexCrime'].replace(to_replace=0, value="not index crime", inplace=True)
            myDataPredict['arrest'].replace(to_replace=True, value="arrest", inplace=True)
            myDataPredict['arrest'].replace(to_replace=False, value="no arrest", inplace=True)
            myDataPredict['domestic'].replace(to_replace=True, value="domestic", inplace=True)
            myDataPredict['domestic'].replace(to_replace=False, value="not domestic", inplace=True)

            print(myDataPredict.info())
            print(myDataPredict[1:10])

            #association rules with 20, 30, 40 percent support and 80% confidence
            print(fim.arules(myDataPredict.values,supp=20,report="aSC",zmin=2))
            print(fim.arules(myDataPredict.values,supp=30,report="aSC",zmin=2))
            print(fim.arules(myDataPredict.values,supp=40,report="aSC",zmin=2))

            #Write out results to file
            outfile.write("Association rules with 20% support and 80% confidence: \n")
            outfile.write(str(fim.arules(myDataPredict.values,supp=20,report="aSC",zmin=2)))
            outfile.write("\n")
            outfile.write("Association rules with 30% support and 80% confidence: \n")
            outfile.write(str(fim.arules(myDataPredict.values,supp=30,report="aSC",zmin=2)))
            outfile.write("\n")
            outfile.write("Association rules with 40% support and 80% confidence: \n")
            outfile.write(str(fim.arules(myDataPredict.values,supp=40,report="aSC",zmin=2)))
            outfile.write("\n")

        outfile.closed
    file.closed

def predictchidata(data,out):

    #For use as function
    with open(data, 'r') as file:
    #For use in interactive exploration
    #with open("chidata2016cleaned.tsv", 'r') as file:
        myData = pd.read_table(file , sep='\t', encoding='latin1')
        myData = myData.drop('Unnamed: 0',axis=1)
        print(myData.info())
        #Quant vars
        print(myData.describe())
        print(myData[1:10])

        #Write out results to file
        with open(out, 'w') as outfile:


            # Remove missing data and focus on just first 10,000 recent crimes to lower runtime...
            myDataNoNA = myData.dropna()
            myDataNoNA = myDataNoNA.reset_index(drop=True)
            print(len(myDataNoNA.index))
            myDataNoNA = myDataNoNA[myDataNoNA['recentCrime']==1]
            myDataNoNA = myDataNoNA[1:10000]
            myDataPredict = myDataNoNA[['indexCrime','arrest','domestic','latitude','longitude','day_of_week','time_crime']]
            print(myDataPredict.info())


            #T-test to see if time of index and non-index crimes are different
            indexcrimesample = myDataPredict[myDataPredict['indexCrime']==1]
            a = indexcrimesample['time_crime']
            nonindexcrimesample = myDataPredict[myDataPredict['indexCrime']==0]
            b = nonindexcrimesample['time_crime']
            print(stats.ttest_ind( a,b,equal_var=False))

            outfile.write("T-test to test hypothesis that the time of index (more serious and non-index crimes are the same: /n")
            outfile.write(str(stats.ttest_ind( a,b,equal_var=False)))
            outfile.write("\n")


            #regression-- predict the time of a crime based on these other factors
            # Separate training and final validation data set. First remove target variable (time)
            #  from data (X). Setup target variable (Y)
            # Then make the validation set 20% of the entire
            # set of labeled data (X_validate, Y_validate)
            myDataPredict2 = myDataPredict[['time_crime','indexCrime','arrest','domestic','latitude','longitude','day_of_week']]
            valueArray = myDataPredict2.values
            X = valueArray[:,1:7]
            print(X[1:10])
            Y = valueArray[:,0]
            print(Y[1:10])
            test_size = 0.20
            seed = 7
            X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
            print(X_train[1:10])
            print(Y_train[1:10])
            Y_validate = Y_validate.tolist()
            Y_train = Y_train.tolist()
            # Create linear regression object
            regr = linear_model.LinearRegression()
            # Train the model using the training sets
            regr.fit(X_train, Y_train)
            # The coefficients
            print('Coefficients: \n', regr.coef_)
            # The mean squared error
            print("Mean squared error: %.4f"
                  % np.mean((regr.predict(X_validate) - Y_validate) ** 2))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % regr.score(X_validate, Y_validate))
            #This model performs rather poorly

            #write results out to file
            outfile.write("regression to see if we can predict the time of a crime based on its location, day, and whether it was domestic, an index crime, or resulted in an arrest: \n")
            outfile.write('Coefficients: \n'+ str(regr.coef_))
            outfile.write("Mean squared error: "+ str(np.mean((regr.predict(X_validate) - Y_validate) ** 2)))
            outfile.write('Variance score: ' + str(regr.score(X_validate, Y_validate)))
            outfile.write("\n")
            ######################################################
            # Evaluate algorithms
            ######################################################


            #Run A: numeric/quant attributes used
            print("Run A: using all numeric/quantitative attributes to try and classify index crimes")
            outfile.write("Machine learning techniquest to try and classify whether crimes fall under the index category (more serious): \n")
            #outfile.write("Run A: using all attributes \n")
            # Separate training and final validation data set. First remove class
            # label from data (X). Setup target class (Y)
            # Then make the validation set 20% of the entire
            # set of labeled data (X_validate, Y_validate)
            valueArray = myDataPredict.values
            X = valueArray[:,1:7]
            X = sklearn.preprocessing.normalize(X)
            print(X[1:10])
            Y = valueArray[:,0]
            print(Y[1:10])
            test_size = 0.20
            seed = 7
            X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
            print(X_train[1:10])
            print(Y_train[1:10])
            Y_validate = Y_validate.tolist()
            Y_train = Y_train.tolist()
            # Setup 10-fold cross validation to estimate the accuracy of different models
            # Split data into 10 parts
            # Test options and evaluation metric
            num_folds = 10
            num_instances = len(X_train)
            print(num_instances)
            seed = 7
            scoring = 'accuracy'

            ######################################################
            # Use different algorithms to build models
            ######################################################

            # Add each algorithm and its name to the model array
            models = []
            models.append(('NB', GaussianNB()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('RF', RandomForestClassifier()))
            models.append(('KNN', KNeighborsClassifier()))
            models.append(('SVM', SVC()))
            print(models)
            # Evaluate each model, add results to a results array,
            # Print the accuracy results (remember these are averages and std
            results = []
            names = []
            print(models[1:10])
            for name, model in models:
                kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
                cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                print(msg)
                outfile.write(msg+"\n")
            outfile.write("\n")
            print()

            ######################################################
            # See how well each model does on the
            # validation test
            ######################################################
            # Make predictions on validation dataset
            for name, model in models:
                mod = model
                mod.fit(X_train, Y_train)
                predictions = mod.predict(X_validate)
                #predictions = np.array(predictions)
                #Y_validate = np.array(Y_validate)
                print(predictions,Y_validate)
                print(accuracy_score(Y_validate, predictions))
                print(confusion_matrix(Y_validate, predictions))
                print(classification_report(Y_validate, predictions))
                #plot ROC curve and calucatlate AUC
                fpr,tpr,thresholds = roc_curve(Y_validate, predictions)
                plt.plot(fpr, tpr)
                plt.savefig(str(name)+'../data/images/roc.png')
                roc_auc = auc(fpr,tpr)
                print(roc_auc)
                #write out results
                outfile.write("The results of the " + name + "model on the validation set for classification of index crimes are: \n")
                outfile.write(str(accuracy_score(Y_validate, predictions)))
                outfile.write("\n")
                outfile.write(str(confusion_matrix(Y_validate, predictions)))
                outfile.write("\n")
                outfile.write(str(classification_report(Y_validate, predictions)) )
                outfile.write("\n")

                #outfile.write("The accuracy of the "+name+" model on the final validation data is: "+ "%f"%accuracy_score(Y_validate, predictions)+ "\n" )
           # outfile.write("\n")
        outfile.closed
    file.closed

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
