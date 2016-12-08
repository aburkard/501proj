# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:50:16 2016

@author: caryt
"""


#Cary Lou
#ANLY501
#Project data checking
#9/21/2016
import requests
import pandas as pd
import sys
import numpy as np



def main(argv):
     #open file
    with open('../data/ChicagoCrime2016.txt', 'w') as OutFile:


        #Write column header
        OutFile.write("arrest\tbeat\tblock\tcase_number\tcommunity_area\tdate\tdescription\tdistrict\tdomestic\tfbi_code\tincident_id\tiucr\tlatitude\tlocation_description\tlongitude\tlocation_type\tlocation_1\tlocation_2\tprimary_type\tupdated\tward\tx_coord\ty_coord\tyear\n")

        #Define endpoint
        baseurl = "https://data.cityofchicago.org/resource/6zsd-86xi.json"

        #Pull in all 2015 crime report records for the city of chicago by passing year and app token in URL
        #UPDATE to read in from file instead of hard coded:
        with open('resources/ChiCrimeURLpost.csv', 'r') as urlpostfile:
            csvDataFrame=pd.read_csv(urlpostfile)
            urlpost = csvDataFrame.ix[0].to_dict()
        urlpostfile.closed
        #Get data from Chicago Crime API site in JSON format
        response = requests.get(baseurl,urlpost)
        if response.status_code == 200:
            data = response.json()

        #Confirm data was pulled by printing beginning
        #print(data[1:100])

        #For each record, assign different data return elements to a variable to prep for writing the record into our file
        #Try writing all variables, but if an attribte is not returned write an empty string instead.
        for y in data:
            arrest = str(y["arrest"])
            beat = str(y["beat"])
            block = str(y["block"])
            case_number = str(y["case_number"])
            community_area = str(y["community_area"])
            date = str(y["date"])
            description = str(y["description"])
            district = str(y["district"])
            domestic = str(y["domestic"])
            fbi_code = str(y["fbi_code"])
            incident_id = str(y["id"])
            iucr = str(y["iucr"])
            try:
                latitude = str(y["latitude"])
            except:
                latitude = ""
            try:
                location_description = str(y["location_description"])
            except:
                location_description = ""
            try:
                longitude = str(y["longitude"])
            except:
                longitude = ""
            try:
                location_type = str(y["location"]["type"])
            except:
                location_type = ""
            try:
                location_1 = str(y["location"]["coordinates"][0])
            except:
                location_1 = ""
            try:
                location_2 = str(y["location"]["coordinates"][1])
            except:
                location_2 = ""
            primary_type = str(y["primary_type"])
            updated = str(y["updated_on"])
            try:
                ward = str(y["ward"])
            except:
                ward = ""
            try:
                x_coord = str(y["x_coordinate"])
            except:
                x_coord = ""
            try:
                y_coord = str(y["y_coordinate"])
            except:
                y_coord = ""
            year = str(y["year"])

            #Write/Output to tab delimited file instead because some data has commas ',' in it:
            OutFile.write(arrest + "\t" + beat + "\t" + block + "\t" + case_number + "\t" + community_area + "\t" + date + "\t" +  description + "\t" + district + "\t" +
            domestic + "\t" + fbi_code + "\t" + incident_id + "\t" + iucr + "\t" +  latitude + "\t" +  location_description + "\t" +  longitude + "\t" + location_type + "\t" + location_1 + "\t" + location_2 + "\t" +  primary_type + "\t" +
            updated + "\t" +  ward + "\t" +  x_coord + "\t" +   y_coord + "\t" +  year + "\n")


        #Notify user that script has finished running
        print("done")

    OutFile.closed

    ##CLEAN DATA AND GENERATE NEW ATTRIBUTES AND OUTPUT TO NEW FILE
    with open('../data/ChicagoCrime2016.txt', 'r') as file:
        #read data into data frame
        Data = pd.read_table('../data/ChicagoCrime2016.txt' , sep='\t', encoding='latin1')
        #print first 10 records
        print(Data[0:10])



        Data.dtypes

        ##TRY TO REMOVE HARD CODING OF VALID RANGES AND VALUES OF DIFF. VARS.

        #Part 2: Analyzing the Data
        #write out to text fiel
        with   open('../data/ChicagoDataMissingnoisevalues.txt', 'w' ) as outfile:
            #1. number/fraction of missing values for each variable:
            totalMissing = 0
            for y in Data:
                countMissing = 0
                for i in range(len(Data[y])):
                    if str(Data.ix[i,y]) == "nan" :
                        countMissing = countMissing + 1
                print("\n\nThe number of missing values for", y, "is:")
                print(countMissing)
                print("Out of", len(Data[y]), "records.")
                outfile.write("The number of missing values for "+ str(y)+ " is: "+ str(countMissing) + " out of " + str(len(Data[y])) + " records, or " + "{:.2%}".format(countMissing/len(Data[y])) +".\n")
                totalMissing = totalMissing + countMissing
            #2. number/fraction of noise values (implausible values) for each variable:
            outfile.write("\n")
            totalNoise = 0
            checkvals = pd.read_csv("resources/ChicagoDataCheckVals.csv")
            checkrange = pd.read_csv("resources/ChicagoDataCheckRanges.csv")
            for y in Data:
                try:
                    checkframe = Data[y].isin(checkvals[y])
                    countWrong = 0
                    for i in range(len(checkframe)):
                        if checkframe[i] == False:
                            countWrong = countWrong + 1
                            #print(i, Data.ix[i,y])
                    print(countWrong, y)
                    outfile.write("The number of noise values for "+ str(y)+ " is: "+ str(countWrong) + " out of " + str(len(Data[y])) + " records, or " + "{:.2%}".format(countWrong/len(Data[y])) +".\n")
                    totalNoise = totalNoise + countWrong
                except:
                    countWrong = 0
                    for i in range(len(Data[y])):
                        if Data.ix[i,y] > checkrange.ix[1,y] or Data.ix[i,y] < checkrange.ix[0,y]:
                            countWrong = countWrong + 1
                            #print(i, Data.ix[i,y])
                    print(countWrong, y)
                    outfile.write("The number of noise values for "+ str(y)+ " is: "+ str(countWrong) + " out of " + str(len(Data[y])) + " records, or " + "{:.2%}".format(countWrong/len(Data[y])) +".\n")
                    totalNoise = totalNoise + countWrong

            #data quality score:
            totalCells = len(Data.index)*len(Data.columns)
            shareMissing = totalMissing/totalCells
            shareNoise =  totalNoise/totalCells
            shareBadData = (totalMissing+totalNoise)/totalCells
            print("The share of missing cells is "+"{:.2%}".format(shareMissing))
            print("The share of cells with noise values is "+"{:.2%}".format(shareNoise))
            print("Therefore, the total share of cells with bad data (either missing or noise) is "+"{:.2%}".format(shareBadData)+", which is our data quality score for this dataset.")
            outfile.write("\n")
            outfile.write("The share of missing cells is " + "{:.2%}".format(shareMissing) +"\n")
            outfile.write("The share of cells with noise values is " + "{:.2%}".format(shareNoise) +"\n")
            outfile.write("Therefore, the total share of cells with bad data (either missing or noise) is " + "{:.2%}".format(shareBadData)+", which is our data quality score for this dataset.\n")

        outfile.closed

        #Part 3: Feature generation: construct 3 new variables using downloaded data
        #Create an attribute for reported domestic crimes where an arrest was made:
        Data['domesticArrest']= 0
        for i in range(len(Data['domesticArrest'])):
            if Data.ix[i,'domestic'] == 1 and Data.ix[i,'arrest'] == 1 :
                Data.ix[i,'domesticArrest'] = 1
        #Create an attribute for reported domestic crimes where an arrest was made:
        crimeTypes = pd.read_csv("resources/crimeTypes.csv")
        Data['indexCrime'] = 0
        indexCrimeCheck = Data['fbi_code'].isin(crimeTypes['indexCrime'])
        for i in range(len(indexCrimeCheck)):
            if indexCrimeCheck[i] == True :
                Data.ix[i,'indexCrime'] = 1
        #Create an attribute for reported property crimes:
        Data['propertyCrime'] = 0
        propertyCrimeCheck = Data['fbi_code'].isin(crimeTypes['propertyCrime'])
        for i in range(len(propertyCrimeCheck)):
            if propertyCrimeCheck[i] == True :
                Data.ix[i,'propertyCrime'] = 1

        print(Data[0:10])
        #Create an attribute for reported
        #Write data out:
        myFileName="../data/ChicagoCrime2016W3newfeatures.txt"
        Data.to_csv(myFileName,sep='\t')


    file.closed



if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
