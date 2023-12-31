# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: finalprojectcf
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Final Project

# %%
import re
from dateutil import parser

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import missingno as msn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# %% [markdown]
# ## Utilities

# %%
def calculateMeanOrModePerColumnWithMissingValues(dataset, targetColumn:str) -> list:
    '''
    ***DESCRIPTION***
    In order to fill NA or empty values of every single column with
    missing values this function helps to get mean or mode of a 
    given column. That would help to fill up each cell of a given 
    column with its regarding mean o mode based on Favorite Genres
    groups.
    
    ***PARAMETERS***
    dataset = Data from .csv file
    targetColumn = Given column we wish to calculate the mean
    
    ***NOTES***
    'Favorite Genre' column was choosen to group the data due to is
    one of the 100% filled columns and is easy to clasify people by
    musical genre.
    '''
    
    # Get the list of fav genres per null values
    favGenres = dataset.loc[dataset[targetColumn].isnull(), 'Fav genre'].tolist()
    # Dictionary with the values for NA per column
    restultDict = {}
    # Process for columns with numerical values
    try:
        for favGenre in favGenres:
            mean = int((dataset[dataset["Fav genre"] == favGenre][targetColumn].mean()).round())
            restultDict[favGenre] = mean
        return restultDict
    # Process for columns with string values
    except:
        for favGenre in favGenres:
            mode = dataset[dataset["Fav genre"] == favGenre][targetColumn].mode().iloc[0]
            restultDict[favGenre] = mode
        return restultDict

def fillUpEmptyFieldsByColumn(dataset, columnNames, calculateMeanOrModePerColumnWithMissingValuesMethod):
    '''
    ***DESCRIPTION***
    This method fills up the empty cells of the dataset by column 
    with its regarding mean or mode depending of the Favorite 
    Music Genre the respondant selected.
    
    ***PARAMETERS***
    dataset = Data from .csv file
    columnNames = List with the names of the columns in the 
    dataset
    calculateMeanOrModePerColumnWithMissingValuesMethod = Function
    to calculate the mean or mode per column and favorite genre
    '''
    
    # Iterate over the list of column names
    for columnName in columnNames:
        # Identify if that column has any empty value
        if dataset[columnName].isna().any():
            # If true, then calculates the mean or mode for that column by fav music genre
            valuesReplacement = calculateMeanOrModePerColumnWithMissingValuesMethod(dataset=dataset, targetColumn=columnName)
            # Iterates over each row of the given dataset
            for index, row in dataset.iterrows():
                # Check if that row has and empty value in the current iteration of column name
                if pd.isna(row[columnName]) and row["Fav genre"] in valuesReplacement:
                    # If true, then fill up that empty cell with its regarding mean or mode by favorite music genre
                    dataset.at[index, columnName] = valuesReplacement[row["Fav genre"]]
    # Returns a new dataset without empty cells
    return dataset

def createBoxplotChart(dataset, axis_y:str):
    '''
    ***DESCRIPTION***
    This function creates a single boxplot chart. 
    
    
    ***PARAMETERS*** 
    dataset = Data from .csv file
    axis_y = Name of the desired column to measure
    
    ***NOTES***
    The 'X' axis is considered as the 'Fav genre' column of a 
    given dataset
    '''
    
    # Create a boxplot chart without style parameters
    boxplotChart = alt.Chart(dataset).mark_boxplot().encode(
    alt.X("Fav genre:N"),
    alt.Y(f"{axis_y}:Q")
    )
    # Returns the chart
    return boxplotChart

def graphPeopleByPathologyAndMusicGenre(dataset, pathologyThreshold:int, pathologyName:str, musicGenreFrecuency="Very frequently"):
    '''
    ***DESCRIPTION***
    This function creates a single graphic bar that shows
    the total of people listening each music genre with certain
    frequency VS the level of highness they reported for each
    pathology
    
    Pathologies on this case are: Anxiety, Depression, Insomnia
    and OCD (Obsessive Compulsive Disorder).
    
    ***PARAMETERS***
    dataset = Data from .csv file
    pathologyThreshold = From 1 to 10 considered the level of 
    drasticity of the pathology
    pathologyName = Desired pathology to analyze
    musicGenreFrecuency = By default 'Very frequently. Nominal
    frequency considered to count people results.
    '''
    # Get music genres columns names
    columnNames = [column for column in dataset.columns if re.match(r"Frequency\b", column)]
    # Specify pathology threshold
    threshold = pathologyThreshold
    # Filter the given dataset by pathology and threshold
    data = dataset[dataset[pathologyName] > threshold]
    # Store music genres and total of people with certain 
    # pathology listening each genre with an specific frequency
    totals = {"genre": [], "total": []}

    # Iterate over each column
    for column in columnNames:
        # Count people by genre
        count = data[data[column] == musicGenreFrecuency][column].count()
        # Store music genre in the dictionary
        totals["genre"].append(column)
        # Store total count in the dictionary
        totals["total"].append(count)

    # Create a temporal dataframe with the needed counts
    df = pd.DataFrame(totals)
    
    # Create a chart for the temporal df results
    chart = alt.Chart(df).mark_bar().encode(
    alt.X("total:Q"),
    alt.Y("genre:N", sort="-x"),
    ).properties(
        title=f"{pathologyName}"
    )
    
    # Create labels for showing the totals by column in the chart
    texts = chart.mark_text(
        align="left",
        baseline="middle"
    ).encode(
        text="total:Q"
    )
    # Combine the bar chart and the text labels
    chart = chart + texts
    # Return a single chart
    return chart

def calculateMeanPerListenersOverAndUnderNinetyBPM(dataset, pathologyName:str) -> dict:
    '''
    ***DESCRIPTION***
    This function calculates the mean over the total
    people listening to music over and under 90 BPM
    levels by a given pathology name and returns a
    bar chart showing the results to compare both
    means.
    
    ***NOTES***
    This function requires a column named 'BPM over 90'
    containing True = 1 for people listening to music
    over 90 BPM and False = 0 for people listening to
    music under 90 BPM.
    '''
    
    # Calculate mean for people over 90 BPM by the given pathology name
    meanOverLimit = dataset[dataset["BPM over 90"] == 1][pathologyName].mean()
    # Calculate mean for people under 90 BPM by the given pathology name
    meanUnderLimit = dataset[dataset["BPM over 90"] == 0][pathologyName].mean()
    # Create a temporal mino dataframe with the former calculated means
    tempDataset = pd.DataFrame({
        "BPM level": ["BMP over 90", "BPM under 90"],
        "Means": [meanOverLimit, meanUnderLimit]
        })
    
    # Create a single bar chart showing labels and means
    chart = alt.Chart(tempDataset).mark_bar().encode(
        alt.X("BPM level:N", title="BPM music level people listen to"),
        alt.Y("Means:Q", title="Mean Level"),
        color=alt.condition(
            alt.datum["Means"] < 5,
            alt.value("green"),
            alt.value("red")
        )
    ).properties(
        title=f"{pathologyName} mean comparison for 90 BPM music listeners and no listeners"
    )
    
    # Return bar chart
    return chart


# %% [markdown]
# ## Dataset Load

# %%
# Load dataset
dataset = pd.read_csv("dataset.csv")

# %% [markdown]
# ## EDA Level

# %%
# View of the dataset's current structure
dataset.head(5)

# %%
# View dataset's current columns object types
dataset.info()

# %%
''' 
Quick view of missing values per column.

White lines = Missing values
Gray lines = True values

'BPM' column has the mayority the missing values far beyond. 
Probably due to the most of the people are not familiar with the 
concept of what 'BPM' is or has no notion of the beats per minute 
of the music they listen to. Here we realize it is needed to work
all of those missing values.
'''

# Create matrix with True and Missing values
msn.matrix(dataset)

# %%
''' 
Quick view of the correlation between column with missing values.

Red color = Weak correlation
White color = Normal correlation
Blue color = Strong correlation

'Composer' column is the one with the strongest correlation with 
other columns with missing values. Which means that people is not 
composer, in the most of the cases, is not musician either and 
they skip these columns.
'''

# Create a heatmap with the correlation between columns with missing values
msn.heatmap(dataset)

# %%
''' 
Count of true or filled values per column.
'''

# Create a bar chart with the total of true values per column
msn.bar(dataset)

# %% [markdown]
# At this point of the EDA stage has been detected two mayor insights.
#
# 1) BPM column has the most of the missed values and needs to be the priority.
# 2) Is not a good a idea to simply delete the rows with any empty value, due to the fact we have more filled
# information than empty in every single row with NaN and would be a loose of valuable data.
#
# Before working with empty values is necessary to look for any outlier in the current dataset structure.

# %%
'''
First, it was identified all the columns in dataset with values
with numeric type since those are the ones with probability of
having drastic outliers. It was used the technique of graphic
visualization first. The one choosen was a boxplot chart due to 
the accuracy offered by the quartiles calculation. The pourpose 
was look up if there were columns with events out of their own mean 
and it was found a column that actually does.

The unique chart that actually breaks its own measure scale and has 
a value far from the rest of the dataset is the based on the axis Y
'BPM'. That insight makes sense due to is the one with the most of
the missing values in the dataset.

Now that the outliers were found is necessary to fix them.
'''

outlierForAge = createBoxplotChart(dataset=dataset, axis_y="Age")

outlierForHoursPerDay = createBoxplotChart(dataset=dataset, axis_y="Hours per day")

outlierForBPM = createBoxplotChart(dataset=dataset, axis_y="BPM")

outlierForAnxiety = createBoxplotChart(dataset=dataset, axis_y="Anxiety")

outlierForDepression = createBoxplotChart(dataset=dataset, axis_y="Depression")

outlierForInsomnia = createBoxplotChart(dataset=dataset, axis_y="Insomnia")

outlierForOCD = createBoxplotChart(dataset=dataset, axis_y="OCD")

outlierGeneralChart = outlierForAge & outlierForHoursPerDay | outlierForBPM & outlierForAnxiety | outlierForDepression & outlierForInsomnia | outlierForOCD

# General boxplot chart visualization
outlierGeneralChart

# %%
'''
In order to confirm the former finding an individual test was
applied. For this pourpose it was used the previous function to 
calculate mean and mode by column, in this case, for the 'BPM' 
column grouped by Favorite music genre.

The mean bpm for the 'Video game music' genre is overstated, that
confirms something goes wrong with that nominal category in that
specific column.
'''

# Calculate mean for the 'BPM' column by favorite genre
bpmMean = calculateMeanOrModePerColumnWithMissingValues(dataset=dataset, targetColumn="BPM")

# Show result
bpmMean

# %%
'''
Here is explicit shown the outlier. A value of 999999999 is the
root of the problem. Probably a surveyed mistake or bad intention.
'''

# Get the max value in 'BPM' column
dataset["BPM"].max()

# Show the row with that max value in the 'BPM' column
dataset.loc[dataset["BPM"] == dataset["BPM"].max()]["BPM"]

# %%
'''
The outlier is replaced by the mean of bpm for the Favorite genre
'Video game music' obviously without the value of the outlier.
'''

# Replace the outlier for the mean value of 'BPM' for that music genre
dataset.at[568, "BPM"] = dataset[(dataset["BPM"] < 999999999.0) & (dataset["Fav genre"] == "Video game music")]["BPM"].mean()

# Replace the outlier for the mean value of 'BPM' for EDM genre
dataset.at[644, "BPM"] = dataset[dataset["Fav genre"] == "EDM"]["BPM"].mean().round()

# %%
'''
Calculating once again the mean for the 'BPM' column we get normal
values.
'''

# Calculate mean for the 'BPM' column by favorite genre
bpmMean = calculateMeanOrModePerColumnWithMissingValues(dataset=dataset, targetColumn="BPM")

# Show result
bpmMean

# %%
'''
Now the 'fillUpEmptyFieldsByColumn' function can replace the
missing values efficiently.
'''

# Get a list with the name of every column in the dataset
columnNames = dataset.columns.tolist()

# Fill empty cells in every column
newDataSet = fillUpEmptyFieldsByColumn(dataset=dataset, 
                                        columnNames=columnNames, 
                                        calculateMeanOrModePerColumnWithMissingValuesMethod=calculateMeanOrModePerColumnWithMissingValues)

# %%
'''
A new matrix is generated to have a quick view of empty data. 
Now that the problem of outliers and missing values are solved 
is necessary to reformat the object types for the dataset's 
columns and drop the unuseful ones if any.
'''

# Generate a new matrix of empty data
msn.matrix(newDataSet)

# %% [markdown]
# At this point of the EDA process we fixed the problem of missing data and outliers, now
# we need to reformat the dataset with the correct object type by column and clean it from
# unuseful columns as well before visualizations.

# %%
'''
A dictionary is designed with the types needed for each column
of the dataset to assure future processes success.
'''

# Desired object types per column
columnTypes = {
    "Age": int,
    "Primary streaming service": str,
    "Hours per day": "Float64",
    "While working": str,
    "Instrumentalist": str,
    "Composer": str,
    "Fav genre": str,
    "Exploratory": str,
    "Foreign languages": str,
    "BPM": int,
    "Frequency [Classical]": str,
    "Frequency [Country]": str,
    "Frequency [EDM]": str,
    "Frequency [Folk]": str,
    "Frequency [Gospel]": str,
    "Frequency [Hip hop]": str,
    "Frequency [Jazz]": str,
    "Frequency [K pop]": str,
    "Frequency [Latin]": str,
    "Frequency [Lofi]": str,
    "Frequency [Metal]": str,
    "Frequency [Pop]": str,
    "Frequency [R&B]": str,
    "Frequency [Rap]": str,
    "Frequency [Rock]": str,
    "Frequency [Video game music]": str,
    "Anxiety": int,
    "Depression": int,
    "Insomnia": int,
    "OCD": int,
    "Music effects": str,
    "Permissions": str
}

# %%
'''
Before transformation, each column is checked just to make sure
that the cast process does not have issues.
'''

# Iterate over every dataset's column
for column in dataset.columns:
    # Try to convert every column to numeric and print True
    if pd.to_numeric(dataset[column], errors="coerce").notna().all():
        print(f"{column}", True)
    # If can't convert to numeric because there is any other data with different type on the column print False
    else:
        print(f"{column}", False)

# Parse the 'Timestamp' column into a real datetime format
dataset["Timestamp"] = dataset["Timestamp"].apply(lambda x: parser.parse(x, dayfirst=True))

# %% [markdown]
# The previous code result confirm is posible turn column types without issues
# on the try.

# %%
'''
Now the data is clean and organized the transformation to the new 
column types is made.
'''

# Change object type per column based on the column types dictionary
dataset = dataset.astype(columnTypes)

# Confirm the new dtypes
dataset.info()

# %%
'''
Columns considered as no related to medical pourpuses are dropped
from the dataset.
'''

# List of columns desired to be removed
unnecessaryColumns = ["Primary streaming service", "While working", "Exploratory", "Foreign languages", "Permissions"]

# Drop columns in the list
dataset.drop(unnecessaryColumns, axis=1, inplace=True)


# %% [markdown]
# ## What music genres does people with high levels of anxiety, depression, insomnia and OCD listen to more frequently?

# %%
'''
Rock, Pop and Metal are the common genres between people with 
pathology levels higher than 5 points. This insight could mean
two statements.

1) Rock, Pop and Metal are just the most popular music genres
and that fact makes them the easy to be choose between people
with any type of disorder.
2) Rock, Pop and Metal has a negative impact in people mental
health.
'''

anxietyCount = graphPeopleByPathologyAndMusicGenre(dataset=dataset,
                                                    pathologyThreshold=5,
                                                    pathologyName="Anxiety")

depressionCount = graphPeopleByPathologyAndMusicGenre(dataset=dataset,
                                                    pathologyThreshold=5,
                                                    pathologyName="Depression")

insomniaCount = graphPeopleByPathologyAndMusicGenre(dataset=dataset,
                                                    pathologyThreshold=5,
                                                    pathologyName="Insomnia")

ocdCount = graphPeopleByPathologyAndMusicGenre(dataset=dataset,
                                                    pathologyThreshold=5,
                                                    pathologyName="OCD")

# Concatenate charts 
pathologiesGeneralChart = anxietyCount & depressionCount | insomniaCount & ocdCount

# Show general chart
pathologiesGeneralChart




# %% [markdown]
# ## Which BPM rank has more listeners?

# %%
'''
Most of the people listen to music over 90 BPM, probably related
to Rock, Pop and Metal music which could make sense along the
result of the former question.
'''


# Limits to cut each label rank
bins = [0, 50, 70, 90, 110, 130, 150, 170, 190, 210]

# Labels to name each rank created
labels = ["0 - 50", "51 - 70", "71 - 90", "91 - 110", "111 - 130", "131 - 150", "151 - 170", "171 - 190", 'Más de 200']

# Cut the 'BPM' column into ranks
dataset["BPM ranks"] = pd.cut(dataset["BPM"], bins=bins, labels=labels)

# Temporal dataset with the total count of people by BPM rank
ranksCounts = dataset["BPM ranks"].value_counts().reset_index()

# Rename columns of the temporal dataset
ranksCounts.columns = ["BPM ranks", "Count"]

# Chart of BPM ranks and count of listeners
chart = alt.Chart(ranksCounts).mark_bar().encode(
    alt.X("BPM ranks:N", title="BPM ranks", sort="-y"),
    alt.Y("Count:Q", title="Listeners total count")
)

# Total count labels
labels = chart.mark_text().encode(
    text="Count:Q",
)

# Concatenate bar chart and labels
chart = chart + labels

# Show chart
chart

# %% [markdown]
# ## Considering 90 BPM as high level music regarding Dr. Emma Gray's investigation, Can we confirm that people listening to music over 90 BPM presents higher levels of anxity, depression, insomnia and OCD than people listening to music under that level?

# %%
'''
A new column into the dataset is created to specify wheter BPM
score is higher than 90 or not. This transformation was made
using the techniche of 'Label Encoding' and the object 
LabelEncoder from the scikit-learn library was useful for this.
'''

# Instance of the preprocessor object LabelEncoder
labelEncoder = LabelEncoder()

# Training of the object and transformation by the BPM column
dataset["BPM over 90"] = labelEncoder.fit_transform(dataset["BPM"] > 90).astype(int)


# %%
'''
From 1 to 10 on the allowed scale to rate pathologies levels, a
mean score over 5 is considered as a critical level. Regarding
the below chart, the anxiety mean is the only one above 
that limit for both cathegories (people listening to music over
90 BPM and under).

Red bars = Means over the limit of 5
Green bars = Means under the limit of 5

These results aims to the fact anxiety, depression, insomnia
and OCD are not related to the BPM of the music people listen 
to. Both variables has similar means in every pathology case
and are below the limit with exception of anxiety.
'''

# Bar chart to compare means of people listening to music above 90 BPM and under 90 BPM by anxiety pathology
meansAnxiety = calculateMeanPerListenersOverAndUnderNinetyBPM(dataset=dataset, pathologyName="Anxiety")

# Bar chart to compare means of people listening to music above 90 BPM and under 90 BPM by depression pathology
meansDepression = calculateMeanPerListenersOverAndUnderNinetyBPM(dataset=dataset, pathologyName="Depression")

# Bar chart to compare means of people listening to music above 90 BPM and under 90 BPM by insomnia pathology
meansInsomnia = calculateMeanPerListenersOverAndUnderNinetyBPM(dataset=dataset, pathologyName="Insomnia")

# Bar chart to compare means of people listening to music above 90 BPM and under 90 BPM by ocd pathology
meansOCD = calculateMeanPerListenersOverAndUnderNinetyBPM(dataset=dataset, pathologyName="OCD")

# Concatenate bar charts
meansAnxiety & meansDepression | meansInsomnia & meansOCD


# %% [markdown]
# As product of this EDA process we got:
#
# 1) No empty values
# 2) No outliers
# 3) Correct object types per variable on the dataset
# 4) Answers for main project's questions
#

# %% [markdown]
# # Modeling level

# %% [markdown]
# ## Create a model to predict wheter a person can confirm that music improves its mental healths or not.

# %% [markdown]
# Due the problem is about predicting if a person improves its mental health or not by listening to music we deal with a classification problem. The selected model to propose the solution is Logistical Regresion. The main task of this model is classify new entrys between two possible categories "Music effects" = Improve or Not improve.

# %%
'''
As machine Learning models works with numerical values, before
training a logistical regresion model is necessary to transform
the data into numerical values. Here the One-Hot Encoding 
technique was used to turn the three different answers of the
column "Music effects" into True (1) or False (0) values.
'''

# Instance of the One-Hot Encoder object
oneHotEncoder = OneHotEncoder(dtype=int)

# Train the One-Hot Encoder object with the values of the column 'Music effects'
musicEffectsOneHot = oneHotEncoder.fit_transform(dataset[["Music effects"]])

# Turn the One-Hot Encoded sparse matrix into a pandas dataframe
dfMusicEffectsOneHot = pd.DataFrame(musicEffectsOneHot.toarray(), columns=oneHotEncoder.get_feature_names_out(["Music effects"]))

# Concatenate the general dataset with the One-Hot Encoded dataset columns
dataset = pd.concat([dataset, dfMusicEffectsOneHot], axis=1)

# Create a new column that will contain value = 1 whether 'Music effects_No effect' or 'Music effects_Worsen' has value 1
dataset['Music effects_No improve'] = dataset['Music effects_No effect'] | dataset['Music effects_Worsen']

# Remove all the unuseful columns for the Logistic Regresion model pouposes
dataset = dataset.drop(["Timestamp", 
                "Fav genre",
                "Instrumentalist", 
                "Composer", 
                'Frequency [Classical]', 
                'Frequency [Country]',
                'Frequency [EDM]',
                'Frequency [Folk]',
                'Frequency [Gospel]',
                'Frequency [Hip hop]',
                'Frequency [Jazz]',
                'Frequency [K pop]',
                'Frequency [Latin]',
                'Frequency [Lofi]',
                'Frequency [Metal]',
                'Frequency [Pop]',
                'Frequency [R&B]',
                'Frequency [Rap]',
                'Frequency [Rock]',
                'Frequency [Video game music]',
                "Music effects",
                "Music effects_No effect",
                "Music effects_Worsen",
                "BPM ranks"], axis=1)

# %%
'''
In order to have better visualizations and make the logistical
regresion model calcs easier the data is passed by a 
normalization process using the technique of Min-Max to fit
the data in a 0 to 1 scale.
'''

# Instance of the MinMax normalizing method
normalizeMethod = MinMaxScaler()

# Remplace values on the needed by its normalized form
dataset[["Age", "Hours per day", "BPM", "Anxiety", "Depression", "Insomnia", "OCD"]] = normalizeMethod.fit_transform(dataset[["Age", "Hours per day", "BPM", "Anxiety", "Depression", "Insomnia", "OCD"]])

# Show the new normalized dataset
dataset


# %%
'''
The current dataset is ready to be passed to the Logistical
Regresion Model and train it to make predictions.
'''


# (Model's independant variables). Get a temporal dataset with all the columns except the ones in the drop() method
X = dataset.drop(["Music effects_Improve", "Music effects_No improve"], axis=1)

# (Model's dependant variables). Get a single panda's series with the column to predict
y = dataset["Music effects_Improve"]

# Separate dataset between training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=10)

# Instance of the logistic regresion model object
model = LogisticRegression()

# Train the logistic regresion model object
model.fit(X_train, y_train)

# Test the model by predicting results for the testing data
y_predict = model.predict(X_test)

# Test model accuracy comparing the real values of testing data VS the values predicted
accuracy = accuracy_score(y_test, y_predict)

# Get a confusion matrix to see results
confusion = confusion_matrix(y_test, y_predict).tolist()

# Show results
accuracy, confusion

# %%
'''
Report results.
'''

# Creates a general report with differents accuracy methods
report = classification_report(y_test, y_predict, zero_division=0, output_dict=True)

# Transform the report into a dataframe
x = pd.DataFrame(report).transpose()

# Show dataframe report
x

# %% [markdown]
# Regarding to the results the model has an accuracy of 85% which is quite acceptable. Now the model is ready to make predictions over new data entries. A new prediction is made to prove the model's functionality

# %%
newData = {"Age":0.101266, "Hours per day":0.166667, "BPM":0.600000, "Anxiety":.7, "Depression":.7, "Insomnia":.10, "OCD":.2, "BPM over 90": .1}

newPredict = model.predict([list(newData.values())]).tolist()[0]

newPredict

# %%
dataset.info()
