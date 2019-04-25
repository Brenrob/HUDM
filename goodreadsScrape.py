from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut as leaveOne
import numpy as np

myUrl = "https://www.goodreads.com/list/show/6"

uClient = uReq(myUrl) # opening connection and grabbing page
pageHTML = uClient.read() # storing HTML
uClient.close() # closing connection

# Create the HTML "soup"
page_soup = soup(pageHTML, "html.parser") 

# array to hold all of our titles
titleList = []

# text.strip is used to get rid of extra whitespace and newlines
for title in page_soup.findAll("a", {"class":'bookTitle'}):
    titleList.append(title.text.strip())

# array to hold all of our authors
# array that contains a 1 if the author is listed as a Goodreads Author, and a 0 otherwise
authorList = []
goodreadsList = []

# authorRegex is used to match the "(Goodreads Author)" appended to the author's name
authorRegex = r"(Goodreads Author)"

# text.strip is used to get rid of extra whitespace and newlines
for author in page_soup.findAll("span", {"itemprop":"author"}):
    # will either be a match or a nothing value. "re.search" searches the whole string
    authorMatch = re.search(authorRegex, author.text.strip())
    
    # if we have a match...
    if(authorMatch):
        # append 1 to identify goodreads author
        goodreadsList.append(1)
    else:
        # otherwise we append zero for no match
        goodreadsList.append(0)
    # append author to author list
    authorList.append(author.text.strip())
    
# array to hold our ratings
# This is the actual rating it got out of 5 stars, represented as a decimal number
ratingList = []

for rating in page_soup.findAll("span", {"class":"minirating"}):
    decimalRating = rating.text[1:5]
    ratingList.append(decimalRating)

# array to hold the actual number of ratings, can be in the millions down to the tens of thousands
ratingCountList = []
# regular expression for matching the number of ratings
ratingCountRegex = r"([0-9]{0,1}[,]{0,1}[0-9]{2,3}[,]{1}[0-9]{3})" 
for ratingCount in page_soup.findAll("span", {"class" : "minirating"}):
    ratingCountMatch = re.search(ratingCountRegex, ratingCount.text)
    if (ratingCountMatch):
        # print (ratingCountMatch.group(0))
        ratingCountList.append(ratingCountMatch.group(0))
        
# array to hold the number of votes
voteList = []

# array to hold the books score
scoreList = []

# contents[5] gives us the sixth child of the span tag we searched for (which is the a class containing number of votes)
# and text gives us the actual string

# regular expression for a number of format -,--- or ---
voteRegex = r"(-{0,1}[1-9]{0,1}[,]{0,1}[0-9]{3})"

# regular expression to match the "score: " string at the beginning of every score
scoreRegex = r"(score: )"
for string in page_soup.findAll("span", {"class":"smallText uitext"}):
    # find if there's a match within the text we scraped
    voteMatch = re.search(voteRegex, string.contents[5].text)
    
    # if there's a match, get the text corresponding to the first "group" in the regular expression
    # append that to voteList
    if (voteMatch):
        voteList.append(voteMatch.group(0))
    
    # using the find and replace function in the regular expression library, we can replace
    # the beginning of the score string, leaving only the number
    score = re.sub(scoreRegex, '', string.contents[1].text)
    scoreList.append(score)

# converting all the string values we have into appropriate types for data analysis
for i in range (0, len(ratingList)):
    ratingList[i] = float(ratingList[i]) # can be converted directly
    ratingCountList[i] = float(ratingCountList[i].replace(',','')) # commas must be removed from string
    voteList[i] = float(voteList[i].replace(',','')) # commas must be removed from string
    scoreList[i] = float(scoreList[i].replace(',','')) # commas must be removed from string

# dictionary of info to hold info for data frame
bookInfo = {
    'Title': titleList,
    'Author': authorList,
    'Rating': ratingList,
    'NumRatings': ratingCountList,
    'NumVotes': voteList,
    'Goodreads': goodreadsList,
    'Score': scoreList
}

# pandas data frame
bookData = pd.DataFrame(bookInfo)
print(bookData.head(20)) # printing first 5 rows of bookData

print(bookData.corr())

# make a scatter plot with x-axis NumVotes, y-axis Score
plt.scatter(bookData['NumVotes'], bookData['Score'], label = "Votes")
plt.xlabel('Votes') 
plt.ylabel('Score')
plt.title('Score vs. Votes')
plt.show()

# make a scatter plot with x-axis NumRatings, y-axis Score
plt.scatter(bookData['NumRatings'], bookData['Score'], label = "Ratings")
plt.xlabel('Number of Ratings')
plt.ylabel('Score')
plt.title('Score vs. Number of Ratings')
plt.show()

# make a scatter plot with x-axis Rating, y-axis Score
plt.scatter(bookData['Rating'], bookData['Score'], label = "Ratings")
plt.xlabel('Rating')
plt.ylabel('Score')
plt.title('Score vs.Rating')
plt.show()

# make a scatter plot with x-axis NumVotes, y-axis NumRatings
plt.scatter(bookData['NumVotes'], bookData['NumRatings'], label = "Ratings")
plt.xlabel('Votes')
plt.ylabel('Ratings')
plt.title('Number of Votes vs. Number of Ratings')
plt.show()

# turn our NumVotes column into an array for processing with sklearn, reshape to 2D array for fit()
votesData = np.array(bookData.NumVotes)
votesData = votesData.reshape(-1,1)

# run model and display coefficients
reg = linear_model.LinearRegression()
reg.fit(votesData, bookData.Score)
print(reg.coef_)

# Run prediction with numbers from books 56 and 57
predictionTest = np.array([1390, 1475])
predictionTest = predictionTest.reshape(-1,1)
reg.predict(predictionTest)

# turn our NumVotes and Goodreads columns into an array for processing with sklearn
votesDataGR = np.array(bookData[['NumVotes','Goodreads']])

# run model and display coefficients
regGR = linear_model.LinearRegression()
regGR.fit(votesDataGR, bookData.Score)
print(regGR.coef_)

# Run prediction with numbers from books 56 and 57
predictionTest = np.array([[1390, 1], [1475,0]])
regGR.predict(predictionTest)

# create shorthand variable, validate the number of splits
loo = leaveOne()
print(loo.get_n_splits(votesData))

# create numpy array for the score
scoreData = np.array(bookData.Score)

# for loop using votesData, which only takes into account votes
sum = 0
for trainIndex, testIndex in loo.split(votesData):
    # this will divide our modeling values into a train and test set
    votesDataTrain, votesDataTest = votesData[trainIndex], votesData[testIndex]
    # this will divide our dependent values into a train and test set
    scoreDataTrain, scoreDataTest = scoreData[trainIndex], scoreData[testIndex]
    
    # create a linear regression model and fit it with the train data
    regLoo = linear_model.LinearRegression()
    regLoo.fit(votesDataTrain, scoreDataTrain)

    # predict the score based on test sample
    predicted = regLoo.predict(votesDataTest)
    
    # calculate difference between actual and predicted point
    # square it to remove negatives 
    difference = (predicted - scoreDataTest) ** 2
    # add each of these to sum
    sum += difference
    
# take the root mean of the sum    
mean = (sum/100) ** 0.5
print("RMS Value of votes model: ", mean)

# for loop using votesDataGR, which takes into account if they are a "Goodreads Author"
sumGR = 0
for trainIndex, testIndex in loo.split(votesDataGR):
    # this will divide our modeling values into a train and test set
    votesDataTrain, votesDataTest = votesDataGR[trainIndex], votesDataGR[testIndex]
    # this will divide our dependent values into a train and test set
    scoreDataTrain, scoreDataTest = scoreData[trainIndex], scoreData[testIndex]
    
    # create a linear regression model and fit it with the train data
    regLoo = linear_model.LinearRegression()
    regLoo.fit(votesDataTrain, scoreDataTrain)

    # predict the score based on test sample
    predicted = regLoo.predict(votesDataTest)
    
    # calculate difference between actual and predicted point
    # square it to remove negatives 
    difference = (predicted - scoreDataTest) ** 2
    # add each of these to sum
    sumGR += difference
    
# take the root mean of the sum    
meanGR = (sumGR/100) ** 0.5
print("RMS value of votesGR model: ", meanGR)