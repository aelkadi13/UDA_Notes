#Idea: Analyze the amount of apologies in last statements of death row inmates
#Importing the necessary libraries
import pandas as pd
import nltk
from nltk.stem import PorterStemmer

#Import data table
data = "/Users/adamel-kadi/Desktop/UDA Data/last_statements.csv"

#Read the data
data = pd.read_csv(data)

#Check data
print(data.head())

#I want to count how many statements contain a "sorry" in them
apology = data["statements"].str.contains("sorry", case=False, na=False)
print("The word 'sorry' is found in ", apology.sum(), "statements")

#Now give me what percentage of all statements contain this apology
total_statements = data["statements"].count()
percentage = (apology.sum() / total_statements) * 100
print("This accounts for", round(percentage, 2), "percent of all statements")

#Let's take this to the next level. Let's use nltk to account for statements that include "apologize" or "apology"
#Stemming the words
apologetic_strings = ["apology", "apologize", "sorry", "regret", "apologizing", "apologized"]
apologetic = data["statements"].str.contains("|".join(apologetic_strings), case=False, na=False)
print("Apologetic words are found in ", apologetic.sum(), "statements")

#Now give me what percentage of all statements contain this apology
percentage = (apologetic.sum() / total_statements) * 100
print("This accounts for", round(percentage, 2), "percent of all statements")

#Now, I want to disclude statements that aren't genuine, such as "I do not apologize..." or "I am not sorry..."
#I will do this by checking if the word "not" directly preceds the apologetic word
#I will use nltk to tokenize the words and then check if "not" is in the list
#Tokenize the words

#I could not get this idea to work, here is the code that I ran to try
nltk.download('punkt')
ps = PorterStemmer()
data["statements"] = data["statements"].apply(lambda x: nltk.word_tokenize(x))
data["statements"] = data["statements"].apply(lambda x: [ps.stem(word) for word in x])

#Check if "not" is in the list  
data["statements"] = data["statements"].apply(lambda x: "not" in x)
data["statements"] = data["statements"].apply(lambda x: not x)
print(data["statements"])

























