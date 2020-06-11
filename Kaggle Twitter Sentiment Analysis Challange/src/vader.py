#vader operates on full sentences that have not been processed for punctuation or anything else similar
import nltk
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def main():
    vader = SentimentIntensityAnalyzer()
    
    #train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    vaderText = test['text']
    vaderID = test['textID']
    
    #writing the initial file
    f = open("/kaggle/working/submission.csv", "w")
    f.write("textID,selected_text")
    
    #3534 in test set
    #The plan is to:
    #vader analysis each sentence, then strip off one word from front to back at a time
    #each iteration will compare to the previous one's compound score, and if the compound score increases we repeat
    #if the compound score decreases, we revert back to the previous string and use that one
    #if the compound score stays the same, we repeat the iteration as if it increased
    for i in range (3534):
        compoundBefore = abs(float(vader.polarity_scores(vaderText[i])['compound']))
        compoundAfter = compoundBefore
        frontBack = 1
        splitSentence = vaderText[i].split()
        joinSentence = " ".join(splitSentence)
        #check for whitespace/blanks or full neutrality to just skip the current sentence
        if (vaderText[i] == "" or vaderText[i].isspace() or compoundBefore == 0):
            #write data to file
            f.write("\n" + vaderID[i] + ",\"" + joinSentence + "\"")
            continue
        else:
            #Use frontBack to shave off either the front or the back of the word, 1 is front and 2 is back
            while (compoundAfter >= compoundBefore):
                #remove front
                if (frontBack == 1):
                    frontBack == 2
                    popped = splitSentence.pop(0)
                    joinSentence = " ".join(splitSentence)
                    compoundAfter = abs(float(vader.polarity_scores(joinSentence)['compound']))
                    if (compoundAfter < compoundBefore):
                        #previous score was better so we return the popped value to where it belongs and then exit the while loop
                        splitSentence.insert(0, popped)
                        break
                    else:
                        compoundBefore = compoundAfter
                #remove back
                else:
                    frontBack == 1
                    popped = splitSentence.pop()
                    joinSentence = " ".join(splitSentence)
                    compoundAfter = abs(float(vader.polarity_scores(joinSentence)['compound']))
                    if (compoundAfter < compoundBefore):
                        #previous score was better so we return the popped value to where it belongs and then exit the while loop
                        splitSentence.append(popped)
                        break
                    else:
                        compoundBefore = compoundAfter
            #while loop is complete, splitSentence holds our truncated value that we want
            joinSentence = " ".join(splitSentence)
            f.write("\n" + vaderID[i] + ",\"" + joinSentence + "\"")
