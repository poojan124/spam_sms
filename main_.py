from prepro import preprocess
from NB import NBC

if __name__ == "__main__":
    file_name = 'data'

    #object for preprocess data read,clean,transform
    process = preprocess(file_name)
    data = process.read_and_clean()

    #Naive Bayes classifier object
    classifier = NBC(data,split_size=0.2)
    classifier.train()
    classifier.test_run()
    classifier.run()
