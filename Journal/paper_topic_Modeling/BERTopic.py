from bertopic import BERTopic
import csv


class BertTopic:

    def __init__(self):
        pass


    def data_reading():
        with open("springer_ta.csv") as file:
            docs = csv.reader(file)
            abstract = list()
            for doc in docs:
                abstract.append(doc[-1])

        return abstract


    def training(abstract):
        topic_model = BERTopic(language="english", calculate_probabilities=True)
        topics, probs = topic_model.fit_transform(abstract)

        return topic_model, topics, probs


    def extracting_topics(topic_model):
        freq = topic_model.get_topic_info()
        print(freq.head(5))

        return topic_model.get_topic(0)

