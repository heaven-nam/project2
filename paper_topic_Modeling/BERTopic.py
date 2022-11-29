from bertopic import BERTopic
import csv


# Data reading
with open("springer_ta.csv") as file:
    docs = csv.reader(file)
    abstract = list()
    for doc in docs:
        abstract.append(doc[-1])

    # training
    topic_model = BERTopic(language="english", calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(abstract)

    # Extracting Topics
    freq = topic_model.get_topic_info()
    print(freq.head(5))

print(topic_model.get_topic(0))