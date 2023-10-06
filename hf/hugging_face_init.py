from transformers import pipeline


def hello_world_hugging_face():
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained("bert-base-cased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print(output)


def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    res = classifier("today is a great day")
    print(res)


def text_generation():
    generator = pipeline("text-generation", model="distilgpt2")
    res = generator('i love the history of american civil war, we both know president lincoln ended', max_length=30,
                    num_return_sequences=2)
    print(res)


text_generation()
