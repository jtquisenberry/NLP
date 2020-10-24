from project import app

class NERBert():
    def __init__(self, text):
        self.text = text

    def run(self):



        label_list = [
            "O",  # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",  # Beginning of a person's name right after another person's name
            "I-PER",  # Person's name
            "B-ORG",  # Beginning of an organisation right after another organisation
            "I-ORG",  # Organisation
            "B-LOC",  # Beginning of a location right after another location
            "I-LOC"  # Location
        ]

        def get_ner(sequence):

            tokenizer = app.config['tokenizer']
            model = app.config['model']

            # Bit of a hack to get the tokens with the special tokens
            tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
            inputs = tokenizer.encode(sequence, return_tensors="tf")

            outputs = model(inputs)[0]
            import tensorflow as tf
            predictions = tf.argmax(outputs, axis=2)

            print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])
            entities = format_for_brat(sequence, tokens, predictions)
            return entities

        def format_for_brat(sequence, tokens, predictions):
            entities = []
            import re
            my_regex = ''

            for token in tokens:
                if not token in ['[CLS]', '[SEP]']:
                    massaged_token = token.replace('\\', '\\\\').replace('.', r'\.').replace('##', '')
                    my_regex += '(' + massaged_token + ')' + r'\s*'

            expressions = re.search(my_regex, sequence)

            tokens2 = tokens[1:-1]
            predictions2 = predictions.numpy().tolist()[0][1:-1]

            for i in range(0, len(tokens2)):
                token = tokens2[i]
                prediction = predictions2[i]
                start_position = expressions.span(i + 1)[0]
                end_position = expressions.span(i + 1)[1]
                normalized_prediction = normalize_prediction(label_list[prediction])


                entity = ['T{0}'.format(i+1), normalized_prediction, [[start_position, end_position]]]
                entities.append(entity)


                #['T1', 'Location', [[69, 72]]],
                #['T2', 'Location', [[75, 77]]]

            b = 1

            return entities




        def normalize_prediction(prediction):

            prediction_map = {
                'O': 'Other',
                'B-MISC': 'Miscellaneous',
                'I-MISC': 'Miscellaneous',
                'B-PER': 'Person',
                'I-PER': 'Person',
                'B-ORG': 'Organization',
                'I-ORG': 'Organization',
                'B-LOC': 'Location',
                'I-LOC': 'Location'
            }

            return prediction_map[prediction]


        sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
                   "close to the Manhattan Bridge."

        sequence = self.text

        entities = get_ner(sequence=sequence)
        return entities