from ner_flaskr import app


class NERBert():
    def __init__(self, text):
        self.text = text
        self.label_list = label_list = [
            "O",  # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",   # Beginning of a person's name right after another person's name
            "I-PER",   # Person's name
            "B-ORG",   # Beginning of an organisation right after another organisation
            "I-ORG",   # Organisation
            "B-LOC",   # Beginning of a location right after another location
            "I-LOC"    # Location
        ]

    def run(self, sequence=""):

        tokenizer = app.config['tokenizer']
        model = app.config['model']

        # Bit of a hack to get the tokens with the special tokens
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
        inputs = tokenizer.encode(sequence, return_tensors="tf")

        outputs = model(inputs)[0]
        import tensorflow as tf
        predictions = tf.argmax(outputs, axis=2)

        print([(token, self.label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])
        entities = self._format_for_brat(sequence, tokens, predictions)
        return entities

    def _format_for_brat(self, sequence, tokens, predictions):
        entities = []
        import re
        my_regex = ''

        for token in tokens:
            if token not in ['[CLS]', '[SEP]']:
                # JQ
                # Regex substitutions
                massaged_token = token.replace('\\', '\\\\').replace('.', r'\.').replace('##', '')\
                    .replace(r'?', r'\?').replace(r'(', r'\(').replace(r')', r'\)').replace('*', r'\*')\
                    .replace('[', r'\[').replace(']', r'\]').replace('+', r'\+').replace('{', r'\{')\
                    .replace(r'}', r'\}').replace(r',', r'\,').replace(r'-', r'\-')

                my_regex += '(' + massaged_token + ')' + r'\s*'

        expressions = re.search(my_regex, sequence)
        tokens2 = tokens[1:-1]
        predictions2 = predictions.numpy().tolist()[0][1:-1]

        breakpointA = 1

        for i in range(0, len(tokens2)):
            token = tokens2[i]
            prediction = predictions2[i]
            start_position = expressions.span(i + 1)[0]
            end_position = expressions.span(i + 1)[1]
            normalized_prediction = self._normalize_prediction(self.label_list[prediction])
            # ['T1', 'Location', [[69, 72]]],
            # ['T2', 'Location', [[75, 77]]]
            entity = ['T{0}'.format(i+1), normalized_prediction, [[start_position, end_position]]]
            entities.append(entity)

        entities = self.combine_pound_entities(entities=entities, tokens=tokens2, sequence=sequence)
        entities = self.combine_location_entities(entities, sequence)

        return entities

    def _normalize_prediction(self, prediction):

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

    def combine_pound_entities(self, entities, tokens, sequence):

        type_map = {'Other': 0, 'Miscellaneous': 1, 'Organization': 2, 'Person': 3, 'Location': 4}

        previous_token = ''
        previous_type = 'Other'
        previous_end = -1
        combined_entities = []
        for entity_number, entity in enumerate(entities):

            token = tokens[entity_number]

            # Adjacent tokens
            current_start = entity[2][0][0]
            current_type = entity[1]
            if current_start == previous_end:
                # Either the previous or the current has ##
                if (len(previous_token) >= 3 and previous_token[-2:] == '##') or (
                        len(token) >= 3 and token[:2] == '##'):
                    # Merge
                    combined_entities[-1][2][0][1] = entity[2][0][1]
                    # Get best type
                    if type_map[previous_type] >= type_map[current_type]:
                        combined_entities[-1][1] = previous_type
                    else:
                        combined_entities[-1][1] = current_type

                else:
                    combined_entities.append(entity)
            else:
                combined_entities.append(entity)

            previous_end = entity[2][0][1]
            previous_type = entity[1]

        for x in combined_entities:
            print(x, sequence[x[2][0][0]:x[2][0][1]])

        return combined_entities

    def combine_location_entities(self, entities, sequence):

        combined_entities = []
        previous_type = 'Other'
        previous_token = ''
        keep_adding = False

        for entity_number, entity in enumerate(entities):
            # Append or merge
            # append
            if entity_number == 11:
                breakpointA = 0
            token = sequence[entity[2][0][0]:entity[2][0][1]]
            token_type = entity[1]
            if not keep_adding:
                combined_entities.append(entity)
                if token_type == 'Location':
                    keep_adding = True
                else:
                    keep_adding = False
            else:
                if token_type == 'Location':
                    if previous_type == 'Location':
                        combined_entities[-1][2][0][1] = entity[2][0][1]
                        keep_adding = True
                    elif previous_type == 'Other' and len(token) == 1:
                        combined_entities[-1][2][0][1] = entity[2][0][1]
                        keep_adding = True
                    else:
                        combined_entities.append(entity)
                elif token_type == 'Other' and token == '.' and len(
                        previous_token) == 1 and previous_type == 'Location':
                    combined_entities[-1][2][0][1] = entity[2][0][1]
                    keep_adding = True
                else:
                    combined_entities.append(entity)
                    keep_adding = False

            previous_token = token
            previous_type = token_type

            aaa = 1
        bbb = 1

        '''
        for e in combined_entities:
            print(e)
        '''

        return combined_entities
