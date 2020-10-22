from flask import request, url_for, Flask, render_template, request, redirect, current_app, g, send_from_directory
import json
import os
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

# Ignoring visible gpu device (device: 0, name: GeForce GTX 660 Ti, pci bus id: 0000:05:00.0,
# compute capability: 3.0) with Cuda compute capability 3.0. The minimum required Cuda capability is 3.5.
# Force use of CPU even if a GPU is available.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
print()




app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        raise NotImplementedError('POST is not implemented')
    # request.method == 'GET'
    else:
        groceries = list()
        return render_template('index.html', groceries=groceries)


@app.route('/ajax', methods=['GET', 'POST'])
def get_annotation():
    if request.method == 'POST':
        print(request.args)
        text_to_annotate = list(request.form.items())[0][0]
        print(text_to_annotate)

        # text = r"Produced wimzipFQm«tiy24)U)^ 33330FiteenlAe Road, Smith. Mict^ 480^; USA; ,TX; DRIVER LICENSE; 9 Class C; 4b Exp; 17777777; I4. iss 01/01/2011; 01/01/1975; Ud DL; - •,; ^'^3 DOB"
        text = text_to_annotate

        nb = NERBert(text_to_annotate)
        entities = nb.run()


        collData = {'entity_types':  [{
            'type': 'Location',
            'labels': ['Location', 'LOC'],
            'bgColor': '#9affe6',
            'borderColor': 'darken'
        }, {
            'type': 'Other',
            'labels': ['Other', 'O'],
            'bgColor': '#e3e3e3',
            'borderColor': 'darken'
        }, {
            'type': 'Person',
            'labels': ['Person', 'PER'],
            'bgColor': '#ffccaa',
            'borderColor': 'darken'
        }, {
            'type': 'Organization',
            'labels': ['Organization', 'ORG'],
            'bgColor': '#7fa2ff',
            'borderColor': 'darken'
        }, {
            'type': 'Miscellaneous',
            'labels': ['Miscellaneous', 'MISC'],
            'bgColor': '#e3e3e3',
            'borderColor': 'darken'
        }]
        }

        '''
        docData = {
            "text": text,
            "entities": [
                ['T1', 'Location', [[69, 72]]],
                ['T2', 'Location', [[75, 77]]]
            ],
        }
        '''
        docData = {
            "text": text,
            "entities": entities,
        }

        payload = {
            "docData": docData,
            "collData": collData
        }


        return_value = json.dumps(payload)
        # return_value = r'{"sentences": [{"basicDependencies": [{"dep": "nsubj", "governor": 4, "governorGloss": "state", "dependent": 1, "dependentGloss": "Washington"}, {"dep": "cop", "governor": 4, "governorGloss": "state", "dependent": 2, "dependentGloss": "is"}, {"dep": "det", "governor": 4, "governorGloss": "state", "dependent": 3, "dependentGloss": "a"}, {"dep": "root", "governor": 0, "governorGloss": "U.S.", "dependent": 4, "dependentGloss": "state"}, {"dep": "case", "governor": 7, "governorGloss": "U.S.", "dependent": 5, "dependentGloss": "in"}, {"dep": "det", "governor": 7, "governorGloss": "U.S.", "dependent": 6, "dependentGloss": "the"}, {"dep": "nmod", "governor": 4, "governorGloss": "state", "dependent": 7, "dependentGloss": "U.S."}], "tokens": [{"index": 1, "word": "Washington", "lemma": "Washington", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}, {"index": 2, "word": "is", "lemma": "be", "pos": "VBZ", "upos": "AUX", "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin", "ner": "O"}, {"index": 3, "word": "a", "lemma": "a", "pos": "DT", "upos": "DET", "feats": "Definite=Ind|PronType=Art", "ner": "O"}, {"index": 4, "word": "state", "lemma": "state", "pos": "NN", "upos": "NOUN", "feats": "Number=Sing", "ner": "O"}, {"index": 5, "word": "in", "lemma": "in", "pos": "IN", "upos": "ADP", "feats": null, "ner": "O"}, {"index": 6, "word": "the", "lemma": "the", "pos": "DT", "upos": "DET", "feats": "Definite=Def|PronType=Art", "ner": "O"}, {"index": 7, "word": "U.S.", "lemma": "U.S.", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}]}]}'
        return return_value

    else:
        # doc = app.config['doc']
        out_value = r'{"sentences": [{"basicDependencies": [{"dep": "nsubj", "governor": 4, "governorGloss": "state", "dependent": 1, "dependentGloss": "Washington"}, {"dep": "cop", "governor": 4, "governorGloss": "state", "dependent": 2, "dependentGloss": "is"}, {"dep": "det", "governor": 4, "governorGloss": "state", "dependent": 3, "dependentGloss": "a"}, {"dep": "root", "governor": 0, "governorGloss": "U.S.", "dependent": 4, "dependentGloss": "state"}, {"dep": "case", "governor": 7, "governorGloss": "U.S.", "dependent": 5, "dependentGloss": "in"}, {"dep": "det", "governor": 7, "governorGloss": "U.S.", "dependent": 6, "dependentGloss": "the"}, {"dep": "nmod", "governor": 4, "governorGloss": "state", "dependent": 7, "dependentGloss": "U.S."}], "tokens": [{"index": 1, "word": "Washington", "lemma": "Washington", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}, {"index": 2, "word": "is", "lemma": "be", "pos": "VBZ", "upos": "AUX", "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin", "ner": "O"}, {"index": 3, "word": "a", "lemma": "a", "pos": "DT", "upos": "DET", "feats": "Definite=Ind|PronType=Art", "ner": "O"}, {"index": 4, "word": "state", "lemma": "state", "pos": "NN", "upos": "NOUN", "feats": "Number=Sing", "ner": "O"}, {"index": 5, "word": "in", "lemma": "in", "pos": "IN", "upos": "ADP", "feats": null, "ner": "O"}, {"index": 6, "word": "the", "lemma": "the", "pos": "DT", "upos": "DET", "feats": "Definite=Def|PronType=Art", "ner": "O"}, {"index": 7, "word": "U.S.", "lemma": "U.S.", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}]}]}'
        raise NotImplementedError('GET is not implemented')

@app.route('/stanza-parseviewer.js', methods=['GET', 'POST'])
def load_viewer():
    return send_from_directory('static', 'stanza-parseviewer.js')

class NERBert():
    def __init__(self, text):
        self.text = text

    def run(self):

        model = TFAutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

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

            # Bit of a hack to get the tokens with the special tokens
            tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
            inputs = tokenizer.encode(sequence, return_tensors="tf")

            outputs = model(inputs)[0]
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



if __name__ == "__main__":
    app.run(debug=True)
