from flask import request, url_for, Flask, render_template, request, redirect, current_app, g, send_from_directory
import json
# from flask_api import FlaskAPI, status, exceptions

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
        return_value = json.dumps({'xxx': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'})
        return_value = r'{"sentences": [{"basicDependencies": [{"dep": "nsubj", "governor": 4, "governorGloss": "state", "dependent": 1, "dependentGloss": "Washington"}, {"dep": "cop", "governor": 4, "governorGloss": "state", "dependent": 2, "dependentGloss": "is"}, {"dep": "det", "governor": 4, "governorGloss": "state", "dependent": 3, "dependentGloss": "a"}, {"dep": "root", "governor": 0, "governorGloss": "U.S.", "dependent": 4, "dependentGloss": "state"}, {"dep": "case", "governor": 7, "governorGloss": "U.S.", "dependent": 5, "dependentGloss": "in"}, {"dep": "det", "governor": 7, "governorGloss": "U.S.", "dependent": 6, "dependentGloss": "the"}, {"dep": "nmod", "governor": 4, "governorGloss": "state", "dependent": 7, "dependentGloss": "U.S."}], "tokens": [{"index": 1, "word": "Washington", "lemma": "Washington", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}, {"index": 2, "word": "is", "lemma": "be", "pos": "VBZ", "upos": "AUX", "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin", "ner": "O"}, {"index": 3, "word": "a", "lemma": "a", "pos": "DT", "upos": "DET", "feats": "Definite=Ind|PronType=Art", "ner": "O"}, {"index": 4, "word": "state", "lemma": "state", "pos": "NN", "upos": "NOUN", "feats": "Number=Sing", "ner": "O"}, {"index": 5, "word": "in", "lemma": "in", "pos": "IN", "upos": "ADP", "feats": null, "ner": "O"}, {"index": 6, "word": "the", "lemma": "the", "pos": "DT", "upos": "DET", "feats": "Definite=Def|PronType=Art", "ner": "O"}, {"index": 7, "word": "U.S.", "lemma": "U.S.", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}]}]}'
        return return_value

    else:
        # doc = app.config['doc']
        out_value = r'{"sentences": [{"basicDependencies": [{"dep": "nsubj", "governor": 4, "governorGloss": "state", "dependent": 1, "dependentGloss": "Washington"}, {"dep": "cop", "governor": 4, "governorGloss": "state", "dependent": 2, "dependentGloss": "is"}, {"dep": "det", "governor": 4, "governorGloss": "state", "dependent": 3, "dependentGloss": "a"}, {"dep": "root", "governor": 0, "governorGloss": "U.S.", "dependent": 4, "dependentGloss": "state"}, {"dep": "case", "governor": 7, "governorGloss": "U.S.", "dependent": 5, "dependentGloss": "in"}, {"dep": "det", "governor": 7, "governorGloss": "U.S.", "dependent": 6, "dependentGloss": "the"}, {"dep": "nmod", "governor": 4, "governorGloss": "state", "dependent": 7, "dependentGloss": "U.S."}], "tokens": [{"index": 1, "word": "Washington", "lemma": "Washington", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}, {"index": 2, "word": "is", "lemma": "be", "pos": "VBZ", "upos": "AUX", "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin", "ner": "O"}, {"index": 3, "word": "a", "lemma": "a", "pos": "DT", "upos": "DET", "feats": "Definite=Ind|PronType=Art", "ner": "O"}, {"index": 4, "word": "state", "lemma": "state", "pos": "NN", "upos": "NOUN", "feats": "Number=Sing", "ner": "O"}, {"index": 5, "word": "in", "lemma": "in", "pos": "IN", "upos": "ADP", "feats": null, "ner": "O"}, {"index": 6, "word": "the", "lemma": "the", "pos": "DT", "upos": "DET", "feats": "Definite=Def|PronType=Art", "ner": "O"}, {"index": 7, "word": "U.S.", "lemma": "U.S.", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}]}]}'
        raise NotImplementedError('GET is not implemented')

@app.route('/stanza-parseviewer.js', methods=['GET', 'POST'])
def load_viewer():
    return send_from_directory('static', 'stanza-parseviewer.js')




if __name__ == "__main__":
    app.run(debug=True)
