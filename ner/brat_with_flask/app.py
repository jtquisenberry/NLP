from flask import request, url_for, Flask
import json
# from flask_api import FlaskAPI, status, exceptions

app = Flask(__name__)


notes = {
    0: 'do the shopping',
    1: 'build the codez',
    2: 'paint the door',
}

def note_repr(key):
    return {
        'url': notes[key],
        'text': notes[key]
    }


@app.route("/", methods=['GET', 'POST'])
def notes_list():
    """
    List or create notes.
    """
    if request.method == 'POST':
        note = str(request.data.get('text', ''))
        idx = max(notes.keys()) + 1
        notes[idx] = note
        # return note_repr(idx), status.HTTP_201_CREATED
        return "junk"

    # request.method == 'GET'
    else:
        out_list = [note_repr(idx) for idx in sorted(notes.keys())]
        out_json = json.dumps(out_list)
        '''
        {"sentences": [{"basicDependencies": [{"dep": "nsubj", "governor": 4, "governorGloss": "state", "dependent": 1, "dependentGloss": "Washington"}, {"dep": "cop", "governor": 4, "governorGloss": "state", "dependent": 2, "dependentGloss": "is"}, {"dep": "det", "governor": 4, "governorGloss": "state", "dependent": 3, "dependentGloss": "a"}, {"dep": "root", "governor": 0, "governorGloss": "U.S.", "dependent": 4, "dependentGloss": "state"}, {"dep": "case", "governor": 7, "governorGloss": "U.S.", "dependent": 5, "dependentGloss": "in"}, {"dep": "det", "governor": 7, "governorGloss": "U.S.", "dependent": 6, "dependentGloss": "the"}, {"dep": "nmod", "governor": 4, "governorGloss": "state", "dependent": 7, "dependentGloss": "U.S."}], "tokens": [{"index": 1, "word": "Washington", "lemma": "Washington", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}, {"index": 2, "word": "is", "lemma": "be", "pos": "VBZ", "upos": "AUX", "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin", "ner": "O"}, {"index": 3, "word": "a", "lemma": "a", "pos": "DT", "upos": "DET", "feats": "Definite=Ind|PronType=Art", "ner": "O"}, {"index": 4, "word": "state", "lemma": "state", "pos": "NN", "upos": "NOUN", "feats": "Number=Sing", "ner": "O"}, {"index": 5, "word": "in", "lemma": "in", "pos": "IN", "upos": "ADP", "feats": null, "ner": "O"}, {"index": 6, "word": "the", "lemma": "the", "pos": "DT", "upos": "DET", "feats": "Definite=Def|PronType=Art", "ner": "O"}, {"index": 7, "word": "U.S.", "lemma": "U.S.", "pos": "NNP", "upos": "PROPN", "feats": "Number=Sing", "ner": "GPE"}]}]}
        '''
        return out_json

'''
@app.route("/<int:key>/", methods=['GET', 'PUT', 'DELETE'])
def notes_detail(key):
    """
    Retrieve, update or delete note instances.
    """
    if request.method == 'PUT':
        note = str(request.data.get('text', ''))
        notes[key] = note
        return note_repr(key)

    elif request.method == 'DELETE':
        notes.pop(key, None)
        return '', status.HTTP_204_NO_CONTENT

    # request.method == 'GET'
    if key not in notes:
        raise exceptions.NotFound()
    return note_repr(key)
'''

if __name__ == "__main__":
    app.run(debug=True)
