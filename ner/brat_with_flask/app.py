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
