from __future__ import unicode_literals

import os
import subprocess
import sys
import time
from functools import wraps

from flask import Flask, request, jsonify, current_app
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src/jingjuElemSeg'))
from jingjuElementSegmentation import jingjuElementSegmentation
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src/jingjuSegAlign'))
from jingjuSegmentAlignment import jingjuSegmentAlignment

app = Flask(__name__)
CORS(app)

def support_jsonp(f):
    """Wraps JSONified output for JSONP"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        callback = request.args.get('callback', False)
        if callback:
            content = str(callback) + '(' + str(f(*args, **kwargs)) + ')'
            return current_app.response_class(content, mimetype='application/javascript')
        else:
            return f(*args, **kwargs)

    return decorated_function

@app.route('/do_seg', methods=['GET', 'POST'])
@support_jsonp
def do_seg():
    """
    Example: http://127.0.0.1:5000/do_seg?audio=[1,2,3]
    you have to see how to send json object, we can parse that. We might need that for passing audio arrays
    """
    if request.method == 'POST':
        blob = request.files['data']
        blob.save('../data/updateFiles/student.mp3')

        phraseNumberBlob = request.files['phraseNumber']
        phraseNumber = phraseNumberBlob.read()

        print 'phraseNumber', phraseNumber

        start_time = time.time()  # starting time

        if os.path.isfile('../data/updateFiles/student_pitchtrack.csv'):
            subprocess.call(["rm","../data/updateFiles/student_pitchtrack.csv"])
        if os.path.isfile('../data/updateFiles/student_monoNoteOut.csv'):
            subprocess.call(["rm","../data/updateFiles/student_monoNoteOut.csv"])

        # run bash code for getting pitch track and notes
        if sys.platform == 'linux' or sys.platform == 'linux2':
            subprocess.call(["./sonic-annotator-linux64", "-t", "pitchtrack.n3", "../data/updateFiles/student.mp3", "-w", "csv", "--csv-one-file", "../data/updateFiles/student_pitchtrack.csv"])
            subprocess.call(["./sonic-annotator-linux64", "-t", "monoNoteOut.n3", "../data/updateFiles/student.mp3", "-w", "csv", "--csv-one-file", "../data/updateFiles/student_monoNoteOut.csv"])
        else:
            subprocess.call(["./sonic-annotator", "-t", "pitchtrack.n3", "../data/updateFiles/student.mp3", "-w", "csv", "--csv-one-file", "../data/updateFiles/student_pitchtrack.csv"])
            subprocess.call(["./sonic-annotator", "-t", "monoNoteOut.n3", "../data/updateFiles/student.mp3", "-w", "csv", "--csv-one-file", "../data/updateFiles/student_monoNoteOut.csv"])

        # process
        if os.path.isfile('../data/updateFiles/student_monoNoteOut.csv') and os.path.isfile('../data/updateFiles/student_pitchtrack.csv'):
            jingjuElementSegmentation()
            print 'semgnetation is done!'
            jingjuSegmentAlignment(phraseNumber)
            print 'alignment is done!'

        runningTime = time.time() - start_time
        output = {'running Time': runningTime}
    return jsonify(**output)

if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run(host= '0.0.0.0', port=5001, debug=True)
