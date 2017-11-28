from __future__ import unicode_literals

import os
import subprocess
import sys
import time
from functools import wraps
import tempfile
import stat

from flask import Flask, request, jsonify, current_app
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src'))
from jingjuElemSeg.jingjuElementSegmentation import jingjuElementSegmentation
from jingjuSegAlign.jingjuSegmentAlignment import jingjuSegmentAlignment


DATA_DIR = os.getenv('APP_DATA_DIR')

if not DATA_DIR:
    raise ValueError("Cannot find configuration for APP_DATA_DIR")

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

        tmp_workdir = tempfile.mkdtemp(dir='/data')
        student_file_fn = os.path.join(tmp_workdir, 'student.mp3')
        student_file_fp = open(student_file_fn, 'w')

        blob.save(student_file_fp)
        student_file_fp.close()

        phraseNumberBlob = request.files['phraseNumber']
        phraseNumber = phraseNumberBlob.read()

        print 'phraseNumber', phraseNumber

        start_time = time.time()  # starting time

        os.chmod(tmp_workdir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        student_pitchtrack_fn = os.path.join(tmp_workdir, 'student_pitchtrack.csv')
        student_monoout_fn = os.path.join(tmp_workdir, 'student_monoNoteOut.csv')

        # run bash code for getting pitch track and notes
        if sys.platform == 'linux' or sys.platform == 'linux2':
            subprocess.call(["./sonic-annotator-linux64", "-t", "pitchtrack.n3", student_file_fn, "-w", "csv", "--csv-one-file", student_pitchtrack_fn])
            subprocess.call(["./sonic-annotator-linux64", "-t", "monoNoteOut.n3", student_file_fn, "-w", "csv", "--csv-one-file", student_monoout_fn])
        else:
            subprocess.call(["./sonic-annotator", "-t", "pitchtrack.n3", student_file_fn, "-w", "csv", "--csv-one-file", student_pitchtrack_fn])
            subprocess.call(["./sonic-annotator", "-t", "monoNoteOut.n3", student_file_fn, "-w", "csv", "--csv-one-file", student_monoout_fn])

        # process
        if os.path.isfile(student_pitchtrack_fn) and os.path.isfile(student_monoout_fn):
            try:
                jingjuElementSegmentation(tmp_workdir)
                print 'semgnetation is done!'
                jingjuSegmentAlignment(phraseNumber, tmp_workdir)
                print 'alignment is done!'
            except ValueError:
                #  ElementSegmentation can raise an error if the files are not present
                pass
            status = 'ok'
        else:
            status = 'error'

        runningTime = time.time() - start_time
        output = {'running Time': runningTime, 'data_root': tmp_workdir, 'status': status}
    return jsonify(**output)


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run(host= '0.0.0.0', port=5001, debug=True)
