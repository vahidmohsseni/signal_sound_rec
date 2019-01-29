from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import plot_wav

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/check')
def check():
    t = plot_wav.calc_mse("up")
    p = plot_wav.Sound('up')
    p.save_plot()
    
    if t[0] == 0:
        return render_template('result.html', result="kheyr", bale_p=str(t[1]), kheyr_p=str(t[2]))
       
    return render_template('result.html', result="bale", bale_p=str(t[1]), kheyr_p=str(t[2]))



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['data']
        f.save(secure_filename("up"+'.wav'))


        return "done"
    return render_template('upload.html')



if __name__ == "__main__":
    app.run(debug=True)