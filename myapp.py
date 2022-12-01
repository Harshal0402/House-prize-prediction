from flask import *
from mlmodel import *
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/getprediction", methods=["POST"])
def getpredict():
    aai = request.form["aai"]
    aaha = request.form["aaha"]
    aanr = request.form["aanr"]
    aanb = request.form["aanb"]
    ap = request.form["ap"]
    lo = request.form["lo"]

    newob = [[aai, aaha, aanr, aanb, ap,lo]]
    print(newob)
    model = makeprediction()
    yp = model.predict(newob)[0]
    print(yp)
    return render_template("landing.html", data=yp)
    


    
    




if(__name__ == "__main__"):
    app.run(debug=True)
