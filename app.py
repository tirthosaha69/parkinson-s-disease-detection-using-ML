from flask import Flask, render_template, request
from model import predict_parkinson

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    if request.method == "POST":
        try:
            raw_input = request.form["features"]
            # Split by comma and convert to float
            features = [float(x.strip()) for x in raw_input.split(",")]
            if len(features) != 22:
                prediction_text = "Error: Please enter exactly 22 values separated by commas."
            else:
                prediction_text = predict_parkinson(features)
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
