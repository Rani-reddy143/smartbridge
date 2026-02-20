from flask import Flask, render_template, request
import joblib
from pathlib import Path
import sys

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model safely
BASE_DIR = Path(__file__).resolve().parent
# Preferred model filename
preferred = BASE_DIR / 'random_forest_model.pkl'
# Look for common model file extensions if preferred file missing
candidates = []
for ext in ('*.pkl', '*.joblib', '*.sav', '*.pickle'):
    candidates.extend(list(BASE_DIR.glob(ext)))

model_file = None
if preferred.exists():
    model_file = preferred
elif candidates:
    # choose the first candidate found
    model_file = candidates[0]
class DummyModel:
    """Very small fallback model with a predict method.

    It returns 0 for every input so the app can start even when the
    serialized model is missing. Replace with your real model file
    for production use.
    """
    def predict(self, X):
        # return zeros matching sklearn's API (list/array-like)
        return [0 for _ in range(len(X))]


if model_file is None:
    # No model file found — fall back to DummyModel and log to stderr
    sys.stderr.write(
        f"WARNING: Model file not found. Expected '{preferred.name}'. Starting with DummyModel.\n")
    model = DummyModel()
else:
    try:
        model = joblib.load(model_file)
    except Exception as e:
        # If joblib load fails, fall back to DummyModel but surface the error on startup
        sys.stderr.write(f"WARNING: Failed to load model from {model_file}: {e}\n")
        model = DummyModel()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        age_first_funding_year = float(request.form['age_first_funding_year'])
        age_last_funding_year = float(request.form['age_last_funding_year'])
        age_first_milestone_year = float(request.form['age_first_milestone_year'])
        age_last_milestone_year = float(request.form['age_last_milestone_year'])
        relationships = float(request.form['relationships'])
        funding_rounds = float(request.form['funding_rounds'])
        funding_total_usd = float(request.form['funding_total_usd'])
        milestones = float(request.form['milestones'])
        avg_participants = float(request.form['avg_participants'])

        # Create input list
        input_data = [[
            age_first_funding_year,
            age_last_funding_year,
            age_first_milestone_year,
            age_last_milestone_year,
            relationships,
            funding_rounds,
            funding_total_usd,
            milestones,
            avg_participants
        ]]

        # Prediction
        prediction = model.predict(input_data)[0]

        # Result mapping
        result = "Acquired" if prediction == 1 else "Closed"

        return render_template('results.html', result=result)

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
