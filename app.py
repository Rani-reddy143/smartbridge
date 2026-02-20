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
else:
    # Look for files with similar base name (for example a notebook named
    # "random_forest_model.pkl.ipynb") and provide an actionable hint.
    similar = list(BASE_DIR.glob('random_forest_model*'))
    # remove any exact matches we already considered
    similar = [p for p in similar if p != preferred and p not in candidates]
    if similar:
        sys.stderr.write("WARNING: Found files with similar names to the expected model:\n")
        for p in similar:
            sys.stderr.write(f"  - {p.name}\n")
        sys.stderr.write(
            "Hint: If one of these is a notebook containing the model, export or save\n"
            "the serialized model as 'random_forest_model.pkl' (or any '*.pkl'/'*.joblib')\n"
            "and place it in the project root so the app can load it.\n"
        )


class DummyModel:
    """Very small fallback model with a predict method.

    It returns 0 for every input so the app can start even when the
    serialized model is missing. Replace with your real model file
    for production use.
    """
    def predict(self, X):
        return [0 for _ in range(len(X))]


if model_file is None:
    sys.stderr.write(f"WARNING: Model file not found. Expected '{preferred.name}'. Starting with DummyModel.\n")
    model = DummyModel()
else:
    try:
        model = joblib.load(model_file)
    except Exception as e:
        sys.stderr.write(f"WARNING: Failed to load model from {model_file}: {e}\n")
        model = DummyModel()


# Known 25-feature ordering used during training (fallback)
KNOWN_FEATURES_25 = [
    'age_first_funding_year','age_last_funding_year',
    'age_first_milestone_year','age_last_milestone_year',
    'relationships','funding_rounds','funding_total_usd',
    'milestones','avg_participants',
    'is_CA','is_NY','is_MA','is_TX','is_otherstate',
    'is_software','is_web','is_mobile','is_enterprise',
    'is_advertising','is_gamesvideo','is_ecommerce',
    'is_biotech','is_consulting','is_othercategory',
    'is_top500'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # aliases map canonical feature name -> possible form field names
        aliases = {
            'age_first_funding_year': ['age_first_funding_year', 'affy'],
            'age_last_funding_year': ['age_last_funding_year', 'alfy'],
            'age_first_milestone_year': ['age_first_milestone_year', 'afmy'],
            'age_last_milestone_year': ['age_last_milestone_year', 'almy'],
            'relationships': ['relationships'],
            'funding_rounds': ['funding_rounds', 'funding'],
            'funding_total_usd': ['funding_total_usd', 'totalfunding'],
            'milestones': ['milestones'],
            'avg_participants': ['avg_participants', 'participants'],
        }

        def get_form_val(names, default=None):
            for n in names:
                v = request.form.get(n)
                if v is not None and v != '':
                    return v
            return default

        # Decide which feature ordering to use
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = KNOWN_FEATURES_25

        # Build input vector matching feature_names
        input_vector = []
        for fname in feature_names:
            if fname in aliases:
                raw = get_form_val(aliases[fname], default=None)
            else:
                raw = request.form.get(fname)

            if raw is None:
                val = 0.0
            else:
                try:
                    val = float(raw)
                except Exception:
                    sval = str(raw).strip().lower()
                    if sval in ('true', '1', 'yes', 'on'):
                        val = 1.0
                    else:
                        val = 0.0

            input_vector.append(val)

        # Ensure length matches model expectation if available
        n_in = getattr(model, 'n_features_in_', None)
        if n_in is not None:
            if len(input_vector) < n_in:
                input_vector += [0.0] * (n_in - len(input_vector))
            elif len(input_vector) > n_in:
                input_vector = input_vector[:n_in]

        prediction = model.predict([input_vector])[0]

        # Result mapping (use High/Low for the UI)
        result = "acquired" if prediction == 1 else "closed"

        return render_template('results.html', result=result)

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
