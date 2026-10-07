from flask import Flask, render_template, request,jsonify
import pandas as pd
import pickle
import numpy as np
import json

# from sklearn import preprocessing
# label_encoder=preprocessing.LabelEncoder()

app = Flask(__name__)

# Load the trained model

model=pickle.load(open('model.pkl','rb'))
print('model loaded')

# Exact column order the model expects (built in model.py from X.columns)
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
print(f'{len(feature_columns)} feature columns loaded')

 
# Median values for every feature, used to auto-fill anything the form
# doesn't ask the user for directly (the model needs all 59 features,
# but a 59-field form isn't realistic — this is the standard workaround)
feature_defaults = pickle.load(open('feature_defaults.pkl', 'rb'))
print('feature defaults loaded')
 
 
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page
 
 
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # ---- Get the user-facing form inputs ----
            data_channel = request.form['data_channel']       # e.g. "tech"
            weekday = request.form['weekday']                 # e.g. "monday"
            n_tokens_title = float(request.form['n_tokens_title'])
            n_tokens_content = float(request.form['n_tokens_content'])
            num_imgs = float(request.form['num_imgs'])
            num_hrefs = float(request.form['num_hrefs'])
            kw_avg_avg = float(request.form['kw_avg_avg'])
            global_subjectivity = float(request.form['global_subjectivity'])
 
            print(data_channel, weekday, n_tokens_title, n_tokens_content,
                  num_imgs, num_hrefs, kw_avg_avg, global_subjectivity)
 
            # ---- Start from the median defaults for all 59 features ----
            row = feature_defaults.copy()
 
            # ---- Overwrite with the values the user actually provided ----
            row['n_tokens_title'] = n_tokens_title
            row['n_tokens_content'] = n_tokens_content
            row['num_imgs'] = num_imgs
            row['num_hrefs'] = num_hrefs
            row['kw_avg_avg'] = kw_avg_avg
            row['global_subjectivity'] = global_subjectivity
 
            # ---- One-hot: data_channel dropdown -> the matching dummy column ----
            channel_cols = [c for c in feature_columns if c.startswith('data_channel_is_')]
            for c in channel_cols:
                row[c] = 0
            selected_channel_col = f'data_channel_is_{data_channel}'
            if selected_channel_col in row:
                row[selected_channel_col] = 1
 
            # ---- One-hot: weekday dropdown -> the matching dummy column ----
            weekday_cols = [c for c in feature_columns if c.startswith('weekday_is_')]
            for c in weekday_cols:
                row[c] = 0
            selected_weekday_col = f'weekday_is_{weekday}'
            if selected_weekday_col in row:
                row[selected_weekday_col] = 1
            row['is_weekend'] = 1 if weekday in ('saturday', 'sunday') else 0
 
            # ---- Build the row in the EXACT column order the model expects ----
            input_df = pd.DataFrame([row], columns=feature_columns)
            print(input_df)
 
            # ---- Predict (model outputs log_shares — undo the log transform) ----
            log_pred = model.predict(input_df)[0]
            prediction = np.expm1(log_pred)
            print('Predicted shares:', prediction)
 
            return render_template(
                'index.html',
                prediction_text=f'Estimated Shares: {int(round(prediction))}'
            )
 
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)