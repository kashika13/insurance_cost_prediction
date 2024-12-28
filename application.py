import pickle
from flask import Flask, request, render_template

application = Flask(__name__)
app = application

# Load the model and scaler
lasso_model = pickle.load(open('models/lasso.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Collect the data from the form
        age = float(request.form.get('age'))
        sex = 1 if request.form.get('sex') == 'male' else 0
        bmi = float(request.form.get('bmi'))
        children = int(request.form.get('children'))
        smoker = 1 if request.form.get('smoker') == 'yes' else 0
        region = request.form.get('region')
        
        # One-hot encode the region
        region_southeast = 0
        region_southwest = 0
        region_northeast = 0
        region_northwest = 0
        
        if region == 'southeast':
            region_southeast = 1
        elif region == 'southwest':
            region_southwest = 1
        elif region == 'northeast':
            region_northeast = 1
        elif region == 'northwest':
            region_northwest = 1
        
        # Create the final input array with one-hot encoded region
        input_data = [[age, sex, bmi, children, smoker, region_northeast,region_northwest,region_southeast, region_southwest]]
        
        # Scale the input data
        scaled_data = standard_scaler.transform(input_data)

        # Predict the charges
        result = lasso_model.predict(scaled_data)

        # Return the result to the HTML template
        return render_template('index.html', results=result[0])
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
