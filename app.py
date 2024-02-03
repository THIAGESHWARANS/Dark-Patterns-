from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import requests
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load your models
model_detect = pickle.load(open('model_detect.pkl', 'rb'))
model_presence = pickle.load(open('model_presence.pkl', 'rb'))

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all text content within the HTML document and split into lines
        text_content = soup.get_text("\n", strip=True)
        text_lines = text_content.split("\n")
        text_df = pd.DataFrame({'Index': range(1, len(text_lines) + 1), 'Text Content': text_lines})

        presence_predictions = model_presence.predict(text_lines)
        final_predictions = []

        for presence_prediction, text_line in zip(presence_predictions, text_lines):
            if presence_prediction == 'Dark':
                # Run model_detect if 'Dark' is predicted
                detect_prediction = model_detect.predict([text_line])[0]
                final_predictions.append(detect_prediction)
            else:
                # If 'Not Dark' is predicted, append 'Not Dark' to the final predictions
                final_predictions.append('Not Dark')
        text_df['Final Predictions'] = final_predictions

        return text_df

    except requests.exceptions.RequestException as err:
        # Handle request exceptions
        return None  # Returning None to indicate error

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('dark.html')


@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/index')
def find():
    return render_template('index.html')




@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        url = request.form['url']
        # Call the function to scrape website and perform analysis
        text_df = scrape_website(url)

        #if text_df is None:
            #return render_template('error.html', message="An error occurred while fetching the website content.")

        # Filter text_df to include only rows with final predictions as "Dark"
        dark_text_df = text_df[text_df['Final Predictions'] != 'Not Dark']

        if dark_text_df.empty:
            return render_template('error.html', message="No dark patterns were detected in the provided URL.")

        # Pie chart
        dark_patterns_count = dark_text_df['Final Predictions'].value_counts()
        labels = dark_patterns_count.index.tolist()
        values = dark_patterns_count.values.tolist()

        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Percentage of Dark Patterns')
        plt.axis('equal')
        pie_chart_path = 'static/pie_chart.png'
        plt.savefig(pie_chart_path)
        plt.close()

        # Calculate the count of 'Not Dark' predictions in the entire DataFrame
        not_dark_count_total = text_df['Final Predictions'].value_counts().get('Not Dark', 0)

        # Calculate the count of 'Not Dark' predictions and other keywords in the filtered DataFrame
        not_dark_count_filtered = dark_text_df['Final Predictions'].value_counts().get('Not Dark', 0)
        other_keywords_count_filtered = dark_text_df.shape[0] - not_dark_count_filtered

        # Calculate the count of 'Other Keywords' (excluding 'Not Dark') in the entire DataFrame
        other_keywords_count_total = text_df.shape[0] - not_dark_count_total

        # Adjust the counts to include 'Not Dark' predictions in the bar chart
        not_dark_count_display = not_dark_count_total - not_dark_count_filtered
        other_keywords_count_display = other_keywords_count_total - other_keywords_count_filtered

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(['Not Dark'], [not_dark_count_display, other_keywords_count_display], color=['#1f77b4'])
        plt.xlabel('Prediction')
        plt.ylabel('Count')
        plt.title('Distribution of Dark Patterns in Text Content')
        bar_chart_path = 'static/bar_chart.png'
        plt.savefig(bar_chart_path)
        plt.close()

        # Pass the paths of the generated charts and the filtered DataFrame to the template
        return render_template('result.html', pie_chart_path=pie_chart_path, bar_chart_path=bar_chart_path, text_df=dark_text_df)


if __name__ == '__main__':
    app.run(debug=True)
