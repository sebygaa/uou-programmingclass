
# %%
from flask import Flask
import dash
from dash import dcc, html
import plotly.express as px

# Initialize the Flask server
server = Flask(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/')

# Create a simple dataframe
df = px.data.iris()  # Using an example dataset from Plotly

# Create a scatter plot figure
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species',
                 title="Iris Dataset: Sepal Width vs Sepal Length")

# Layout for the Dash app
app.layout = html.Div(children=[
    html.H1(children='Flask + Dash + Plotly Example'),

    html.Div(children='''
        A simple example of a Dash app running on a Flask server!!!!
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

# Flask route for the main page
@server.route('/')
def home():
    return "Welcome to the Flask home page! Go to /dash to see the Dash app."

# Run the server
if __name__ == '__main__':
    server.run(debug=True)