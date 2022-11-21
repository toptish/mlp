"""
Plotting losses for different models
"""
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

def main():
    """
    Plots different losses on a same graph with plotly given a long-format dataframe (csv)

    """
    df = pd.read_csv("data/compare_dfs.csv", dtype={'learning_rate': float, 'batches': int})

    app = Dash(__name__)
    app.layout = html.Div([
        html.H4('Loss comparison between models'),
        dcc.Graph(id="graph"),
        html.Label(
            'Activation function',
            style={"margin-top": "30px"}
        ),
        dcc.Checklist(
            id="checklist",
            options=["tanh", "relu", "sigmoid", "leaky_relu"],
            value=["sigmoid", "relu"],
            inline=True,
            style={"margin-top": "30px"}
        ),
        html.Label(
            'Number of batches',
            style={"margin-top": "30px"}
        ),
        dcc.RangeSlider(
            id='range_batches',
            min=1,
            max=30,
            step=1,
            value=[1, 6],
        ),
        html.Label(
            'Learning rate',
            style={"margin-top": "30px"}
            ),
        dcc.RangeSlider(
            id='range_rate',
            min=0.01,
            max=0.1,
            value=[0.01, 0.1],
        ),
        html.Label(
            'Loss type',
            style={"margin-top": "30px"}
        ),
        dcc.Checklist(
            id='losses',
            options=['train_loss', 'val_loss'],
            value=['train_loss', 'val_loss'],
        )
    ])


    @app.callback(
        Output("graph", "figure"),
        Input("checklist", "value"),
        Input("losses", "value"),
        Input("range_batches", "value"),
        Input("range_rate", "value"))
    def update_line_chart(activations, losses, batches, learning_rate):
        """
        Dash update
        :param activations: checklist activations
        :param losses: loss to display
        :param batches: range of batches number
        :param learning_rate: learning rate range (0.01, 0.1)
        :return:
        """
        batches = list(range(batches[0], batches[1]))
        fig = px.line(df[df.activation.isin(activations)
                         & df.batches.isin(batches)
                         & (df.learning_rate >= learning_rate[0])
                         & (df.learning_rate <= learning_rate[1])],
                      x="epoch", y=losses, color='name')
        return fig


    app.run_server(debug=True)


if __name__ == '__main__':
    main()
