"""
Prograam for scatter plot matrix of dataset features
"""
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from data_analyser import DataAnalyser


def main():
    """
    Plots the initial data scatter matrix.
    """
    dataset = DataAnalyser()
    dataset.load_data()

    app = Dash(__name__)

    app.layout = html.Div([
        html.H4('Analysis of Dataset using scatter matrix'),
        dcc.Dropdown(
            id="dropdown",
            options=list(range(2, 32)),
            value=[2, 4, 5],
            multi=True
        ),
        dcc.Graph(id="graph"),
    ])

    @app.callback(
        Output("graph", "figure"),
        Input("dropdown", "value"))
    def update_bar_chart(dims):
        df_data = dataset.data
        fig = px.scatter_matrix(
            df_data, dimensions=dims, color=1)
        return fig

    app.run_server(debug=True)


if __name__ == '__main__':
    main()
