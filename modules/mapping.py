import plotly.express as px

def plot_well_map(df):
    fig = px.scatter(
        df,
        x="Surface X",
        y="Surface Y",
        color="Well symbol",
        text="Name",
        hover_data=["Target"],
    )

    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")

    fig.update_traces(textposition="top center")

    fig.update_layout(
        title="Well Location Map",
        height=600
    )

    return fig