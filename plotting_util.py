import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def scatter_plot(df, dim_1, dim_2, symbol_map, color_dict, markersize=10, alpha=0.85, y_log=False):
    fig = go.Figure()
    scatter = px.scatter(
        df.reset_index(),
        x=dim_1,
        y=dim_2,
        color='class',
        color_discrete_map=color_dict,
        hover_name='sample_key',
        symbol='class',
        symbol_map=symbol_map,
    )
    scatter.update_traces(marker=dict(
        size=markersize,
        opacity=alpha,
        line=dict(color='black', width=1)
    ), showlegend=True, legendgroup=1)
    fig.add_traces(scatter.data)

    fig.update_layout(
        width=1000,
        height=600,
        xaxis=dict(
            title=dim_1,
            ticks='outside',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            minor=dict(dtick=1, showgrid=True, gridcolor='rgba(0,0,0,0.04)'),
            showline=True, linewidth=1, linecolor='black', mirror=True
        ),
        yaxis=dict(
            title=dim_2,
            type='log' if y_log else 'linear',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            minor=dict(showgrid=True, gridcolor='rgba(0,0,0,0.04)'),
            showline=True, linewidth=1, linecolor='black', mirror=True,
        ),
        template='simple_white',
        legend=dict(title='Class', itemclick='toggle'),
        margin=dict(l=80, r=40, t=40, b=60)
    )
    fig.show()


def scatter_plot_1d(df, dim_1, marker_map, marker_dict, color_dict, x_log=False, markersize=10, jitter_strength=0.1, alpha_pts=0.85, random_state=42):
    fig = go.Figure()
    seen = set()
    rng = np.random.default_rng(random_state)

    for cls, group in df.groupby(level='class'):
        sample_keys = group.reset_index(level='class', drop=True).index.values
        x = group[dim_1].values
        n = x.size
        if n == 0:
            continue
        y_jitter = rng.uniform(-jitter_strength, jitter_strength, size=n)
        symbol = marker_map.get(marker_dict.get(cls, 'o'), 'circle')
        showleg = cls not in seen
        fig.add_trace(go.Scatter(
            x = x,
            y = y_jitter,
            mode = 'markers',
            name = cls,
            legendgroup = cls,
            showlegend = showleg,
            marker = dict(
                symbol = symbol,
                size = markersize,
                color = color_dict.get(cls, "gray"),
                opacity = alpha_pts,
                line = dict(color='black', width=1)
            ),
            text = sample_keys, 
            hovertemplate = f"<b>%{{text}}</b><br><br>class: {cls}<br>{dim_1}: %{{x}}<br><extra></extra>"
        ))
        seen.add(cls)  

    fig.update_layout(
        width = 1000,
        height = 300,
        xaxis = dict(
            title = dim_1,
            ticks = 'outside',
            type='log' if x_log else 'linear',
            showgrid = True, gridcolor = 'rgba(0,0,0,0.08)',
            showline=True, linewidth=1, linecolor='black', mirror=True
        ),
        yaxis = dict(
            tickmode='array', 
            tickvals=[], 
            ticktext=[''],
            range=[-1, 1],
            showgrid = True,
            gridcolor = 'rgba(0,0,0,0.08)',
            showline=True, linewidth=1, linecolor='black', mirror=True
        ),
        template = 'simple_white',
        legend = dict(title='Class', x=1.05, y=1, itemclick='toggle'),
        margin = dict(l=60, r=200, t=40, b=60)
    )

    fig.show()
    
# Function for plotting new test data along with training data
def scatter_plot_train_test(df_train, df_test, dim_1, dim_2, symbol_map, color_dict, markersize=10, y_log=False):
    fig = go.Figure()
    train_scatter = px.scatter(
        df_train.reset_index(),
        x=dim_1,
        y=dim_2,
        color='class',
        color_discrete_map=color_dict,
        hover_name='sample_key',
        symbol='class',
        symbol_map=symbol_map,
    )
    train_scatter.update_traces(marker=dict(
        size=markersize,
        opacity=0.3,
        line=dict(color='black', width=1)
    ), showlegend=False, legendgroup='train')
    fig.add_traces(train_scatter.data)

    test_scatter = px.scatter(
        df_test.reset_index(),
        x=dim_1,
        y=dim_2,
        color='class',
        color_discrete_map=color_dict,
        hover_name='sample_key',
        symbol='class',
        symbol_map=symbol_map,
    )
    test_scatter.update_traces(marker=dict(
        size=markersize,
        opacity=1.0,
        line=dict(color='black', width=1)
    ), showlegend=True, legendgroup='test')
    fig.add_traces(test_scatter.data)

    fig.update_layout(
        width=1000,
        height=600,
        xaxis=dict(
            title=dim_1,
            ticks='outside',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            minor=dict(dtick=1, showgrid=True, gridcolor='rgba(0,0,0,0.04)'),
            showline=True, linewidth=1, linecolor='black', mirror=True
        ),
        yaxis=dict(
            title=dim_2,
            type='log' if y_log else 'linear',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            minor=dict(showgrid=True, gridcolor='rgba(0,0,0,0.04)'),
            showline=True, linewidth=1, linecolor='black', mirror=True
        ),
        template='simple_white',
        legend=dict(title='Class', itemclick='toggle'),
        margin=dict(l=80, r=40, t=40, b=60)
    )
    fig.show()

def scatter_plot_train_test_1d(df_train, df_test, dim_1, marker_map, marker_dict, color_dict, x_log=False, markersize=10, jitter_strength=0.1, alpha_pts=0.85, random_state=42):
    fig = go.Figure()
    seen = set()
    rng = np.random.default_rng(random_state)

    for cls, group in df_train.groupby(level='class'):
        sample_keys = group.reset_index(level='class', drop=True).index.values
        x = group[dim_1].values
        n = x.size
        if n == 0:
            continue
        y_jitter = 0.5 + rng.uniform(-jitter_strength, jitter_strength, size=n)
        symbol = marker_map.get(marker_dict.get(cls, 'o'), 'circle')
        showleg = cls not in seen
        fig.add_trace(go.Scatter(
            x = x,
            y = y_jitter,
            mode = 'markers',
            name = cls,
            legendgroup = cls,
            showlegend = showleg,
            marker = dict(
                symbol = symbol,
                size = markersize,
                color = color_dict.get(cls, "gray"),
                opacity = alpha_pts,
                line = dict(color='black', width=1)
            ),
            text = sample_keys, 
            hovertemplate = f"<b>%{{text}}</b><br><br>class: {cls}<br>{dim_1}: %{{x}}<br><extra></extra>"
        ))
        seen.add(cls)

    for cls, group in df_test.groupby(level='class'):
        sample_keys = group.reset_index(level='class', drop=True).index.values
        x = group[dim_1].values
        n = x.size
        if n == 0:
            continue
        y_jitter = -0.5 + rng.uniform(-jitter_strength, jitter_strength, size=n)
        symbol = marker_map.get(marker_dict.get(cls, 'o'), 'circle')
        showleg = cls not in seen
        fig.add_trace(go.Scatter(
            x = x,
            y = y_jitter,
            mode = 'markers',
            name = cls,
            legendgroup = cls,
            showlegend = showleg,
            marker = dict(
                symbol = symbol,
                size = markersize,
                color = color_dict.get(cls, "gray"),
                opacity = alpha_pts,
                line = dict(color='black', width=1)
            ),
            text = sample_keys, 
            hovertemplate = f"<b>%{{text}}</b><br><br>class: {cls}<br>{dim_1}: %{{x}}<br><extra></extra>"
        ))
        seen.add(cls)

    fig.update_layout(
        width = 1000,
        height = 300,
        xaxis = dict(
            title = dim_1,
            ticks = 'outside',
            type='log' if x_log else 'linear',
            showgrid = True, gridcolor = 'rgba(0,0,0,0.08)',
            showline=True, linewidth=1, linecolor='black', mirror=True
        ),
        yaxis = dict(
            tickmode='array', 
            tickvals=[0.5, -0.5], 
            ticktext=['Train', 'Test'],
            range=[-1, 1],
            showgrid = True,
            gridcolor = 'rgba(0,0,0,0.08)',
            showline=True, linewidth=1, linecolor='black', mirror=True
        ),
        template = 'simple_white',
        legend = dict(x=1.05, y=1),
        margin = dict(l=60, r=200, t=40, b=60)
    )

    fig.show()

def plot_spectra_by_class(
    X_minmax,
    wavelengths,
    color_dict=None,
    width=1000,
    height=600,
    x_range=(370, 925),
    y_range=(-0.1, 1.1),
    opacity=0.2,
    line_width=1,
    legend_title="Class"
) -> None:
    """
    Create a Plotly figure with spectra grouped by class.
    """
    if color_dict is None:
        color_dict = {}

    fig = go.Figure()
    for cls, sub in X_minmax.groupby(level="class"):
        color = color_dict.get(cls, "black")
        sub2 = sub.reset_index(level="class", drop=True)
        for sample_key, row in sub2.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=row.values,
                    mode="lines",
                    line=dict(color=color, width=line_width),
                    opacity=opacity,
                    name=str(sample_key),
                    legendgroup=cls,
                    showlegend=False,
                )
            )
        # add class legend entry
        fig.add_trace(
            go.Scatter(
                x=[wavelengths[0]],
                y=[np.nan],
                mode="lines",
                line=dict(color=color, width=3),
                name=cls,
                legendgroup=cls,
                showlegend=True,
            )
        )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalised Counts (a.u.)",
        yaxis=dict(
            range=list(y_range),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            ticks="outside",
        ),
        xaxis=dict(
            range=list(x_range),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            ticks="outside",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(title=legend_title, itemclick="toggle"),
        template="simple_white",
        margin=dict(l=80, r=40, t=40, b=60),
    )

    fig.show()


def plot_box_parameters(
    df_data_complete: pd.DataFrame,
    param_names: "list[str]",
    width=1000,
    height=600,
    dropdown_x =1.15,
    dropdown_y=1.0,
    initial_idx=0,
    template="simple_white"
) -> None:
    """
    Create a Plotly figure with grouped boxplots for the parameter lists in `param_names`.
    A dropdown lets the user switch which parameter is visible.

    Returns the Plotly Figure.
    """

    traces = []
    # build one box trace per parameter (x=sample, y=values)
    for idx, param in enumerate(param_names):
        if param not in df_data_complete.columns:
            # skip missing parameter columns
            print(f"Parameter '{param}' not found in dataframe — skipped.")
            continue

        y = np.concatenate([np.ravel(vals) for vals in df_data_complete[param].values])
        x = np.concatenate(
            [[sample] * len(np.ravel(vals)) for sample, vals in zip(df_data_complete.index, df_data_complete[param])]
        )
        traces.append(
            go.Box(
                y=y,
                x=x,
                name=param,
                visible=(len(traces) == initial_idx),
                boxmean="sd",
            )
        )

    # build dropdown buttons (one per existing trace)
    buttons = []
    for i, trace in enumerate(traces):
        visible = [False] * len(traces)
        visible[i] = True
        buttons.append(
            dict(
                label=trace.name,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "yaxis": {
                            "title": trace.name,
                            "showline": True,
                            "linewidth": 1,
                            "linecolor": "black",
                            "mirror": True,
                            "ticks": "outside",
                        },
                        "xaxis": {
                            "showline": True,
                            "linewidth": 1,
                            "linecolor": "black",
                            "mirror": True,
                            "ticks": "outside",
                        },
                    },
                ],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[
            dict(type="dropdown", direction="down", buttons=buttons, x=dropdown_x, y=dropdown_y, showactive=True)
        ],
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True, ticks="outside"),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True, ticks="outside"),
        yaxis_title=param_names[initial_idx] if len(param_names) > initial_idx else "",
        xaxis_title="Sample",
        width=width,
        height=height,
        boxmode="group",
        template=template,
        margin=dict(l=60, r=40, t=40, b=60),
    )

    fig.show()

def plot_lda_explained_variance(evr, width: int = 1000, height: int = 600) -> None:
    """
    Plot LDA explained variance (bars) and cumulative explained variance (line).
    """

    evr = np.asarray(evr)
    if evr.size == 0:
        raise ValueError("evr is empty")

    indices = np.arange(1, len(evr) + 1)
    labels = [f"LD{i}" for i in indices]
    explained_var = evr * 100
    cumulative_var = np.cumsum(evr) * 100

    fig = go.Figure()

    # bar: explained variance
    fig.add_trace(go.Bar(
        x=labels,
        y=explained_var,
        name="Explained variance",
        marker_color="steelblue"
    ))

    # line: cumulative explained variance
    fig.add_trace(go.Scatter(
        x=labels,
        y=cumulative_var,
        mode="lines+markers",
        name="Cumulative",
        line=dict(color="black"),
        marker=dict(symbol="circle", color="black")
    ))

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title="Linear Discriminant",
        yaxis_title="Explained variance (%)",
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            tickmode="array",
            tickvals=labels,
            ticktext=labels
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)"
        ),
        legend=dict(itemclick="toggle"),
        template="simple_white",
        margin=dict(l=60, r=40, t=60, b=60)
    )

    fig.show()