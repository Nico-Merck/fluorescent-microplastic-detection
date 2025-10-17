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

    for class_ in df.index.get_level_values("class").unique():
        idx = df.index.get_level_values("class") == class_
        x = df.loc[idx, dim_1].values
        n = x.size
        if n == 0:
            continue
        y_jitter = rng.uniform(-jitter_strength, jitter_strength, size=n)
        symbol = marker_map.get(marker_dict.get(class_, 'o'), 'circle')
        showleg = class_ not in seen
        fig.add_trace(go.Scatter(
            x = x,
            y = y_jitter,
            mode = 'markers',
            name = class_,
            legendgroup = class_,
            showlegend = showleg,
            marker = dict(
                symbol = symbol,
                size = markersize,
                color = color_dict.get(class_, "gray"),
                opacity = alpha_pts,
                line = dict(color='black', width=1)
            ),
            hovertemplate = f"{class_}<br>{dim_1}: %{{x}}<extra></extra>"
        ))
        seen.add(class_)

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

    for class_ in df_train.index.get_level_values("class").unique():
        idx = df_train.index.get_level_values("class") == class_
        x = df_train.loc[idx, dim_1].values
        n = x.size
        if n == 0:
            continue
        y_jitter = 0.5 + rng.uniform(-jitter_strength, jitter_strength, size=n)
        symbol = marker_map.get(marker_dict.get(class_, 'o'), 'circle')
        showleg = class_ not in seen
        fig.add_trace(go.Scatter(
            x = x,
            y = y_jitter,
            mode = 'markers',
            name = class_,
            legendgroup = class_,
            showlegend = showleg,
            marker = dict(
                symbol = symbol,
                size = markersize,
                color = color_dict.get(class_, "gray"),
                opacity = alpha_pts,
                line = dict(color='black', width=1)
            ),
            hovertemplate = f"{class_}<br>{dim_1}: %{{x}}<extra></extra>"
        ))
        seen.add(class_)

    for class_ in df_test.index.get_level_values("class").unique():
        idx = df_test.index.get_level_values("class") == class_
        x = df_test.loc[idx, dim_1].values
        n = x.size
        if n == 0:
            continue
        y_jitter = -0.5 + rng.uniform(-jitter_strength, jitter_strength, size=n)
        symbol = marker_map.get(marker_dict.get(class_, 'o'), 'circle')
        showleg = class_ not in seen
        fig.add_trace(go.Scatter(
            x = x,
            y = y_jitter,
            mode = 'markers',
            name = class_,
            legendgroup = class_,
            showlegend = showleg,
            marker = dict(
                symbol = symbol,
                size = markersize,
                color = color_dict.get(class_, "gray"),
                opacity = alpha_pts,
                line = dict(color='black', width=1)
            ),
            hovertemplate = f"{class_}<br>{dim_1}: %{{x}}<extra></extra>"
        ))
        seen.add(class_)

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

