# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from textwrap import dedent

import glob
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff

DEBUG = True

app = dash.Dash(__name__)
server = app.server # the Flask app

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True

colors = {
    'black': '#000000',
    'changes': '#2420AC',
    'gt_true':'#1a92c6',
    'gt_false':'#cce5ff',
    'pr_true':'#ff8000',
    'pr_false':'#ffe5cc',
    'heatmap_font_1':'#3c3636',
    'heatmap_font_2':'#efecee'
}

# Load Data
def load_data(path,dataset):
    # Load the dataframe
    label_path=glob.glob(path)
    video_info_df = pd.read_csv(label_path[0])

    video_length = len(video_info_df)

    file_ptr = open(dataset, 'r')
    class_act = file_ptr.read().split()[:]
    file_ptr.close()
    classes_list = []
    for i,a in enumerate(class_act):
         if i % 2 == 1:
             classes_list.append(a)

    #file_ptr = open(dataset, 'r')
    #class_map = file_ptr.read().split('\n')[:-1]
    #file_ptr.close()
    #classes_dict = dict()
    #for a in class_map:
    #     classes_dict[a.split()[1]] = int(a.split()[0])

    n_classes = len(classes_list)
        
    # Round the number of classes
    total_size = np.round(n_classes + 0.5)
    padding_value = int(total_size - n_classes)

    #"0"クラスの追加
    if n_classes % 2 == 1:
         classes_padded = np.pad(classes_list, (0, padding_value), mode='constant')
    
    else:
         classes_padded = classes_list

    classes_matrix = np.reshape(classes_padded, (2, int(total_size/2)))
    #class_padを2行にしてフリップ
    classes_matrix = np.flip(classes_matrix, axis=0)
    
    #add to data_dict
    data_dict = {
            "video_length": video_length,
            #"classes_dict": classes_dict,
            "video_info_df": video_info_df,
            "n_classes": n_classes,
            "classes_matrix": classes_matrix,
            "classes_padded": classes_padded,
            "total_size": total_size
    }
    if DEBUG:
        print('{} loaded.'.format(path))
    return data_dict


# load video
def load_video(path):
    video_path=glob.glob(path)
    return os.path.join(video_path[0]+"/")


def get_heatmap(data_dict, frame_num):
    video_df = data_dict["video_info_df"]
    classes_padded = data_dict["classes_padded"]
    total_size = data_dict["total_size"]
    classes_matrix = data_dict["classes_matrix"]
    # The list of scores   "class_str_top1"=最高スコアのクラス　
    score_list = []
    for el in classes_padded:
        if el in video_df["predict"][frame_num]:
            score_list.append(1)
        else:
            score_list.append(0)

    # Generate the score matrix, and flip it for visual
    score_matrix = np.reshape(score_list, (-1, int(total_size/2)))
    score_matrix = np.flip(score_matrix, axis=0)
    # color scale
    colorscale = [[0, colors['pr_false']], [1,colors['pr_true']]]
    font_colors = [colors['heatmap_font_1'], colors['heatmap_font_2']]
    # Hover Text
    hover_text = ['{:.2f}% confidence'.format(score * 100) for score in score_list]
    hover_text = np.reshape(hover_text, (-1, int(total_size/2)))
    hover_text = np.flip(hover_text, axis=0)
    return score_matrix, classes_matrix, colorscale, font_colors, hover_text


def get_correct_label(data_dict, frame_num):
    video_df = data_dict["video_info_df"]
    classes_padded = data_dict["classes_padded"]
    total_size = data_dict["total_size"]
    classes_matrix = data_dict["classes_matrix"]
    # The list of scores   "class_str_top1"=最高スコアのクラス　"Top1_score"=そのスコア値
    score_list = []
    for el in classes_padded:
        if el in video_df["gt"][frame_num]:
            score_list.append(1)
        else:
            score_list.append(0)

    # Generate the score matrix, and flip it for visual
    score_matrix = np.reshape(score_list, (-1, int(total_size/2)))
    score_matrix = np.flip(score_matrix, axis=0)
    # color scale
    colorscale = [[0, colors['gt_false']], [1, colors['gt_true']]]
    font_colors = [colors['heatmap_font_1'], colors['heatmap_font_2']]
    # Hover Text
    hover_text = ['{:.2f}% confidence'.format(score * 100) for score in score_list]
    hover_text = np.reshape(hover_text, (-1, int(total_size/2)))
    hover_text = np.flip(hover_text, axis=0)
    return score_matrix, classes_matrix, colorscale, font_colors, hover_text

def get_length_of_video(data_dict):

    videos_length = data_dict["video_length"]

    return videos_length



    # Main App
app.layout = html.Div([
    # Banner display
    html.Div([
        html.H2(
            'Action Segmentation',
            id='title',
            style={
                'height': '110px',
                'margin-top': '15px',
                'margin-bottom': '0px'
                }
        ),
        html.H2(
            'Keio Aoki lab.　Takuya Kobayashi',
            id='name',
            style={
                'height': '110px',
                'font-size': '1.5em',
                'margin-top': '15px',
                'margin-left': '200px',
                'margin-bottom': '0px'
                }
        ),
        html.Img(
            src="https://wwwdc05.adst.keio.ac.jp/kj/vi/common/img/thumbF2.png",
            style={
                'height': '60px',
                'margin-top': '10px',
                'margin-bottom': '0px'
                }
        )
    ],
        className="banner",
        style={
                'background-color': '#66b2ff',
                'height': '75px',
                'padding-top': '0px',
                'padding-left': '0px',
                'padding-right': '0px',
                'width': '100%',
                'margin-bottom': '10px'
            }
                 
    ),

    # Body
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Img(
                        style={
                            'width': '100%',#600,
                            'height': '400px',#400,
                            'margin': '0px 20px 15px 15px'
                        },
                        id="images",
                        )
                    ],
                    id='div-video-player',
                    style={
                        'color': 'rgb(255,255,255)',
                        'margin-bottom': '0px'
                    }
                ),
                
                html.Div([html.Div([
                    "Dataset",
                    dcc.Dropdown(
                        options=[
                            {'label':'gtea','value':'gtea'},
                            {'label':'50salads','value':'50salads'},
                            {'label':'breakfast','value':'breakfast'}
                     ],
                        value='gtea',
                        id='dataset-selection',
                        clearable=False,
                        style={'color': colors['black']}
                    )
                    ],
                    style={'margin': '15px 20px 20px 10px','vertical-align': 'middle',
                           'color': colors['changes']},
                    className='four columns'
                ),
                # chose video 
                html.Div([
                    "Video ",
                    dcc.Dropdown(
                        options=[
                            {'label': 'Video A', 'value': 'video_a'},
                            {'label': 'Video B', 'value': 'video_b'}
                            
                     ],
                        value='video_a',
                        id='dropdown-video-selection',
                        clearable=False,
                        style={'color': colors['black']}
                    )
                    ],
                    style={'margin': '15px 20px 20px 70px','vertical-align': 'middle',
                           'color': colors['changes']},
                    className='four columns'
                )


                ],
                className='row'

                ),

                #chose play mode
                html.Div([html.Div([
                    "Play Mode",
                    dcc.RadioItems(
                        options=[
                            {'label': ' Manual ', 'value': 24*60*60*1000},
                            {'label': ' Auto Play ', 'value': 1*300} #1*430
                        ],
                        value=24*60*60*1000,
                        id='set-time',   
                        style={'color': colors['black']}                    
                    )
                ],
                    style={'margin': '5px 20px 10px 10px',
                            'color': colors['changes']},
                    className='six columns'
                    
                )

                
                ],
                className='row'

                ),

                 #  frame start holder
                html.Div(
                        id='frame-start-holder',
                        style={'display': 'none'}
                ),

                #  frame number holder
                html.Div(
                        id='frame-number-holder',
                        style={'display': 'none'}
                ),
                
                 #chose frame number
                html.Div(
                        id='slider-frame',
                        style={'margin': '10px 10px 30px 5px','width':'89%',
                                'color': colors['changes']}
                ),

                html.Div(
                        id='slider-auto-frame',
                        style={'margin': '10px 10px 20px 5px','width':'89%',
                               'color': colors['changes']}
                ),
                dcc.Interval(
                id='interval-component',
                n_intervals=0
                )

                
                      
                
            ],
                className="six columns",
                style={'margin-bottom': '20px'}
            ),

         

            #  Heatmap Area and Classification Score
            html.Div(id="div-visual-mode", 
                     style={'margin-top': '30px'},
                     className="six columns"
            )
        ],
            
            className="row"
        ),
        
        ],
        className="container scalable"
    )

])






# PATH for datas
RESULT_PATH = os.path.join(os.getcwd(), 'datas/result/')
CLASS_PATH = os.path.join(os.getcwd(), 'datas/mapping/')

# Data Loading
@app.server.before_first_request
def load_all_match():
    global data_dict, url_dict
    data_dict = {'gtea': {}, '50salads': {}, 'breakfast': {}}
    data_dict['gtea'] = {'video_a': load_data(os.path.join(RESULT_PATH+"gtea","data_a","label/*"),os.path.join(CLASS_PATH+"gtea.txt")),
                         'video_b': load_data(os.path.join(RESULT_PATH+"gtea","data_b","label/*"),os.path.join(CLASS_PATH+"gtea.txt"))}
    #data_dict['50salads'] = {'video_a': load_data(os.path.join(RESULT_PATH+"50salads","data_a","label/*"),os.path.join(CLASS_PATH+"50salads.txt")),
    #                     'video_b': load_data(os.path.join(RESULT_PATH+"50salads","data_b","label/*"),os.path.join(CLASS_PATH+"50salads.txt"))}
    #data_dict['breakfast'] ={'video_a': load_data(os.path.join(RESULT_PATH+"breakfast","data_a","label/*"),os.path.join(CLASS_PATH+"breakfast.txt")),
    #                     'video_b': load_data(os.path.join(RESULT_PATH+"breakfast","data_b","label/*"),os.path.join(CLASS_PATH+"breakfast.txt"))}

    url_dict = {'gtea': {}, '50salads': {}, 'breakfast': {}}
    url_dict['gtea'] = {'video_a': load_video(os.path.join(RESULT_PATH +"gtea","data_a","image/*")),
                         'video_b': load_video(os.path.join(RESULT_PATH +"gtea","data_a","image/*"))}
    #url_dict['50salads'] = {'video_a': load_video(os.path.join(RESULT_PATH +"50salads","data_a","image/*")),
    #                     'video_b': load_video(os.path.join(RESULT_PATH +"50salads","data_a","image/*"))}
    #url_dict['breakfast'] = {'video_a': load_video(os.path.join(RESULT_PATH +"breakfast","data_a","image/*")),
    #                     'video_b': load_video(os.path.join(RESULT_PATH +"breakfast","data_a","image/*"))}

           
# main or auto
@app.callback(Output('interval-component', 'interval'),
             [Input('set-time', 'value')])
def update_interval(value):
    return value

# auto mode max interval
@app.callback(Output('interval-component', 'max_intervals'),
             [Input("dataset-selection", "value"),
             Input("dropdown-video-selection", "value")])
def update_max_intervals(dataset,video):
    video_length = get_length_of_video(data_dict[dataset][video])
    return video_length


# frame start holder
@app.callback(Output("frame-start-holder", "children"),
             [Input("slider-frame-position", "value")]
             )
def update_frame_number(start):
    return start


# frame number holder
@app.callback(Output("frame-number-holder", "children"),
             [Input("frame-start-holder", "children"),
              Input("interval-component", "n_intervals")]
             )
def update_frame_number(start,inter):
    return start+inter


# n_interval reset
@app.callback(Output("interval-component", "n_intervals"),
             [Input("frame-start-holder", "children")]
             )
def update_frame_number(start):
    return 0


# frame start slider
@app.callback(Output("slider-frame", "children"),
             [Input("dataset-selection", "value"),
              Input("dropdown-video-selection", "value")]
             )
def update_frame_select(dataset,video):
      video_length = get_length_of_video(data_dict[dataset][video])
      return   [
                "Frame Position",
                    dcc.Slider(
                        min=0,
                        max=video_length,
                        marks={i: '{}th'.format(i) for i in range(0,video_length+1,200)},
                        value=0,
                        updatemode='mouseup',
                        included = False,
                        id='slider-frame-position'
                        )
                    ]


# frame range slider
@app.callback(Output("slider-auto-frame", "children"),
             [Input("dataset-selection", "value"),
              Input("dropdown-video-selection", "value"),
              Input("frame-start-holder", "children"),
              Input("interval-component", "n_intervals")]
             )
def update_frame_auto_select(dataset,video,start,inter):
      video_length = get_length_of_video(data_dict[dataset][video])
      if start == None:
          start = 0
      print("frame {}".format(start+inter))
      return   [
                "(Auto Play)",
                    dcc.RangeSlider(
                        min=0,
                        max=video_length,
                        marks={i: '' for i in range(0,video_length+1,200)},
                        value=[start,start+inter],
                        updatemode='mouseup',
                        id='slider-frame-auto-position'
                        )
                    ]


# Images Display
@app.callback(Output("images", "src"),
             [Input("frame-number-holder", "children")],
             [State("dataset-selection", "value"),
              State("dropdown-video-selection", "value")])
# frame=0~120 , video="working_a"とか
#frame_nameは.csv file内の"Frames"を返す
def update_image_src(frame, dataset, video):
    video_df = data_dict[dataset][video]["video_info_df"]
    frame_name = video_df["image"][frame]
    return url_dict[dataset][video] + frame_name

@app.server.route('{}<path:image_path>.png'.format(RESULT_PATH))
def serve_image(image_path):
    img_name = '{}.png'.format(image_path)
    #return flask.send_from_directory(STATIC_PATH, img_name)
    return flask.send_file(RESULT_PATH + img_name)


# Graph layout
@app.callback(Output("div-visual-mode", "children"),
             [Input("dataset-selection", "value"),
              Input("dropdown-video-selection", "value")])
def update_visual(dataset,video):
    return [
            dcc.Graph(
                style={'margin': '10px 0px 40px 10px',
                       'height': '35vh'},
                id="correct-label"
            ),
            dcc.Graph(
                style={'margin': '10px 0px 40px 10px',
                       'height': '35vh'},
                id="heatmap-confidence"
            )
    ]


# Update GT
@app.callback(Output("correct-label", "figure"),
             [Input("frame-number-holder", "children")],
             [State("dataset-selection", "value"),
              State("dropdown-video-selection", "value")])
def update_heatmap(frame, dataset, video):

    layout = go.Layout(
        title="Ground Truth",
        margin=go.layout.Margin(l=50, r=0, t=50, b=30)
    )
    scoreMatrix, classMatrix, colorScale, fontColors, hoverText = get_correct_label(data_dict[dataset][video], frame)

    pt = ff.create_annotated_heatmap(
            z=scoreMatrix,
            annotation_text=classMatrix,
            colorscale=colorScale,
            font_colors=fontColors,
            hoverinfo='text',
            text=hoverText,
            zmin=0,
            zmax=1
    )
    pt.layout.title = layout.title
    pt.layout.margin = layout.margin
    return pt


# Update Predict
@app.callback(Output("heatmap-confidence", "figure"),
             [Input("frame-number-holder", "children")],
             [State("dataset-selection", "value"),
              State("dropdown-video-selection", "value")])
def update_heatmap(frame, dataset, video):
    layout = go.Layout(
        title="Predicted",
        margin=go.layout.Margin(l=50, r=0, t=50, b=30)
    )
    scoreMatrix, classMatrix, colorScale, fontColors, hoverText = get_heatmap(data_dict[dataset][video], frame)

    pt = ff.create_annotated_heatmap(
            z=scoreMatrix,
            annotation_text=classMatrix,
            colorscale=colorScale,
            font_colors=fontColors,
            hoverinfo='text',
            text=hoverText,
            zmin=0,
            zmax=1
    )
    pt.layout.title = layout.title
    pt.layout.margin = layout.margin
    return pt

external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",  # Normalize the CSS
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto"  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    "https://cdn.rawgit.com/plotly/dash-object-detection/875fdd6b/custom-styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

if __name__ == '__main__':
    app.run_server(debug=DEBUG)