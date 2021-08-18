import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statistics
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


data_path = '/Users/tushar/Documents/CS677 project/final data/5yeardata.csv'
data = pd.read_csv(data_path)
data.dropna(inplace=True)

data_path2019 = '/Users/tushar/Documents/CS677 project/final data/tennis_final_2019.csv'
data2019 = pd.read_csv(data_path2019)
data2019.dropna(inplace=True)

data_path2018 = '/Users/tushar/Documents/CS677 project/final data/tennis_final_2018.csv'
data2018 = pd.read_csv(data_path2018)
data2018.dropna(inplace=True)

data_path2017 = '/Users/tushar/Documents/CS677 project/final data/tennis_final_2017.csv'
data2017 = pd.read_csv(data_path2017)
data2017.dropna(inplace=True)

data_path2016 = '/Users/tushar/Documents/CS677 project/final data/tennis_final_2016.csv'
data2016 = pd.read_csv(data_path2016)
data2016.dropna(inplace=True)

data_path2015 = '/Users/tushar/Documents/CS677 project/final data/tennis_final_2015.csv'
data2015 = pd.read_csv(data_path2015)
data2015.dropna(inplace=True)



def player_avg_score_yearly(name):
    year_data = []
    for i in[data2015,data2016,data2017,data2018,data2019]:
        data = i
        # when player 1 = name
        player1_stats = []
        player1_data = data[data['player1_name']==name]
        for i in range(2,17):
            player1_stats.append(player1_data[player1_data.columns[i]].mean())

        # when player 2 = name
        player2_stats = []
        player2_data = data[data['player2_name']==name]
        for i in range(19,player2_data.shape[1]-1):
            player2_stats.append(player2_data[player2_data.columns[i]].mean())

        # average of above two
        player_stats = []
        for i in range(15):
            player_stats.append((player1_stats[i]+player2_stats[i])/2)

        calc = [1.0 for i in range(15)]
        calc[3] = -1
        score_arr = [a*b for a,b in zip(player_stats,calc)]
        #print(score_arr, sum(score_arr))
        year_data.append(sum(score_arr))

    return year_data



def player_avg_stats(name):

    # when player 1 = name
    player1_stats = []
    player1_data = data[data['player1_name']==name]
    for i in range(2,17):
        player1_stats.append(player1_data[player1_data.columns[i]].mean())

    # when player 2 = name
    player2_stats = []
    player2_data = data[data['player2_name']==name]
    for i in range(19,player2_data.shape[1]-1):
        player2_stats.append(player2_data[player2_data.columns[i]].mean())

    # average of above two
    player_stats = []
    for i in range(15):
        player_stats.append((player1_stats[i]+player2_stats[i])/2)

    return player_stats





def win_predictor(player1,player2):
    res = []
    p1 = [round(x) for x in player_avg_stats(player1)]
    p2 = [round(x) for x in player_avg_stats(player2)]
    res.append(p1)
    res.append(p2)
    prediction_data = player_avg_stats(player1) + player_avg_stats(player2)
    winner = model.predict(np.asmatrix(prediction_data))

    x = data[ ((data['player1_name']==player1)&(data['player2_name']==player2)) | ((data['player1_name']==player2)&(data['player2_name']==player1))]
    #print(x.shape)
    player1_h2h_victory = 0
    player2_h2h_victory = 0
    if x.shape[0]!=0:
        for index,row in x.iterrows():
            print(row['Player1'],row['player1_name'],row['Player2'],row['player2_name'],row['winner'])
            if (row['Player1']==row['winner'] and row['player1_name']==player1) or (row['Player2']==row['winner'] and row['player2_name']==player1):
                player1_h2h_victory += 1
            else:
                player2_h2h_victory += 1
    #print(player1_h2h_victory,player2_h2h_victory)
    res.extend([player1_h2h_victory,player2_h2h_victory])
    if winner[0]=='p1':
        res.append(player1)
    else:
        res.append(player2)

    df = pd.DataFrame(index = ['head_to_head', 'avg_serve_rating',
       'avg_return_rating', 'avg_aces', 'avg_double_faults',
       'avg_first_serve_percent', 'avg_win%_on_first_serve',
       'avg_win%_on_second_serve', 'avg_break_points_faced',
       'avg_break_points_converted', 'avg_tiebreaks_won',
       'avg_return_points_won', 'avg_total_points_won',
       'avg_games_won', 'avg_service_points_won',
       'avg_service_games_won'])

    res[0].insert(0,res[2])
    res[1].insert(0,res[3])
    df[player1] = res[0]
    df[player2] = res[1]
    print(df)
    print(res[4])
    return res


@app.route('/')
def home():
    option_ls = data['player1_name'].unique()
    option_ls.sort()
    return render_template('home.html',options = option_ls)


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        player1 = request.form.get('player1')
        player2 = request.form.get('player2')
        #method = request.form.get('basis')
        res = win_predictor(player1,player2)
        print(res)
        columns = ['Head to Head', 'Average Serve Rating',
           'Average Return Rating', 'Average Aces', 'Average Double Faults',
           'Average 1st Serve%', 'Average Win% on 1st Serve',
           'Average Win% on 2nd Serve', 'Average Break Points Faced',
           'Average Break Points Converted', 'Average Tiebreaks Won',
           'Average Return Points Won', 'Average Total Points Won',
           'Average Games Won', 'Average Service Points Won',
           'Average Service Games Won']

        # columns_pred = ['Head to Head', 'Predicted Serve Rating',
        #    'Predicted Return Rating', 'Predicted Aces', 'Predicted Double Faults',
        #    'Predicted 1st Serve%', 'Predicted Win% on 1st Serve',
        #    'Predicted Win% on 2nd Serve', 'Predicted Break Points Faced',
        #    'Predicted Break Points Converted', 'Predicted Tiebreaks Won',
        #    'Predicted Return Points Won', 'Predicted Total Points Won',
        #    'Predicted Games Won', 'Predicted Service Points Won',
        #    'Predicted Service Games Won']
        ls = zip(res[0],columns,res[1])



    return render_template('test.html', output=res,ls=ls, p1 = player1, p2 = player2)




@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig, ax = plt.subplots(figsize = (3,5))
    fig.patch.set_facecolor('#E8E5DA')

    x = [2015,2016,2017,2018,2019]
    y = player_avg_score_yearly('Novak Djokovic')

    ax.plot(x, y, color = "#304C89")
    plt.xlabel('Year')
    plt.ylabel('Score')

    return fig









if __name__ == "__main__":
    app.run(debug=True)
