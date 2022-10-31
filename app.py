import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt
import plotly.graph_objects as go
shap.initjs()

#_________   Initialisation______________

X_test_10 = pd.read_csv('X_test_10.csv')
lgbm_model = pickle.load(open('LGBM_best_model.pickle', 'rb'))

#__________ Traitement de X_test_10 ______________

#X_test_10 = X_test_10.reset_index()
list_ID = X_test_10['level_0'].to_list()
list_features = list(X_test_10)


#___________ Paramètres de la page_______________
page_title = "Analyse locale et globale de vos clients"
page_icon = ":money_with_wings:"
layout = "centered"


#___________Affichage_infos_client_________________________

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

st.subheader('Veuillez selectionner un client :')
option_2 = st.selectbox('Identifiants :',list_ID)


display_client = X_test_10.loc[X_test_10['level_0']==option_2]

url = 'http://127.0.0.1:8000/Test'
#url = 'https://dry-peak-32185.herokuapp.com/Test'

data = {
          "ID": option_2
        }
response = requests.post(url, json=data)
response_dict = response.json()

probabilité = response_dict[0]["probability"]
prédiction = response_dict[0]["prediction"]

st.write('Voici les informations du client', option_2,':')

st.dataframe(display_client)

#st.write(response.json())
if prédiction==1 :
    st.write('Le client', option_2,' est éligible à un prêt.')
else :
    st.write('Le client', option_2,' n\'est pas éligible à un prêt, augmenter sa valeur de prédiction au dessus de 50% lui permettra de le devenir.')
#st.write(response_dict[0]["probability"])

#________Curseur_________________________________________

proba_curseur = round(probabilité*100, 2)

option = {
    "tooltip": {
        "formatter": '{a} <br/>{b} : {c}%'
    },
    "series": [{
        "name": '进度',
        "type": 'gauge',
        "startAngle": 180,
        "endAngle": 0,
        "progress": {
            "show": "true"
        },
        "radius":'100%', 

        "itemStyle": {
            "color": '#58D9F9',
            "shadowColor": 'rgba(0,138,255,0.45)',
            "shadowBlur": 10,
            "shadowOffsetX": 2,
            "shadowOffsetY": 2,
            "radius": '55%',
        },
        "progress": {
            "show": "true",
            "roundCap": "true",
            "width": 15
        },
        "pointer": {
            "length": '60%',
            "width": 8,
            "offsetCenter": [0, '5%']
        },
        "detail": {
            "valueAnimation": "true",
            "formatter": '{value}%',
            "backgroundColor": '#58D9F9',
            "borderColor": '#999',
            "borderWidth": 4,
            "width": '60%',
            "lineHeight": 20,
            "height": 20,
            "borderRadius": 188,
            "offsetCenter": [0, '40%'],
            "valueAnimation": "true",
        },
        "data": [{
            "value": proba_curseur,
            "name": 'Valeur de prédiction'
        }]
    }]
};

#st_echarts(options=option, key="1")

#________Feature_importance_locale________________________

#display_f_importance = display_client.drop([

#explainer = shap.TreeExplainer(lgbm_model)
#shap_values = explainer.shap_values(display_client)
# Summary plot
#shap.plots.bar(shap_values, max_display=15)

X_test_10_ = X_test_10.drop(['Unnamed: 0', 'index', 'prediction', 'probability'], axis=1)

 # Test set sans l'identifiant
X_bar = X_test_10_.set_index('level_0')
# Entraînement de shap sur le train set
bar_explainer = shap.Explainer(lgbm_model, X_bar)
bar_values = bar_explainer(X_bar, check_additivity=False)

def interpretabilite():
    ''' Affiche l'interpretabilite du modèle
    '''
    html_interpretabilite="""
        <div class="card">
            <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                  background: white; padding-top: 5px; width: auto;
                  height: 40px;">
                  <h3 class="card-title" style="background-color:white; color:Black;
                      font-family:Georgia; text-align: center; padding: 0px 0;">
                      SHAP Values
                  </h3>
            </div>
        </div>
        """

    # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
    if st.checkbox("Interpretabilité du modèle"):     
        
        st.markdown(html_interpretabilite, unsafe_allow_html=True)

        with st.spinner('**Affiche l\'interpretabilité du modèle...**'):                 
                       
            with st.expander('interpretabilité du modèle',
                              expanded=True):
                
                explainer = shap.TreeExplainer(lgbm_model)
                
                client_index = X_test_10_[X_test_10_['level_0'] == option_2].index.item()
                X_shap = X_test_10_.set_index('level_0')
                X_test_courant = X_shap.iloc[client_index]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)
                
                shap_values_courant = explainer.shap_values(X_test_courant_array)
                
                col1, col2 = st.columns([1, 1])
                # BarPlot du client courant
                with col1:

                    plt.clf()
                    

                    # BarPlot du client courant
                    shap.plots.bar(bar_values[client_index], max_display=40)
                    
                    fig = plt.gcf()
                    fig.set_size_inches((10, 20))
                    # Plot the graph on the dashboard
                    st.pyplot(fig)
     
                # Décision plot du client courant
                with col2:
                    plt.clf()

                    # Décision Plot
                    shap.decision_plot(explainer.expected_value[1], shap_values_courant[1],
                                    X_test_courant)
                
                    fig2 = plt.gcf()
                    fig2.set_size_inches((10, 15))
                    # Plot the graph on the dashboard
                    st.pyplot(fig2)
                    
st.subheader('Interpretabilité du modèle : Quelles variables sont les plus importantes ?')
interpretabilite()

#________Analyse_comparative_____________________

st.subheader('Analyse comparative entre le clients courant et les classes :')

feature_1 = st.selectbox('Selectionnez la feature à comparer :',list_features)

X_validé = X_test_10.loc[X_test_10['prediction']==1]
X_val = X_validé[feature_1].mean()
X_refusé = X_test_10.loc[X_test_10['prediction']==0]
X_ref = X_refusé[feature_1].mean()
X_client = display_client[feature_1].values.item()

clients = pd.DataFrame([["Moyenne_des_clients_validé", X_val], ["Moyenne_des_client_refusé", X_ref], ["Client_courant", X_client]], columns=["Clients","Valeur"])

fig = px.bar(clients, x='Clients', y=["Valeur"], barmode='group', height=400)

st.plotly_chart(fig)

#______SHAP_Global_____________________





#________Analyse_bivariée_____________________

feature_2 = st.selectbox('Selectionnez la première feature :',list_features)
feature_3 = st.selectbox('Selectionnez la deuxième feature :',list_features)

x=X_test_10[feature_2]
y=X_test_10[feature_3]

plot = px.scatter(x=x, y=y)

client_point = plot.add_trace(go.Scatter(x=display_client[feature_2].values, y=display_client[feature_3].values, mode = 'markers', marker_symbol = 'star', marker_size = 15))

st.plotly_chart(plot)















