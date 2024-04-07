# ------------------------- Imports Libraries ------------------------------

import numpy as np
import streamlit as st
import pandas as pd
import plotly.io as pio
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
pio.templates.default = "plotly"
# import joblib
# import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------


data = pd.read_csv("diabetes.csv")


st.sidebar.header("Paramètres")


# # --------------------------------------------------------------------------


custom_color_text = st.sidebar.color_picker("Choisir la couleur des textes", "#000000")

def apply_custom_color_text(custom_color_text):
    custom_css = f"""
    <style>
        body, p, .st-emotion-cache-ue6h4q, .e1y5xkzn3{{
            color: {custom_color_text};
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_color_text(custom_color_text)


# --------------------------------------------------------------------------


custom_color_background_select_button = st.sidebar.color_picker("Choisir une couleur pour les selectboxes", "#FFFFFF")

def apply_custom_color_background_select_button(custom_color_background_select_button):
    custom_css = f"""
    <style>
        .st-an, .st-ao, .st-ap, .st-aq, .st-ak, .st-ar, .st-am, .st-as, .st-at, .st-au, .st-av, .st-aw, .st-ax, .st-ay, .st-az, .st-b0, .st-b1, .st-b2, .st-b3, .st-b4, .st-b5, .st-b6, .st-di, .st-dj, .st-dk, .st-dl, .st-bb{{
            background-color: {custom_color_background_select_button};
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


apply_custom_color_background_select_button(custom_color_background_select_button)


# --------------------------------------------------------------------------

custom_color_background = st.sidebar.color_picker("Choisir une couleur d'arrière-plan principal", "#EBF2F5")

def apply_custom_color_background(custom_color_background):
    custom_css = f"""
    <style>
        section {{
            background-color: {custom_color_background};
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


apply_custom_color_background(custom_color_background)


# --------------------------------------------------------------------------

custom_color_background = st.sidebar.color_picker("Choisir une couleur d'arrière-plan de la sidebar", "#ffffff")

def apply_custom_color_background(custom_color_background):
    custom_css = f"""
    <style>
        .st-emotion-cache-6qob1r, .eczjsme3 {{
            background-color: {custom_color_background};
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


apply_custom_color_background(custom_color_background)


# --------------------------------------------------------------------------

font_size = st.sidebar.slider("Choisir la taille de la police d'écriture des textes", 2, 32, 16)

def apply_custom_font_size(font_size):
    custom_css = f"""
    <style>
        label, div, p, .row-widget, .stSelectbox{{
            font-size: {font_size}px !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


apply_custom_font_size(font_size)


# --------------------------------------------------------------------------


font_size_title = st.sidebar.slider("Choisir la taille de la police d'écriture des titres", 10, 55, 40)

def apply_custom_font_size(font_size_title):
    custom_css = f"""
    <style>
        .st-emotion-cache-zt5igj, .e1nzilvr4, h2 {{
            font-size: {font_size_title}px !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


apply_custom_font_size(font_size_title)


# --------------------------------------------------------------------------

daltonian_font = st.sidebar.checkbox("Daltonien")

def apply_custom_daltonian(daltonian_font):
    custom_css = f"""
    <style>

    .st-emotion-cache-zt5igj, .e1nzilvr4 {{
        font-size: {font_size}px;
    }}

    .row-widget, .stSelectbox {{
        font-size: {font_size}px;
    }}

    .st-emotion-cache-6qob1r, .eczjsme3 {{
            background-color: {"#0866ff"};
    }}

    p, h2, td, th {{
            background-color: {"#0866ff"};
            color : white !important;
    }}

     section {{
            background-color: {"#0866ff"};
            color : white;
        }}
    .st-emotion-cache-zt5igj, .e1nzilvr4 {{
            color : white;
        }}
    .st-aw, .st-ak, .st-ax, .st-al, .st-ay, .st-az, .st-b0, .st-b1, .st-b2" {{
            background-color: {"#0866ff"};
        }}

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

if daltonian_font:
    apply_custom_daltonian(daltonian_font)


# --------------------------------------------------------------------------
    

dark_font = st.sidebar.checkbox("Dark mode")

def apply_custom_dark_mode(dark_font):
    custom_css = f"""
    <style>

    .st-emotion-cache-zt5igj, .e1nzilvr4 {{
        font-size: {font_size}px;
        color : white;
    }}

    .row-widget, .stSelectbox, st-emotion-cache-1ec096l e1q9reml4 {{
        font-size: {font_size}px;
        color : white;
    }}

    .st-emotion-cache-6qob1r, .eczjsme3 {{
            background-color: {"#000000"};
            color : white;
    }}

    p, td, th, h2, .st-emotion-cache-jfj0d9, .e115fcil0 {{
            background-color: {"#000000"};
            color : white !important;
    }}

     section {{
            background-color: {"#000000"};
            color : white;
        }}
    .st-emotion-cache-zt5igj, .e1nzilvr4 {{
            color : white;
        }}
    .st-aw, .st-ak, .st-ax, .st-al, .st-ay, .st-az, .st-b0, .st-b1, .st-b2" {{
            background-color: {"#000000"};
            color : white;
        }}

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

if dark_font:
    apply_custom_dark_mode(dark_font)


# --------------------------------------------------------------------------

st.title("Application de prédiction du diabète")
st.write("")
st.write("")
st.write("")





st.write(data.head())
st.caption("<p style='text-align: center;'>Aperçu des premières lignes du dataframe</p>", unsafe_allow_html=True)
st.write(data.describe())
st.caption("<p style='text-align: center;'>Aperçu des données du dataframe</p>", unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.caption("<h2 style='color: grey !important;'>Graphiques</h2>", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")


my_choice_7 = st.selectbox(
    "Choisissez une colonne pour pouvoir visualiser le nombre de diabétiques et non diabétique en fonction de cette colonne",
     [ "BMI", "Insulin", "BloodPressure", "Age", "Glucose", "Pregnancies", 'SkinThickness','DiabetesPedigreeFunction'])


fig = px.histogram(data, x=my_choice_7, color="Outcome")
fig.update_layout(title="Histogramme d'une colonne",
                  title_x=0.3,
                  title_font_size=font_size,
                  xaxis=dict(title_font=dict(size=font_size)),  # Taille de la police de l'axe x
                  yaxis=dict(title_font=dict(size=font_size)))
fig
st.caption("<p style='text-align: center;'>Graphique interactif contenant une colonne diffusé avec un histogramme", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")


my_choice_5 = st.selectbox(
    'Choisissez une première colonne pour laquelle vous désirez des informations',
     [ "BMI", "Insulin", "BloodPressure", "Age", "Glucose", "Pregnancies", 'SkinThickness','DiabetesPedigreeFunction'])

my_choice_6 = st.selectbox(
    'Choisissez une deuxième colonne pour laquelle vous désirez des informations',
     ['DiabetesPedigreeFunction', 'SkinThickness', "Pregnancies", "Glucose", "Age", "BloodPressure", "Insulin", "BMI"])


df = data.astype({"Outcome" : str})
fig = px.scatter(df, x=my_choice_5, y=my_choice_6, color="Outcome",
                 hover_data=[ "BMI", "Insulin", "BloodPressure", "Age", "Glucose", "Pregnancies", 'SkinThickness','DiabetesPedigreeFunction'])

fig.update_layout(title="scatter plot de 2 colonnes",
                  title_x=0.3,
                  title_font_size=font_size,
                  xaxis=dict(title_font=dict(size=font_size)),  # Taille de la police de l'axe x
                  yaxis=dict(title_font=dict(size=font_size)))
fig
st.caption("<p style='text-align: center;'>Graphique interactif contenant 2 colonnes diffusées avec un scatter plot", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")


my_choice_8 = st.selectbox(
    'Choisissez une colonne pour visualiser les modèles utilisés, leur courbe ROC, matrice de confusion, métriques, interprétabilité local et global',
     ['', 'Lightgbm', "RegressionLogistique", "TabPFN"], help="Choisir un modèle pour le visualiser grâce à des graphiques")

if my_choice_8 == '':
    st.write("")

if my_choice_8 == "Lightgbm":

    st.image("image_p10/lgbm-roc.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image auc-roc 0.856 Lightgbm", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/lgbm-matrice.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image confusion matrice Lightgbm", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/lgbm-lime.png", caption = "", width=1000)
    st.caption("<p style='text-align: center;'>image LIME Lightgbm", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/lgbm-shap.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image SHAPE Lightgbm", unsafe_allow_html=True)
    df = {"Métriques et temps d'entraînement": ['AUC ROC', 'Accuracy', "Temps d'entraînement"],
            'Valeur': [0.856, 0.8, 0.186]}
    df = pd.DataFrame(df)
    st.write("")
    st.write("")
    st.write("")
    st.table(df)
    st.caption("<p style='text-align: center;'>Tableau des métriques et du temps d'entraînement du modèle</p>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")


if my_choice_8 == "RegressionLogistique":

    st.image("image_p10/regression-roc.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image auc-roc 0.830 Regression Logistique", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/regression-matrice.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image confusion matrice Logistique", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/regression-lime.png", caption = "", width=1000)
    st.caption("<p style='text-align: center;'>image LIME Logistique", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/regression-shap.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image SHAPE Logistique", unsafe_allow_html=True)
    df = {"Métriques et temps d'entraînement": ['AUC ROC', 'Accuracy', "Temps d'entraînement"],
            'Valeur': [0.830, 0.750, 0.014]}
    df = pd.DataFrame(df)
    st.write("")
    st.write("")
    st.write("")
    st.table(df)
    st.caption("<p style='text-align: center;'>Tableau des métriques et du temps d'entraînement du modèle</p>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")


if my_choice_8 == "TabPFN":

    st.image("image_p10/tabpfn-roc.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image auc-roc 0.882 TabPFN", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/tabpfn-matrice.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image confusion matrice TabPFN", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/tabpfn-lime.png", caption = "", width=1000)
    st.caption("<p style='text-align: center;'>image LIME TabPFN", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.image("image_p10/tabpfn-shap.png", caption = "", width=700)
    st.caption("<p style='text-align: center;'>image SHAPE TabPFN", unsafe_allow_html=True)
    df = {"Métriques et temps d'entraînement": ['AUC ROC', 'Accuracy', "Temps d'entraînement"],
            'Valeur': [0.882, 0.815, 0.010]}
    df = pd.DataFrame(df)
    st.write("")
    st.write("")
    st.write("")
    st.table(df)
    st.caption("<p style='text-align: center;'>Tableau des métriques et du temps d'entraînement du modèle</p>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")



st.caption("<h2 style='color: grey !important;'>Prédiction du diabète</h2>", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
    
# filename = 'tabpfn.pkl'
# loaded_model = joblib.load(filename)

# filename = 'logistic_regression.sav'
# loaded_model = joblib.load(filename)



# pickle_file = open("lr.pkl", "rb")
# loaded_model = pickle.load(pickle_file)

# with open('tabpfn.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

# Input variables 
# X = data.drop(data.columns[-1], axis=1)
X = data.iloc[:,:-1]
# Output variable
# y = data[[data.columns[-1]]]
y = data.iloc[:,-1:] 

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

scaler = StandardScaler()
# fit the scaler into the train set, it will learn the parameters
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns)

loaded_model = TabPFNClassifier(device='cpu', N_ensemble_configurations=1)

loaded_model.fit(X_train_scaled, y_train)

# # --------------------------------------------------------------------------


def main():

    Pregnancies = st.slider("Quel est le nombre de fois où tu es tombé enceinte ?", 0, 17, help="Sélectionne le nombre de fois où tu es tombé enceinte")
    Glucose = st.slider("Quel est ton taux de glucose ?", 0, 199, help="Sélectionne ton taux de glucose")
    BloodPressure = st.slider("Quelle est ta pression sanguine ?", 0, 122, help="Sélectionne ta pression sanguine")
    SkinThickness = st.slider("Quelle est l'épaisseur de ta peau ?", 0, 99, help="Sélectionne l'épaisseur de ta peau")
    Insulin = st.slider("Quel est ton taux d'insuline ?", 0, 846, help="Sélectionne ton taux d'insuline")
    BMI = st.slider("Quel est ton indice de masse corporel ?", 0, 67, help="Sélectionne ton indice de masse corporel")
    DiabetesPedigreeFunction = st.slider("Quel est le nombre de personnes diabétiques dans ta famille ?", 0, 3, help="Sélectionne le nombre de personnes diabétique dans ta famille")
    Age = st.slider("Quel age as-tu ?", 21, 81, help="Sélectionne ton age")

    input_data = (Pregnancies,	Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age	)
    
    scaler = StandardScaler()
    scaler.fit(np.array(input_data).reshape(-1, 1))

    normalized_input = scaler.transform(np.array(input_data).reshape(-1, 1))

    input_data_as_numpy_array = np.asarray(normalized_input)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    result = loaded_model.predict(input_data_reshaped)

    if st.button("predict"):
        updated_res = result.flatten().astype(int)
        if updated_res == 0:
            st.markdown("Vous êtes peu susceptible d'être diabétique <span style='color: white; background-color: green'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>", unsafe_allow_html=True)
        if updated_res == 1:
            st.write("Vous êtes susceptible d'être diabétique <span style='color: white; background-color: red'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>", unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()