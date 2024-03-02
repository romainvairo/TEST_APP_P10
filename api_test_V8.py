# ------------------------- Imports Libraries ------------------------------

# import numpy as np
# import streamlit as st
# import pandas as pd
# import plotly.io as pio
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
# pio.templates.default = "plotly"
# import joblib
import pickle
# import sklearn
# from sklearn.preprocessing import StandardScaler
# from tabpfn import TabPFNClassifier
# import tabpfn
# # --------------------------------------------------------------------------


# data = pd.read_csv("diabetes.csv")


# st.sidebar.header("Paramètres")


# # # --------------------------------------------------------------------------


# custom_color_text = st.sidebar.color_picker("Choisir la couleur des textes", "#000000")

# def apply_custom_color_text(custom_color_text):
#     """
#     Appliquer la couleur personnalisée à tous les textes.
#     """
#     custom_css = f"""
#     <style>
#         body, p, .st-emotion-cache-ue6h4q, .e1y5xkzn3{{
#             color: {custom_color_text};
#         }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# apply_custom_color_text(custom_color_text)


# # --------------------------------------------------------------------------


# custom_color_background_select_button = st.sidebar.color_picker("Choisir une couleur pour les selectboxes", "#FFFFFF")

# def apply_custom_color_background_select_button(custom_color_background_select_button):
#     """
#     Appliquer la couleur personnalisée à tous les textes.
#     """
#     custom_css = f"""
#     <style>
#         .st-an, .st-ao, .st-ap, .st-aq, .st-ak, .st-ar, .st-am, .st-as, .st-at, .st-au, .st-av, .st-aw, .st-ax, .st-ay, .st-az, .st-b0, .st-b1, .st-b2, .st-b3, .st-b4, .st-b5, .st-b6, .st-di, .st-dj, .st-dk, .st-dl, .st-bb{{
#             background-color: {custom_color_background_select_button};
#         }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)


# apply_custom_color_background_select_button(custom_color_background_select_button)


# # --------------------------------------------------------------------------

# custom_color_background = st.sidebar.color_picker("Choisir une couleur d'arriere plan principal", "#EBF2F5")

# def apply_custom_color_background(custom_color_background):
#     """
#     Appliquer la couleur personnalisée à tous les textes.
#     """
#     custom_css = f"""
#     <style>
#         section {{
#             background-color: {custom_color_background};
#         }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)


# apply_custom_color_background(custom_color_background)


# # --------------------------------------------------------------------------

# custom_color_background = st.sidebar.color_picker("Choisir une couleur d'arriere plan du sidebar", "#D85656")

# def apply_custom_color_background(custom_color_background):
#     """
#     Appliquer la couleur personnalisée à tous les textes.
#     """
#     custom_css = f"""
#     <style>
#         .st-emotion-cache-6qob1r, .eczjsme3 {{
#             background-color: {custom_color_background};
#         }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)


# apply_custom_color_background(custom_color_background)


# # --------------------------------------------------------------------------

# font_size = st.sidebar.slider("Choisir la taille de la police d'ecriture des textes", 2, 30, 16)

# def apply_custom_font_size(font_size):
#     """
#     Appliquer la taille de police personnalisée.
#     """
#     custom_css = f"""
#     <style>
#         p, .row-widget, .stSelectbox {{
#             font-size: {font_size}px;
#         }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)


# apply_custom_font_size(font_size)


# # --------------------------------------------------------------------------


# font_size_title = st.sidebar.slider("Choisir la taille de la police d'ecriture des titres", 10, 55, 40)

# def apply_custom_font_size(font_size_title):
#     """
#     Appliquer la taille de police personnalisée.
#     """
#     custom_css = f"""
#     <style>
#         .st-emotion-cache-zt5igj, .e1nzilvr4 {{
#             font-size: {font_size_title}px;
#         }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)


# apply_custom_font_size(font_size_title)


# # --------------------------------------------------------------------------

# daltonian_font = st.sidebar.checkbox("Daltonien")

# def apply_custom_daltonian(daltonian_font):
#     """
#     Appliquer la taille de police personnalisée.
#     """
#     custom_css = f"""
#     <style>

#     .st-emotion-cache-zt5igj, .e1nzilvr4 {{
#         font-size: 80px;
#     }}

#     .row-widget, .stSelectbox {{
#         font-size: {50}px;
#     }}

#     .st-emotion-cache-6qob1r, .eczjsme3 {{
#             background-color: {"#0866ff"};
#     }}

#     p, h2 {{
#             background-color: {"#0866ff"};
#             color : white;
#     }}

#      section {{
#             background-color: {"#0866ff"};
#             color : white;
#         }}
#     .st-emotion-cache-zt5igj, .e1nzilvr4 {{
#             color : white;
#         }}
#     .st-aw, .st-ak, .st-ax, .st-al, .st-ay, .st-az, .st-b0, .st-b1, .st-b2" {{
#             background-color: {"#0866ff"};
#         }}

#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# if daltonian_font:
#     apply_custom_daltonian(daltonian_font)


# # --------------------------------------------------------------------------
    

# dark_font = st.sidebar.checkbox("Dark")

# def apply_custom_dark_mode(dark_font):
#     """
#     Appliquer la taille de police personnalisée.
#     """
#     custom_css = f"""
#     <style>

#     .st-emotion-cache-zt5igj, .e1nzilvr4 {{
#         font-size: 80px;
#     }}

#     .row-widget, .stSelectbox {{
#         font-size: {50}px;
#     }}

#     .st-emotion-cache-6qob1r, .eczjsme3 {{
#             background-color: {"#000000"};
#     }}

#     p, h2, .st-emotion-cache-jfj0d9, .e115fcil0 {{
#             background-color: {"#000000"};
#             color : white;
#     }}

#      section {{
#             background-color: {"#000000"};
#             color : white;
#         }}
#     .st-emotion-cache-zt5igj, .e1nzilvr4 {{
#             color : white;
#         }}
#     .st-aw, .st-ak, .st-ax, .st-al, .st-ay, .st-az, .st-b0, .st-b1, .st-b2" {{
#             background-color: {"#000000"};
#         }}

#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# if dark_font:
#     apply_custom_dark_mode(dark_font)


# # --------------------------------------------------------------------------
    

# st.title("Application de prédiction du diabète")
# st.write("Changez la couleur à l'aide du sélecteur de couleur et cliquez sur le bouton pour voir les modifications en direct.")

# st.write(data.head())
# st.write(data.describe())

# st.write("Graphique :")
# fig, ax = plt.subplots()
# choice = st.selectbox(
#     'Vous pouvez choisir une colonne spécifique s:',
#     ['Pregnancies', 'BloodPressure', 'SkinThickness', 'BMI', 'Glucose', 'Insulin', 'DiabetesPedigreeFunction'])
# sns.histplot(data=data, x=choice, hue='Outcome', kde=True)
# ax.set_title('Répartition des personnes diabétiques selon leurs {}'.format(choice))
# st.pyplot(fig)

# my_choice_7 = st.selectbox(
#     'Vous pouvez choisir une colonne spécifique :',
#      [ "BMI", "Insulin", "BloodPressure", "Age", "Glucose", "Glucose", "Pregnancies", 'SkinThickness','DiabetesPedigreeFunction'])


# fig = px.histogram(data, x=my_choice_7, color="Outcome")
# fig

# my_choice_5 = st.selectbox(
#     'Vous pouvez choisir une col nne spécifique :',
#      [ "BMI", "Insulin", "BloodPressure", "Age", "Glucose", "Glucose", "Pregnancies", 'SkinThickness','DiabetesPedigreeFunction'])

# my_choice_6 = st.selectbox(
#     'Vous possuvez choisir une colonne spécifique :',
#      ['DiabetesPedigreeFunction', 'SkinThickness', "Pregnancies", "Glucose", "Age", "BloodPressure", "Insulin", "BMI"])


# df = data.astype({"Outcome" : str})
# fig = px.scatter(df, x=my_choice_5, y=my_choice_6, color="Outcome",
#                  hover_data=[ "BMI", "Insulin", "BloodPressure", "Age", "Glucose", "Glucose", "Pregnancies", 'SkinThickness','DiabetesPedigreeFunction'])
# fig

# my_choice_8 = st.selectbox(
#     'Vous possuvez cr une colonne spécifique :',
#      ['', 'Lightgbm', "RegressionLogistique", "TabPFN"])

# if my_choice_8 == '':
#     st.write("")

# if my_choice_8 == "Lightgbm":

#     st.image("image_p10/lightgbm_auc_roc.png", caption = "image auc-roc 0.807 Lightgbm", width=500)
#     st.image("image_p10/lightgbm_confusion_matrice.png", caption = "image confusion matrice Lightgbm", width=500)
#     st.image("image_p10/lightgbm_lime.png", caption = "image LIME Lightgbm", width=1150)
#     st.image("image_p10/lightgbm_shap.png", caption = "image SHAPE Lightgbm", width=700)
#     data = {'Métrique': ['AUC ROC', 'Accuracy'],
#             'Valeur': [0.807, 0.74]}
#     df = pd.DataFrame(data)
#     st.table(df)

# if my_choice_8 == "RegressionLogistique":

#     st.image("image_p10/regression_logistique_auc_roc.png", caption = "image auc-roc 0.830 Regression Logistique", width=500)
#     st.image("image_p10/regression_logistique_confusion_matrice.png", caption = "image confusion matrice Logistique", width=500)
#     st.image("image_p10/regression_logistique_lime.png", caption = "image LIME Logistique", width=1150)
#     st.image("image_p10/regression_logistique_shap.png", caption = "image SHAPE Logistique", width=700)
#     data = {'Métrique': ['AUC ROC', 'Accuracy'],
#             'Valeur': [0.830, 0.77]}
#     df = pd.DataFrame(data)
#     st.table(df)


# if my_choice_8 == "TabPFN":

#     st.image("image_p10/tabpfn_auc_roc.png", caption = "image auc-roc 0.843 TabPFN", width=500)
#     st.image("image_p10/tabpfn_confusion_matrice.png", caption = "image confusion matrice TabPFN", width=500)
#     st.image("image_p10/tabpfn_lime.png", caption = "image LIME TabPFN", width=1150)
#     st.image("image_p10/tabpfn_shap.png", caption = "image SHAPE TabPFN", width=700)
#     data = {'Métrique': ['AUC ROC', 'Accuracy'],
#             'Valeur': [0.843, 0.79]}
#     df = pd.DataFrame(data)
#     st.table(df)
    
# filename = 'tabpfn.pkl'
# loaded_model = joblib.load(filename)

# filename = 'logistic_regression.sav'
# loaded_model = joblib.load(filename)



# pickle_file = open("lr.pkl", "rb")
# loaded_model = pickle.load(pickle_file)

with open('tabpfn.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# # --------------------------------------------------------------------------


# def main():

#     Pregnancies = st.slider("met le nombre de fois ou tu es tombé enceinte", 0, 17, help="Sélectionnez le nombre de fois ou tu es tombé enceinte")
#     Glucose = st.slider("De combien est ton taux de glucose", 0, 199)
#     BloodPressure = st.slider("De combien est ta pression sanguine ?", 0, 130)
#     SkinThickness = st.slider("Quelle est l'épaisseur de ta peau ?", 0, 100)
#     Insulin = st.slider("De combien est ton taux d'insuline ?", 0, 200)
#     BMI = st.slider("Quel est ton indice de masse corporel ?", 0, 60)
#     DiabetesPedigreeFunction = st.slider("Combien de personnes sont diabetiques dans ta famille ?", 0, 3)
#     Age = st.slider("Quel age as tu ?", 21, 100)

#     input_data = (Pregnancies,	Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age	)
#     scaler = StandardScaler()

#     scaler.fit(np.array(input_data).reshape(-1, 1))

#     normalized_input = scaler.transform(np.array(input_data).reshape(-1, 1))

#     input_data_as_numpy_array = np.asarray(normalized_input)
#     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#     result = loaded_model.predict(input_data_reshaped)

#     if st.button("predict"):
#         updated_res = result.flatten().astype(int)
#         if updated_res == 0:
#             st.write("Vous êtes peu susceptible d'être diabétique")
#         if updated_res == 1:
#             st.write("Vous êtes susceptible d'être diabétique")
    
# if __name__ == '__main__':
#     main()