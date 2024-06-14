import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import base64
import nest_asyncio
import plotly.express as px
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import base64
import nest_asyncio
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import io


"""# Analyse des tables effects et medications"""

# Les données depuis les fichiers CSV
df_medications = pd.read_csv('medications.csv')
df_effects = pd.read_csv('effects.csv')

# Préparation des données pour les graphiques
def prepare_data():
    df_medications['Nom_du_médicament'] = df_medications['Nom_medicament'].str.split().str[0]
    df_sex_distribution = df_medications.groupby('Nom_du_médicament')[['Hommes (%)', 'Femmes (%)']].mean().reset_index()
    df_age_distribution = df_medications.groupby('Nom_du_médicament')[['20-59 (%)', '60+ (%)']].mean().reset_index()
    df_effects['Nom_du_médicament'] = df_effects['Nom_medicament'].str.split().str[0]
    df_effects_count = df_effects.groupby(['Nom_du_médicament', 'Pathologie'])['Valeur'].sum().reset_index()

    return df_sex_distribution, df_age_distribution, df_effects_count

df_sex_distribution, df_age_distribution, df_effects_count = prepare_data()

"""# NLP"""

# Les ressources nécessaires pour nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

# Chargement des données depuis les fichiers CSV
df_posts = pd.read_csv('forum_posts_30_pages.csv')
df_medications = pd.read_csv('medications.csv')
df_effects = pd.read_csv('effects.csv')

# Les valeurs de la colonne 'Content' en chaînes de caractères
df_posts['Content'] = df_posts['Content'].astype(str)

# Les fonctions de nettoyage
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_newlines(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_digits(text):
    text = re.sub(r'\d', '', text)
    return text

def clean_text(text):
    text = text.lower()  # Convertir en minuscules
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join([token for token in tokens if token not in stop_words])
    return text

stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Appliquation des fonctions de nettoyage aux posts du forum
df_posts['Content'] = df_posts['Content'].apply(lambda x: remove_URL(x))
df_posts['Content'] = df_posts['Content'].apply(lambda x: remove_emoji(x))
df_posts['Content'] = df_posts['Content'].apply(lambda x: remove_html(x))
df_posts['Content'] = df_posts['Content'].apply(lambda x: remove_punct(x))
df_posts['Content'] = df_posts['Content'].apply(lambda x: remove_newlines(x))
df_posts['Content'] = df_posts['Content'].apply(lambda x: remove_digits(x))
df_posts['Content'] = df_posts['Content'].apply(lambda x: clean_text(x))
df_posts['Content'] = df_posts['Content'].apply(lambda x: lemmatize_stemming(x))


big_text = ' '.join(df_posts['Content'].tolist())

# Liste des effets indésirables
effects_list = [
    "nausea", "vomiting", "headache", "dizziness", "fatigue",
    "abdominal pain", "diarrhea", "constipation", "loss of appetite",
    "muscle pain", "muscle spasms", "skin rash", "itching", "redness",
    "excessive sweating", "fever", "palpitations", "hypertension",
    "hypotension", "tachycardia", "bradycardia", "anemia", "weight loss",
    "weight gain", "swelling", "joint pain", "back pain", "chills",
    "dry mouth", "metallic taste", "increased thirst", "increased urination",
    "difficulty swallowing", "chest tightness", "tingling", "numbness",
    "lightheadedness", "malaise", "fatigue", "weakness", "tremors",
    "insomnia", "drowsiness", "confusion", "anxiety", "depression",
    "agitation", "tremors", "seizures", "memory loss", "balance disorders",
    "blurred vision", "restlessness", "panic attacks", "irritability",
    "hallucinations", "nightmares", "paranoia", "mood swings", "delirium",
    "dysarthria", "dystonia", "neuralgia", "paresthesia", "syncope",
    "stomach pain", "bloating", "acid reflux", "gastric ulcers",
    "indigestion", "flatulence", "belching", "heartburn", "intestinal gas",
    "bloody stools", "black stools", "jaundice", "liver dysfunction",
    "pancreatitis", "hepatitis", "colitis", "enteritis",
    "difficulty breathing", "shortness of breath", "cough", "nasal congestion",
    "wheezing", "sinusitis", "respiratory infection", "bronchospasm",
    "pneumonia", "dry throat", "hoarseness", "pharyngitis", "laryngitis",
    "asthma", "emphysema", "pleurisy",
    "hives", "dry skin", "hair loss", "eczema", "acne", "blisters",
    "photosensitivity", "skin discoloration", "bruising", "peeling skin",
    "dermatitis", "psoriasis", "alopecia", "urticaria",
    "chest pain", "heart attack", "stroke", "high blood pressure",
    "low blood pressure", "arrhythmia", "peripheral edema", "venous thrombosis",
    "heart palpitations", "heart failure", "angina", "myocarditis",
    "pericarditis", "varicose veins", "hypertension",
    "kidney pain", "urinary retention", "urinary incontinence", "kidney stones",
    "renal failure", "hematuria", "proteinuria", "dysuria",
    "impotence", "decreased libido", "menstrual irregularities",
    "breast tenderness", "gynecomastia", "hyperglycemia", "hypoglycemia",
    "electrolyte imbalance", "thyroid dysfunction", "increased cholesterol",
    "dehydration", "diabetes", "hyperthyroidism", "hypothyroidism",
    "taste changes", "visual disturbances", "hearing loss", "tinnitus",
    "anaphylaxis", "allergic reactions", "fat redistribution", "hepatotoxicity",
    "cardiotoxicity", "nephrotoxicity", "neurotoxicity", "ototoxicity",
    "musculoskeletal pain", "bone pain", "osteoporosis"
]

# Tokenisation
tokens = word_tokenize(big_text)

effect_terms_count = Counter([word for word in tokens if word in effects_list])

# Le nuage de mots avec tous les termes trouvés
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(effect_terms_count)
wordcloud.to_file("wordcloud.png")

# Analyseur de sentiments
sia = SentimentIntensityAnalyzer()

df_posts['sentiment'] = df_posts['Content'].apply(lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0)

# Classification des sentiments
def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df_posts['sentiment_class'] = df_posts['sentiment'].apply(classify_sentiment)

"""# Dash"""

# les images
wordcloud_path = "wordcloud.png"
logo_path = "logo.png"

# Convert images to base64
encoded_wordcloud = base64.b64encode(open(wordcloud_path, 'rb').read()).decode('ascii')
encoded_logo = base64.b64encode(open(logo_path, 'rb').read()).decode('ascii')

# Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Mise en page de l'application
app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            html.Img(src=f'data:image/jpeg;base64,{encoded_logo}', height="40px"),
            dbc.NavbarBrand(
                "Déclarations des effets indésirables associés aux médicaments de alzeimer",
                className="ms-2"
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Accueil", href="/")),
                        dbc.NavItem(dbc.NavLink("ANSM", href="/ansm")),
                        dbc.NavItem(dbc.NavLink("Forum Alzheimer", href="/forum")),
                    ],
                    className="ms-auto",
                    navbar=True
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ]),
        color="primary",
        dark=True,
    ),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], fluid=True)

# Page d'accueil

home_page = dbc.Container([
    html.Div([
        html.H3("Projet"),
        html.P("Le projet consiste en plusieurs étapes clés : collecte des données, analyse des sentiments, et visualisation des résultats. "
               "La collecte des données est effectuée par web scraping des forums spécialisés, suivi par une analyse de sentiments pour comprendre les perceptions des utilisateurs. "
               "Les résultats sont ensuite présentés sous forme de graphiques interactifs pour une interprétation facile."),
        html.Hr(),  # Add a horizontal rule for separation
        html.H3("Word Cloud des Effets Indésirables"),
        dbc.Card(
            dbc.CardBody(
                html.Div(
                    html.Img(src=f'data:image/png;base64,{encoded_wordcloud}', style={'width': '100%'}),
                    style={'display': 'flex', 'justify-content': 'center'}
                )
            ),
            className="mt-3"
        ),
        html.Hr(),  # Add a horizontal rule for separation

    ])
], fluid=True)

# Page ANSM
ansm_page = dbc.Container([
    html.H3("Sélectionner un Médicament"),
    dcc.Dropdown(
        id='medicament-dropdown',
        options=[{'label': nom, 'value': nom} for nom in df_sex_distribution['Nom_du_médicament'].unique()],
        value=df_sex_distribution['Nom_du_médicament'].unique()[0]  # Valeur par défaut
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id='sex-distribution'), width=6),
        dbc.Col(dcc.Graph(id='age-distribution'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='effects-distribution'), width=12)
    ])
], fluid=True)

# Page Forum
forum_page = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Graph(id='sentiment-bar-chart'), width=6),
        dbc.Col(dcc.Graph(id='sentiment-box-plot'), width=6)
    ])
], fluid=True)

# Mise à jour du contenu en fonction de l'URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/ansm':
        return ansm_page
    elif pathname == '/forum':
        return forum_page
    else:
        return home_page

# Callbacks pour mettre à jour les graphiques de la page ANSM
@app.callback(
    Output('sex-distribution', 'figure'),
    Output('age-distribution', 'figure'),
    Output('effects-distribution', 'figure'),
    Input('medicament-dropdown', 'value')
)
def update_ansm_graphs(selected_medicament):
    filtered_sex_df = df_sex_distribution[df_sex_distribution['Nom_du_médicament'] == selected_medicament]
    filtered_age_df = df_age_distribution[df_age_distribution['Nom_du_médicament'] == selected_medicament]
    filtered_effects_df = df_effects_count[df_effects_count['Nom_du_médicament'] == selected_medicament]

    fig_sex = px.bar(filtered_sex_df, x='Nom_du_médicament', y=['Hommes (%)', 'Femmes (%)'],
                     title=f'Répartition des Patients par Sexe pour {selected_medicament}', barmode='group',
                     color_discrete_sequence=px.colors.qualitative.Pastel)

    age_data = filtered_age_df.melt(id_vars=["Nom_du_médicament"], value_vars=['20-59 (%)', '60+ (%)'],
                                    var_name="Tranche d'âge", value_name="Pourcentage")
    fig_age = px.pie(age_data, names='Tranche d\'âge', values='Pourcentage',
                     title=f'Répartition des Patients par Tranche d\'Âge pour {selected_medicament}',
                     color_discrete_sequence=px.colors.qualitative.Pastel)

    df_effects_sum = filtered_effects_df.groupby('Pathologie')['Valeur'].sum().reset_index()
    fig_effects = px.pie(df_effects_sum, values='Valeur', names='Pathologie',
                         title=f'Types d\'Effets Indésirables pour {selected_medicament}',
                         color_discrete_sequence=px.colors.qualitative.Pastel)

    return fig_sex, fig_age, fig_effects

# Analyseur de sentiments
sia = SentimentIntensityAnalyzer()
df_posts['sentiment'] = df_posts['Content'].apply(lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0)

# Classification des sentiments
def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df_posts['sentiment_class'] = df_posts['sentiment'].apply(classify_sentiment)

# Visualisation des sentiments
sentiment_counts = df_posts['sentiment_class'].value_counts().reset_index()
sentiment_counts.columns = ['sentiment_class', 'count']

# Callbacks pour mettre à jour les graphiques de la page Forum
@app.callback(
    Output('sentiment-bar-chart', 'figure'),
    Output('sentiment-box-plot', 'figure'),
    Input('url', 'pathname')
)
def update_forum_graphs(pathname):
    fig_bar = px.bar(sentiment_counts, x='sentiment_class', y='count',
                     title='Distribution des Sentiments dans les Posts du Forum',
                     color='sentiment_class', color_discrete_sequence=px.colors.qualitative.Pastel)

    fig_box = px.box(df_posts, x='sentiment_class', y='sentiment',
                     title='Scores de Sentiment par Classe de Sentiment',
                     color='sentiment_class', color_discrete_sequence=px.colors.qualitative.Pastel)

    return fig_bar, fig_box

# Exécutation l'application
nest_asyncio.apply()
app.run_server(mode='inline', height=1000)