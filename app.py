
import streamlit as st
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy import displacy
import streamlit.components.v1 as components
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# ========== Carregamento dos dados ==========

with open("todas_portarias_maio.json", encoding='utf-8') as f:
    todas_portarias_maio = json.load(f)

with open("vetores_fasttext.json", encoding='utf-8') as f:
    vetores_fasttext = json.load(f)

with open("portarias_processadas.json", encoding='utf-8') as f:
    portarias_processadas = json.load(f)

with open("dict_combined.json", encoding='utf-8') as f:
    dict_combined = json.load(f)

with open("resultados_entidades_final.json", encoding='utf-8') as f:
    resultados_entidades_final = json.load(f)

# ========== Cores para as Entidades ==========
colors = {
    "ACAO": "#FF9999",      # rosa claro
    "SUJEITO": "#66CCFF",   # azul claro
    "LOCAL": "#99CC66",     # verde claro
    "DATA": "#CE93D8"       # lilás
}

# ========== Funções Auxiliares ==========

def visualizar_anotacoes_manuaais(numero_portaria):
    texto = todas_portarias_maio[numero_portaria]['resumo']
    ents = []
    for entidade in dict_combined[numero_portaria]:
        span = {
            "start": entidade["start"],
            "end": entidade["end"],
            "label": entidade["label"]
        }
        ents.append(span)

    doc = {"text": texto, "ents": ents, "title": f"Anotações Manuais - Portaria {numero_portaria}"}

    html = displacy.render(doc, style="ent", manual=True, options={"colors": colors}, page=True)
    components.html(html, height=300, scrolling=True)

def visualizar_entidades_preditas(numero_portaria):
    texto = todas_portarias_maio[numero_portaria]['resumo']
    tokens = texto.split()
    ents = []
    idx = 0

    for token, label in resultados_entidades_final[numero_portaria]:
        start = texto.find(token, idx)
        if start == -1:
            continue
        end = start + len(token)
        idx = end
        ents.append({
            "start": start,
            "end": end,
            "label": label.split('-')[-1]  # Remove B- ou I-
        })

    doc = {"text": texto, "ents": ents, "title": f"Entidades Preditas - Portaria {numero_portaria}"}

    html = displacy.render(doc, style="ent", manual=True, options={"colors": colors}, page=True)
    components.html(html, height=300, scrolling=True)

def encontrar_similares(numero_desejado, vetores_fasttext, todas_portarias_maio, top_n=10):
    if numero_desejado not in vetores_fasttext:
        return pd.DataFrame()

    vetor_base = np.array(vetores_fasttext[numero_desejado]).reshape(1, -1)
    todos_ids = list(vetores_fasttext.keys())
    todos_vetores = np.array([vetores_fasttext[k] for k in todos_ids])

    similaridades = cosine_similarity(vetor_base, todos_vetores).flatten()
    df = pd.DataFrame({'numero': todos_ids, 'similaridade': similaridades})
    df = df[df['numero'] != numero_desejado].sort_values(by='similaridade', ascending=False).head(top_n)

    textos = []
    for num in df['numero']:
        if num in todas_portarias_maio:
            textos.append(todas_portarias_maio[num]['conteudo'])
        else:
            textos.append("Texto não encontrado.")
    df['texto_portaria'] = textos

    return df[['numero', 'similaridade', 'texto_portaria']]

def gerar_grafico_clusters_plotly(vetores_fasttext, numero_desejado, k=3):
    numeros = list(vetores_fasttext.keys())
    X = np.array([vetores_fasttext[n] for n in numeros])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    clusters = kmeans.labels_

    df_plot = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters.astype(str),
        'Número': numeros
    })

    df_plot['Selecionado'] = df_plot['Número'] == numero_desejado
    df_plot['Tamanho'] = df_plot['Selecionado'].apply(lambda x: 16 if x else 8)

    color_discrete_sequence = px.colors.qualitative.Bold

    fig = px.scatter(
        df_plot,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        size='Tamanho',
        custom_data=['Número', 'Cluster'],
        title="Clusterização das Portarias (KMeans + PCA)",
        color_discrete_sequence=color_discrete_sequence
    )

    fig.update_traces(
        marker=dict(symbol='circle', line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate="Portaria: %{customdata[0]}<br>Cluster: %{customdata[1]}<extra></extra>"
    )

    df_selected = df_plot[df_plot['Selecionado']]
    if not df_selected.empty:
        selected = df_selected.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[selected['PCA1']],
                y=[selected['PCA2']],
                mode='markers',
                marker=dict(size=18, color='red', symbol='circle', line=dict(width=2, color='DarkSlateGrey')),
                name='Selecionado',
                hovertext=f"Portaria: {selected['Número']}<br>Cluster: {selected['Cluster']}",
                hoverinfo='text',
                showlegend=False
            )
        )

    fig.update_layout(
        legend_title_text='Cluster',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            font=dict(color='black')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def gerar_nuvem_por_cluster(vetores_fasttext, todas_portarias_maio, k=3):
    numeros = list(vetores_fasttext.keys())
    X = np.array([vetores_fasttext[n] for n in numeros])

    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    clusters = kmeans.labels_

    df_clusters = pd.DataFrame({
        'Número': numeros,
        'Cluster': clusters
    })

    for cluster_id in sorted(df_clusters['Cluster'].unique()):
        st.markdown(f"#### Nuvem de Palavras - Cluster {cluster_id}")
        numeros_cluster = df_clusters[df_clusters['Cluster'] == cluster_id]['Número']

        textos = " ".join([todas_portarias_maio[num]['resumo'] for num in numeros_cluster if num in todas_portarias_maio])

        if textos.strip() != "":
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(textos)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("Sem texto suficiente para gerar nuvem.")

# ========== Interface Streamlit ==========

st.title("Visualização e Análise de Portarias")

numero_portaria = st.selectbox(
    "Selecione o número da portaria:",
    sorted(todas_portarias_maio.keys(), key=lambda x: int(x), reverse=True)
)

st.markdown("### Conteúdo da portaria selecionada:")
texto_completo = todas_portarias_maio[numero_portaria]['conteudo']
st.text(texto_completo)

st.markdown("### Descrição da portaria selecionada:")
texto_resumo = todas_portarias_maio[numero_portaria]["resumo"]
st.text(texto_resumo)

# Visualização de entidades
st.markdown("### Visualização de Entidades:")

if numero_portaria in dict_combined:
    visualizar_anotacoes_manuaais(numero_portaria)

if numero_portaria in resultados_entidades_final:
    visualizar_entidades_preditas(numero_portaria)

# Mostrar similares
st.markdown("### Portarias mais similares:")
df_similares = encontrar_similares(numero_portaria, vetores_fasttext, todas_portarias_maio)

df_similares = df_similares.rename(columns={'numero': 'Portaria', 'similaridade': 'Similaridade', 'texto_portaria': 'Conteúdo'})

for idx, row in df_similares.iterrows():
    st.write(f"**Portaria:** {row['Portaria']} | **Similaridade:** {row['Similaridade']:.4f}")
    with st.expander("Ver conteúdo"):
        st.write(row['Conteúdo'])

st.markdown("### Visualização de Clusters:")
gerar_grafico_clusters_plotly(vetores_fasttext, numero_portaria)

st.markdown("### Nuvens de Palavras por Cluster:")
gerar_nuvem_por_cluster(vetores_fasttext, todas_portarias_maio)
