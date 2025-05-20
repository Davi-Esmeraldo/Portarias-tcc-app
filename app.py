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


# ========== Carregamento dos dados ==========

with open("/content/drive/My Drive/UnB/EU/TCC/Davi/Dados/todas_portarias_maio.json", encoding='utf-8') as f:
    todas_portarias_maio = json.load(f)

with open("/content/drive/My Drive/UnB/EU/TCC/Davi/Dados/resultados_entidades.json", encoding='utf-8') as f:
    resultados_entidades = json.load(f)

with open("/content/drive/My Drive/UnB/EU/TCC/Davi/Dados/labels_ren_resumo.json", encoding='utf-8') as f:
    labels_ren_resumo = json.load(f)

with open("/content/drive/My Drive/UnB/EU/TCC/Davi/Dados/vetores_fasttext.json", encoding='utf-8') as f:
    vetores_fasttext = json.load(f)

with open("/content/drive/My Drive/UnB/EU/TCC/Davi/Dados/portarias_processadas.json", encoding='utf-8') as f:
    portarias_processadas = json.load(f)

# ========== Funções auxiliares / Manipulações ==========

def encontrar_similares(numero_desejado, vetores_fasttext, top_n=10):
    if numero_desejado not in vetores_fasttext:
        return pd.DataFrame()

    vetor_base = np.array(vetores_fasttext[numero_desejado]).reshape(1, -1)
    todos_ids = list(vetores_fasttext.keys())
    todos_vetores = np.array([vetores_fasttext[k] for k in todos_ids])
    
    similaridades = cosine_similarity(vetor_base, todos_vetores).flatten()
    df = pd.DataFrame({'numero': todos_ids, 'similaridade': similaridades})
    df = df[df['numero'] != numero_desejado].sort_values(by='similaridade', ascending=False)
    return df.head(top_n)[["numero", "similaridade"]]
    #return df.head(top_n)

def gerar_grafico_clusters(vetores_fasttext, numero_desejado, k=3):
    numeros = list(vetores_fasttext.keys())
    X = np.array([vetores_fasttext[n] for n in numeros])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    clusters = kmeans.labels_

    df_plot = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters,
        'Número': numeros
    })
    
    fig, ax = plt.subplots()
    for cluster_id in sorted(df_plot["Cluster"].unique()):
        cluster_df = df_plot[df_plot["Cluster"] == cluster_id]
        ax.scatter(cluster_df["PCA1"], cluster_df["PCA2"], label=f'Cluster {cluster_id}')
        
        # Exibir apenas o número da portaria selecionada
        for i, row in cluster_df.iterrows():
            if row["Número"] == numero_desejado:
                ax.text(row["PCA1"], row["PCA2"], row["Número"], fontsize=9, color='black', weight='bold')

    ax.set_title("Clusterização das Portarias (KMeans + PCA)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()
    st.pyplot(fig)

# Inicializa pipeline spaCy
nlp = spacy.blank("pt")

# Cores para as entidades
colors = {
    "ACAO": "#FF9999",      # rosa claro
    "SUJEITO": "#66CCFF",   # azul claro
    "LOCAL": "#99CC66",     # verde claro
    "DATA": "#CE93D8"       # lilás
}
options = {"ents": list(colors.keys()), "colors": colors}


def renderizar_entidades_streamlit(portaria_id, todas_portarias_maio, resultados_entidades, labels_ren_resumo):
    if portaria_id not in todas_portarias_maio:
        st.error(f"Portaria {portaria_id} não encontrada.")
        return

    texto = todas_portarias_maio[portaria_id]["resumo"]
    st.subheader("Descrição da portaria:")
    st.write(texto)

    nlp = spacy.blank("pt")
    doc = nlp(texto)
    spans = []

    if portaria_id in labels_ren_resumo:
        for entidade in labels_ren_resumo[portaria_id]["labels"]:
            trecho = entidade["text"]
            label = entidade["label"]
            start = texto.lower().find(trecho.lower())
            if start != -1:
                end = start + len(trecho)
                span = doc.char_span(start, end, label=label)
                if span:
                    spans.append(span)

    elif portaria_id in resultados_entidades:
        entidades_previstas = resultados_entidades[portaria_id]
        spans_temp = []
        for token_text, label in entidades_previstas:
            for token in doc:
                if token.text == token_text:
                    span = Span(doc, token.i, token.i + 1, label=label)
                    spans_temp.append(span)
                    break

        # Agrupar spans consecutivos
        if spans_temp:
            agrupados = []
            atual = spans_temp[0]
            for prox in spans_temp[1:]:
                if prox.start == atual.end and prox.label_ == atual.label_:
                    atual = Span(doc, atual.start, prox.end, label=atual.label_)
                else:
                    agrupados.append(atual)
                    atual = prox
            agrupados.append(atual)
            spans = agrupados

    spans = list({(s.start, s.end, s.label_): s for s in spans}.values())
    spans = filter_spans(spans)
    doc.ents = spans

    # Exibir entidades identificadas
    st.subheader("Entidades reconhecidas:")
    for s in spans:
        st.write(f"{s.text} ({s.label_})")

    # Cores
    colors = {
        "ACAO": "#FF9999",
        "SUJEITO": "#66CCFF",
        "LOCAL": "#99CC66",
        "DATA": "#CE93D8"
    }
    options = {"ents": list(colors.keys()), "colors": colors}

    # Visualizar com spaCy
    html = displacy.render(doc, style="ent", options=options, page=True)
    components.html(html, height=300, scrolling=True)


# ========== Interface Streamlit ==========

st.title("Visualização e Análise de Portarias")

numero_portaria = st.selectbox(
    "Selecione o número da portaria:",
    sorted(todas_portarias_maio.keys(), key=lambda x: int(x), reverse=True)
)


# Descrição da portaria e Entidades reconhecidas
renderizar_entidades_streamlit(numero_portaria, todas_portarias_maio, resultados_entidades, labels_ren_resumo)

# Mostrar texto completo
st.markdown("### Conteúdo da portaria selecionada:")
texto_completo = todas_portarias_maio[numero_portaria]['conteudo']
st.text(texto_completo)  

# Mostrar similares
st.markdown("### Portarias mais similares:")
df_similares = encontrar_similares(numero_portaria, vetores_fasttext)
st.dataframe(df_similares.set_index("numero"), use_container_width=True)

# Mostrar gráfico de clusterização
st.markdown("### Visualização de Clusters:")
gerar_grafico_clusters(vetores_fasttext, numero_portaria)
