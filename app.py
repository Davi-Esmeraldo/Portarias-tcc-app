from sklearn.preprocessing import normalize   
import streamlit as st
import json 
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
import streamlit.components.v1 as components
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter
from itertools import islice


# ========== Carregamento dos dados ==========

with open("todas_portarias_maio.json", encoding='utf-8') as f:
    todas_portarias_maio = json.load(f)

with open("vetores_Word2vec.json", encoding='utf-8') as f: 
    vetores_fasttext = json.load(f)

with open("portarias_processadas.json", encoding='utf-8') as f:
    portarias_processadas = json.load(f)

with open("dict_combined.json", encoding='utf-8') as f:
    dict_combined = json.load(f)

with open("resultados_entidades_final_final.json", encoding='utf-8') as f:
    resultados_entidades_final_final = json.load(f)

with open("portarias_ultra_processadas.json", encoding='utf-8') as f:
    portarias_ultra_processadas = json.load(f)



# ========== Cores para as Entidades ==========

colors = {
    "ACAO": "#FF9999",
    "SUJEITO": "#66CCFF",
    "LOCAL": "#99CC66",
    "DATA": "#CE93D8"
}

# ========== Funções Auxiliares ==========

def visualizar_anotacoes_manuaais(numero_portaria):
    texto = todas_portarias_maio[numero_portaria]['resumo']
    ents = []
    idx = 0
    for entidade in dict_combined[numero_portaria]['labels']:
        entidade_texto = entidade["text"]
        start = texto.find(entidade_texto, idx)
        if start == -1:
            continue
        end = start + len(entidade_texto)
        idx = end
        ents.append({
            "start": start,
            "end": end,
            "label": entidade["label"]
        })
    doc = {"text": texto, "ents": ents}
    html = displacy.render(doc, style="ent", manual=True, options={"colors": colors}, page=True)

    # Injetando CSS para texto branco
    style = """
    <style>
    body { color: white !important; }
    .entity { color: white !important; }
    .entity span { color: white !important; }
    </style>
    """
    html = style + html

    components.html(html, height=350, scrolling=False)



def visualizar_entidades_preditas(numero_portaria):
    texto = todas_portarias_maio[numero_portaria]['resumo']
    tokens_labels = resultados_entidades_final_final[numero_portaria]

    tokens = texto.split()
    ents = []
    idx = 0

    # Variáveis para agrupar
    grupo_tokens = []
    grupo_label = None

    for token, label_full in tokens_labels:
        label = label_full.split('-')[-1]  # Remove B- ou I-

        if grupo_label is None:
            grupo_tokens = [token]
            grupo_label = label
        elif label == grupo_label:
            grupo_tokens.append(token)
        else:
            # Salva o grupo anterior
            entidade_texto = ' '.join(grupo_tokens)
            start = texto.find(entidade_texto, idx)
            if start != -1:
                end = start + len(entidade_texto)
                ents.append({
                    "start": start,
                    "end": end,
                    "label": grupo_label
                })
                idx = end

            # Inicia novo grupo
            grupo_tokens = [token]
            grupo_label = label

    # Adiciona o último grupo
    if grupo_tokens:
        entidade_texto = ' '.join(grupo_tokens)
        start = texto.find(entidade_texto, idx)
        if start != -1:
            end = start + len(entidade_texto)
            ents.append({
                "start": start,
                "end": end,
                "label": grupo_label
            })

    doc = {"text": texto, "ents": ents}
    html = displacy.render(doc, style="ent", manual=True, options={"colors": colors}, page=True)

    # Injetando CSS para texto branco
    style = """
    <style>
    body { color: white !important; }
    .entity { color: white !important; }
    .entity span { color: white !important; }
    </style>
    """
    html = style + html

    components.html(html, height=350, scrolling=False)

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
    # Preparação dos dados
    numeros = list(vetores_fasttext.keys())
    X = np.array([vetores_fasttext[n] for n in numeros])
    # Normalização L2  
    X = normalize(X, norm='l2')
    # PCA para reduzir para 2 dimensões
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Clusterização
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    clusters = kmeans.labels_.astype(str)

    # DataFrame para plotagem
    df_plot = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters,
        'Número': numeros
    })

    # Sinalizar portaria selecionada
    df_plot['Selecionado'] = df_plot['Número'] == numero_desejado
    df_plot['Tamanho'] = df_plot['Selecionado'].apply(lambda x: 16 if x else 8)

    # Ordenação dos clusters
    ordem_clusters = sorted(df_plot['Cluster'].unique(), key=lambda x: int(x))
    df_plot['Cluster'] = pd.Categorical(df_plot['Cluster'], categories=ordem_clusters, ordered=True)

    # Criação do gráfico base com os clusters
    fig = px.scatter(
        df_plot[~df_plot['Selecionado']],  # Exclui o ponto selecionado do plot base
        x='PCA1',
        y='PCA2',
        color='Cluster',
        size='Tamanho',
        custom_data=['Número', 'Cluster'],
        title="Clusterização das Portarias",
        color_discrete_sequence=px.colors.qualitative.Bold,
        category_orders={"Cluster": ordem_clusters}
    )

    fig.update_traces(
        marker=dict(symbol='circle', line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate="Portaria: %{customdata[0]}<br>Cluster: %{customdata[1]}<extra></extra>"
    )

    # Adiciona o ponto da portaria selecionada por cima dos demais
    df_selected = df_plot[df_plot['Selecionado']]
    if not df_selected.empty:
        selected = df_selected.iloc[0]
        fig.add_trace(go.Scatter(
            x=[selected['PCA1']],
            y=[selected['PCA2']],
            mode='markers+text',
            marker=dict(
                size=20,
                color='red',
                line=dict(width=3, color='black'),
                symbol='circle',
                opacity=1
            ),
            name=f"Portaria {selected['Número']}",
            hovertext=f"Portaria: {selected['Número']}<br>Cluster: {selected['Cluster']}",
            hoverinfo='text',
            showlegend=False,
            text=[selected['Número']],
            textposition="top center"
        ))

    fig.update_layout(
        legend_title_text='Cluster',
        legend_traceorder="normal",
    )

    st.plotly_chart(fig, use_container_width=True)


from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import re

def gerar_nuvem_por_cluster(vetores_fasttext, todas_portarias_maio, k=3):
    # Palavras a remover
    palavras_remover = {"tribunal", "justiça", "distrito", "federal", "territórios", "a", "o", "os", "as", "um", "uma", "uns", "umas","de", "do", "da", "dos", "das","em","no", "na", "nos", "nas", "por", "para", "com", "sem", "sobre","e", "ou", "mas","que", "porque", "como", "se"}
    
    numeros = list(vetores_fasttext.keys())
    X = np.array([vetores_fasttext[n] for n in numeros])
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    clusters = kmeans.labels_
    df_clusters = pd.DataFrame({'Número': numeros, 'Cluster': clusters})

    for cluster_id in sorted(df_clusters['Cluster'].unique()):
        st.markdown(f"#### Nuvem de Palavras - Cluster {cluster_id}")
        numeros_cluster = df_clusters[df_clusters['Cluster'] == cluster_id]['Número']
        
        # Concatena os resumos do cluster
        textos = " ".join([
            todas_portarias_maio[num]['resumo']
            for num in numeros_cluster
            if num in todas_portarias_maio
        ])

        # Remove palavras indesejadas (case-insensitive)
        palavras = re.findall(r'\b\w+\b', textos.lower())
        palavras_filtradas = [p for p in palavras if p not in palavras_remover]
        textos_limpos = " ".join(palavras_filtradas)

        if textos_limpos.strip() != "":
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(textos_limpos)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("Sem texto suficiente para gerar nuvem.")

def grafico_trigramas_iniciais_por_cluster(vetores_fasttext, portarias_ultra_processadas, k=3):
    # Clusterização
    numeros = list(vetores_fasttext.keys())
    X = np.array([vetores_fasttext[n] for n in numeros])
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    clusters = kmeans.labels_.astype(str)


    # DataFrame base com clusters
    df = pd.DataFrame({
        'Número': numeros,
        'Cluster': clusters
    })

    # Ordenar clusters para garantir ordem na legenda
    ordem_clusters = sorted(df['Cluster'].unique(), key=lambda x: int(x))
    df['Cluster'] = pd.Categorical(df['Cluster'], categories=ordem_clusters, ordered=True)

    # Agrupar por cluster
    for cluster_id in ordem_clusters:
        st.markdown(f"### Trigramas da Resolução - Cluster {cluster_id}")

        numeros_cluster = df[df['Cluster'] == cluster_id]['Número']

        # Extração dos trigramas iniciais
        trigramas = []
        for num in numeros_cluster:
            if num in portarias_ultra_processadas:
                tokens = portarias_ultra_processadas[num]['conteudo']
                if len(tokens) >= 3:
                    trigrama = " ".join(tokens[:3])
                    trigramas.append(trigrama)

        # Contagem dos trigramas
        contador = Counter(trigramas)
        if not contador:
            st.write("Sem trigramas disponíveis para este cluster.")
            continue

        df_trigramas = pd.DataFrame(contador.items(), columns=['Trigrama', 'Frequência'])
        df_trigramas = df_trigramas.sort_values(by='Frequência', ascending=False).head(10)  # Top 10 trigramas

        # Criação do gráfico
        fig = px.bar(
            df_trigramas,
            x='Frequência',
            y='Trigrama',
            orientation='h',
            title=f"Top 10 Trigramas Iniciais - Cluster {cluster_id}",
            labels={'Frequência': 'Frequência', 'Trigrama': 'Trigrama'},
            color_discrete_sequence=['#636EFA']  # Azul padrão plotly
        )

        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Frequência',
            yaxis_title='Trigrama',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

def grafico_portarias_mes_cluster(vetores_fasttext, todas_portarias_maio, k=3):
    # Preparação dos clusters
    numeros = list(vetores_fasttext.keys())
    X = np.array([vetores_fasttext[n] for n in numeros])
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    clusters = kmeans.labels_

    # Montagem do dataframe
    df = pd.DataFrame({
        'Número': numeros,
        'Cluster': clusters.astype(str)  # Convertendo para string para usar como categoria
    })

    # Extração da data e do mês
    datas = []
    nomes_meses = []
    for num in numeros:
        if num in todas_portarias_maio and 'data' in todas_portarias_maio[num]:
            data_str = todas_portarias_maio[num]['data']
            try:
                data_dt = pd.to_datetime(data_str, dayfirst=True, errors='coerce')
                datas.append(data_dt)
                if pd.notnull(data_dt):
                    nomes_meses.append(data_dt.strftime('%B').capitalize())  # Nome do mês
                else:
                    nomes_meses.append('Data inválida')
            except:
                datas.append(None)
                nomes_meses.append('Data inválida')
        else:
            datas.append(None)
            nomes_meses.append('Data inválida')

    df['Data'] = datas
    df['Mês'] = nomes_meses

    # Tradução dos meses para português
    mapeamento_meses = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril',
        'May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto',
        'September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }
    df['Mês'] = df['Mês'].replace(mapeamento_meses)

    # Agrupamento por mês e cluster
    df_agrupado = df.groupby(['Mês', 'Cluster']).size().reset_index(name='Quantidade')

    # Definir ordem dos meses
    ordem_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                   'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    df_agrupado['Mês'] = pd.Categorical(df_agrupado['Mês'], categories=ordem_meses, ordered=True)

    # Definir ordem dos clusters (0, 1, 2, ...)
    ordem_clusters = sorted(df_agrupado['Cluster'].unique(), key=lambda x: int(x))
    df_agrupado['Cluster'] = pd.Categorical(df_agrupado['Cluster'], categories=ordem_clusters, ordered=True)

    # Ordenação final
    df_agrupado = df_agrupado.sort_values(['Mês', 'Cluster'])

    # Criação do gráfico Plotly
    fig = px.bar(
        df_agrupado,
        x='Mês',
        y='Quantidade',
        color='Cluster',
        barmode='stack',  # Colunas empilhadas
        title='Quantidade de Portarias por Mês e Cluster',
        labels={'Mês': 'Mês', 'Quantidade': 'Quantidade de Portarias'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Cluster',
        legend_traceorder="normal"  # Ordem da legenda conforme a ordem definida nos clusters
    )

    st.plotly_chart(fig, use_container_width=True)

# ========== Interface Streamlit ==========

st.title("Visualização e Análise das Portarias do Gabinete da Presidência - 2024")

numero_portaria = st.selectbox(
    "Selecione o número da portaria:",
    sorted(todas_portarias_maio.keys(), key=lambda x: int(x), reverse=False)
)

st.markdown("### Data da portaria selecionada:")
texto_data = todas_portarias_maio[numero_portaria]['data']
st.text(texto_data)

st.markdown("### Conteúdo da portaria selecionada:")

texto_completo = todas_portarias_maio[numero_portaria]['conteudo']

st.markdown(
    f"""
    <div style="
        padding:10px;
        max-height:300px;
        overflow-y:auto;
        background-color:transparent;
        color:white;
        line-height:1.5;
        ">
        {texto_completo.replace('\n', '<br>')}
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("### Descrição da portaria selecionada:")
texto_resumo = todas_portarias_maio[numero_portaria]["resumo"]
st.text(texto_resumo)

st.markdown("### Visualização de Entidades:")

if numero_portaria in dict_combined:
    visualizar_anotacoes_manuaais(numero_portaria)

if numero_portaria in resultados_entidades_final_final:
    visualizar_entidades_preditas(numero_portaria)

st.markdown("### Portarias mais similares:")
df_similares = encontrar_similares(numero_portaria, vetores_fasttext, todas_portarias_maio)
df_similares = df_similares.rename(columns={'numero': 'Portaria', 'similaridade': 'Similaridade', 'texto_portaria': 'Conteúdo'})
for idx, row in df_similares.iterrows():
    st.write(f"**Portaria:** {row['Portaria']} | **Similaridade:** {row['Similaridade']:.4f}")
    with st.expander("Ver conteúdo"):
        st.write(row['Conteúdo'])

st.markdown("### Visualização dos Clusters:")
gerar_grafico_clusters_plotly(vetores_fasttext, numero_portaria)

st.markdown("### Nuvens de Palavras por Cluster:")
gerar_nuvem_por_cluster(vetores_fasttext, todas_portarias_maio)

st.markdown("## Análise de Trigramas por Cluster")
grafico_trigramas_iniciais_por_cluster(vetores_fasttext, portarias_ultra_processadas)

st.markdown("### Evolução Mensal das Portarias por Cluster")
grafico_portarias_mes_cluster(vetores_fasttext, todas_portarias_maio)
