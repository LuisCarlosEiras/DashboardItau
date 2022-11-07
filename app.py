Previsão de ações. 

# Imports
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date
import warnings
warnings.filterwarnings("ignore")

# Define a data de início para coleta  de dados
INICIO = "2015-01-01"

# Define a data de fim para coleta de dados (data de hoje, execução do script)
HOJE = date.today().strftime("%Y-%m-%d")

# Define o título do Dashboard
# st.title("Mini-Projeto 2 - Data App")
st.title("Projeto Alysida. Previsão de entrega da cadeia de suprimentos")

# Define o código das empresas para coleta dos dados de ativos financeiros
# https://finance.yahoo.com/most-active
empresas = ('PBR', 'GOOG', 'UBER', 'PFE', 'VALE', 'GGB', 'ITUB','KO')

# Define de qual empresa usaremos os dados por vez
empresa_selecionada = st.selectbox('Selecione o hash', empresas)

# Define de qual empresa usaremos os dados por vez
# empresa_selecionada = st.selectbox('Selecione a Empresa Para as Previsões de Ativos Financeiros:', empresas)

# st.selectbox('ITUB: Itaú, GGC: Gerdal, PBR: Petrobras, PFG: Pfizer')

# Função para extrair e carregar os dados
@st.cache
def carrega_dados(ticker):
    dados = yf.download(ticker, INICIO, HOJE)
    dados.reset_index(inplace = True)
    return dados

# Mensagem de carga dos dados
mensagem = st.text('Carregando os dados...')

# Carrega os dados
dados = carrega_dados(empresa_selecionada)

# Mensagem de encerramento da carga dos dados
mensagem.text('Carregando os dados...Concluído!')

# Sub-título
st.subheader('Visualização dos Dados Brutos')
st.write(dados.tail())

# Fonte
# st.subheader('Fonte: https://finance.yahoo.com/most-active')

# Função para o plot dos dados brutos
def plot_dados_brutos():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x = dados['Date'], y = dados['Open'], name = "stock_open"))
	fig.add_trace(go.Scatter(x = dados['Date'], y = dados['Close'], name = "stock_close"))
	fig.layout.update(title_text = 'Preço de Abertura e Fechamento das Ações', xaxis_rangeslider_visible = True)
	st.plotly_chart(fig)
	
# Executa a função
plot_dados_brutos()

st.subheader('Previsões com Machine Learning')

# Prepara os dados para as previsões com o pacote Prophet
df_treino = dados[['Date','Close']]
df_treino = df_treino.rename(columns = {"Date": "ds", "Close": "y"})

# Cria o modelo
modelo = Prophet()

# Treina o modelo
modelo.fit(df_treino)

# Define o horizonte de previsão
num_anos = st.slider('Horizonte de Previsão (de 1 a 4 anos):', 1, 4)

# Calcula o período em dias
periodo = num_anos * 365

# Prepara as datas futuras para as previsões
futuro = modelo.make_future_dataframe(periods = periodo)

# Faz as previsões
forecast = modelo.predict(futuro)

# Sub-título
st.subheader('Dados Previstos')

# Dados previstos
st.write(forecast.tail())
    
# Título
st.subheader('Previsão de entrega da mercadoria a partir do hash')

# Plot
grafico2 = plot_plotly(modelo, forecast)
st.plotly_chart(grafico2)

# Referências
# st.write("""Referência: 
# Mini Projeto 2, Data App, Dashboard Financeiro Interativo e em Tempo Real Para Previsão de Ativos Financeiros, 
# Capítulo 5 - Visualização de Dados e Dashboards com Python, Curso Visualização de Dados e Design de Dashboards, Data Science Academy.""")

# Fim


