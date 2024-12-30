import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

df_treino = pd.read_csv('./conjunto_de_treinamento.csv')
df_treino.drop(['Id'], axis=1, inplace=True)
df_teste = pd.read_csv('./conjunto_de_teste.csv')
df_teste.drop(['Id'], axis=1, inplace=True)

valores_diferenciais = set(df_treino['diferenciais'].unique().tolist() + df_teste['diferenciais'].unique().tolist())

diferenciais = {valor: valor for valor in valores_diferenciais}

for valor in valores_diferenciais:
    if 'futebol e' in valor:
        diferenciais[valor] = 'futebol++'
    if 'care e' in valor:
        diferenciais[valor] = 'children care++'
    if 'churrasqueira e' in valor:
        diferenciais[valor] = 'churrasqueira++'
    if 'copa e' in valor:
        diferenciais[valor] = 'copa++'
    if 'esquina e' in valor:
        diferenciais[valor] = 'esquina++'
    if 'estacionamento visitantes e' in valor:
        diferenciais[valor] = 'estacionamento visita++'
    if 'mar e' in valor:
        diferenciais[valor] = 'mar++'
    if 'piscina e' in valor:
        diferenciais[valor] = 'piscina++'
    if 'playground e' in valor:
        diferenciais[valor] = 'playground++'
    if 'quadra poliesportiva e' in valor:
        diferenciais[valor] = 'quadra++'
    if 'ginastica e' in valor:
        diferenciais[valor] = 'ginastica++'
    if 'festas e' in valor:
        diferenciais[valor] = 'festas++'
    if 'jogos e' in valor:
        diferenciais[valor] = 'festas++'
    if 'sauna e' in valor:
        diferenciais[valor] = 'sauna++'
    if 'hidromassagem e' in valor:
        diferenciais[valor] = 'hidromassagem++'
    if valor == 'churrasqueira':
        diferenciais[valor] = np.nan
    if valor == 'estacionamento':
        diferenciais[valor] = np.nan
    if valor == 'piscina':
        diferenciais[valor] = np.nan
    if valor == 'playground':
        diferenciais[valor] = np.nan
    if valor == 'quadra poliesportiva':
        diferenciais[valor] = np.nan
    if valor == 'salao de festas':
        diferenciais[valor] = np.nan
    if valor == 'salao de jogos':
        diferenciais[valor] = np.nan
    if valor == 'sauna':
        diferenciais[valor] = np.nan
    if valor == 'frente para o mar':
        diferenciais[valor] = np.nan
    if valor == 'nenhum':
        diferenciais[valor] = np.nan

df_treino = df_treino.replace(diferenciais)
df_teste = df_teste.replace(diferenciais)

bairros = {'Aflitos': 'baixa',
 'Afogados':'baixa',
 'Agua Fria':'baixa',
 'Apipucos':'media',
 'Areias':'media',
 'Arruda':'media',
 'Barro':'baixa',
 'Beberibe':'baixa',
 'Beira Rio':'baixa',
 'Benfica':'baixa',
 'Boa Viagem':'alta',
 'Boa Vista':'media',
 'Bongi':'media',
 'Cajueiro':'media',
 'Campo Grande':'media',
 'Casa Amarela':'media',
 'Casa Forte':'alta',
 'Caxanga':'media',
 'Centro':'media',
 'Cid Universitaria':'media',
 'Coelhos':'baixa',
 'Cohab':'baixa',
 'Cordeiro':'media',
 'Derby':'media',
 'Dois Irmaos':'media',
 'Encruzilhada':'media',
 'Engenho do Meio':'media',
 'Espinheiro':'alta',
 'Estancia':'baixa',
 'Fundao':'baixa',
 'Gracas':'alta',
 'Guabiraba':'baixa',
 'Hipodromo':'baixa',
 'Ibura': 'baixa',
 'Ilha do Leite':'media',
 'Ilha do Retiro':'alta',
 'Imbiribeira':'media',
 'Ipsep':'media',
 'Iputinga':'media',
 'Jaqueira':'alta',
 'Jd S Paulo':'media',
 'Lagoa do Araca':'media',
 'Macaxeira':'baixa',
 'Madalena':'alta',
 'Monteiro':'media',
 'Paissandu':'alta',
 'Parnamirim':'media',
 'Piedade':'baixa',
 'Pina':'media',
 'Poco':'baixa',
 'Poco da Panela':'media',
 'Ponto de Parada':'baixa',
 'Prado':'media',
 'Recife':'media',
 'Rosarinho':'media',
 'S Jose':'alta',
 'San Martin':'media',
 'Sancho':'media',
 'Santana':'media',
 'Setubal':'alta',
 'Soledade':'media',
 'Sto Amaro':'media',
 'Sto Antonio':'media',
 'Tamarineira':'alta',
 'Tejipio':'media',
 'Torre':'media',
 'Torreao':'media',
 'Varzea':'media',
 'Zumbi':'baixa'}

df_treino = df_treino.replace(bairros)
df_teste = df_teste.replace(bairros)

df_treino = pd.get_dummies(df_treino, columns=['tipo', 'bairro', 'tipo_vendedor', 'diferenciais'], dtype=int)
df_teste = pd.get_dummies(df_teste, columns=['tipo', 'bairro', 'tipo_vendedor', 'diferenciais'], dtype=int)

colunas_treino = set(df_treino.columns.tolist())
colunas_teste = set(df_teste.columns.tolist() + ['preco'])

comuns = colunas_treino & colunas_teste

nao_em_treino = colunas_teste - colunas_treino
nao_em_teste = colunas_treino - colunas_teste

# Remoção das colunas:
if len(nao_em_teste) != 0:
    df_treino.drop(list(nao_em_teste), axis=1, inplace=True)
if len(nao_em_treino) != 0:
    df_teste.drop(list(nao_em_treino), axis=1, inplace=True)

df_treino = df_treino[(df_treino['preco'] > 50_000) & (df_treino['preco'] < 5_000_000)]


att_categoricos = [ x for x in df_treino.columns if df_treino[x].dtype == 'object']

for v in att_categoricos:
    print(f'\n{v:15}: {len(df_treino[v].unique())} valores únicos')
    print(f'{v}: {df_treino[v].unique()}')

colunas = df_treino.columns
colunas_baixo_pearson = []
for col in colunas:
    pearson = pearsonr(df_treino[col], df_treino["preco"])[0]
    if abs(pearson) < 0.1:
        colunas_baixo_pearson.append(col)

# Removendo as colunas com baixo pearson
df_treino.drop(colunas_baixo_pearson, axis=1, inplace=True)
df_teste.drop(colunas_baixo_pearson, axis=1, inplace=True)

X = df_treino.drop(['preco'], axis=1)
y = df_treino['preco']

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=1500,
    random_state=0
)

escala = StandardScaler()
escala.fit(X_treino)

X_treino = escala.transform(X_treino)
X_teste = escala.transform(X_teste)

df_treino_final = df_treino.sample(frac=1, random_state=0)
X_final = df_treino_final.drop(['preco'], axis=1).to_numpy()
y_final = df_treino_final['preco'].to_numpy()

X_teste_final = df_teste.to_numpy()

X_treino_final = X_final
y_treino_final = y_final

X_teste_final = X_teste_final

regressor = RandomForestRegressor(
    bootstrap=True,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=200,
    random_state=0
)
regressor.fit(X_treino, y_treino)

y_pred_teste = regressor.predict(X_teste_final)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
resultado = cross_val_score(regressor, X_treino, y_treino, cv = kfold)

for i in range(1,10):
    regressor = RandomForestRegressor(
        n_estimators=i,
        random_state=0
    )
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
    resultado = cross_val_score(regressor, X_treino, y_treino, cv = kfold)

regressor = RandomForestRegressor(
    bootstrap=True,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=10,
    n_estimators=300,
    random_state=0
)
regressor.fit(X_treino_final, y_treino_final)
y_pred_teste = regressor.predict(X_teste_final)

df_teste_original = pd.read_csv('./conjunto_de_teste.csv')
df_teste_original['preco'] = y_pred_teste

resposta_final = pd.DataFrame({'Id': df_teste_original.pop('Id'), 'preco': np.squeeze((y_pred_teste))})
resposta_final.to_csv('answer.csv', index=False)