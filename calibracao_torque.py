import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import (interp2d, interp1d)
import math
from sklearn.metrics import mean_squared_error

eng_dis = 999 # Cilindradas do motor
bar2nm = eng_dis/40/math.pi # Conversão pressão p/ torque
ar_calibrado = False # Caso o modelo de ar esteja calibrado, colocar True

if ar_calibrado:
    var_ar = 'QACADV'
else:
    var_ar = 'Q_AR_CC'

# Cria uma função de interpolação baseada no mapa passado.
def function(df): 
    if df.shape[0]>2:
        x = df.iloc[0, 1:].values
        y = df.iloc[1:,0].values
        z = df.iloc[1:, 1:].values
        funcao = interp2d(x, y, z, kind = 'linear')
    else:
        x = df.iloc[0, :].values
        z = df.iloc[1, :].values
        funcao = interp1d(x, z, kind = 'linear')
    return funcao

# Cria uma coluna no df em função de uma função criada
def out_df(df, funcao, nome_coluna, coluna_x, coluna_y = None): 
    list_out = []
    for i in range (df.shape[0]):
        if type(funcao) == scipy.interpolate.interp2d:
            z0 = funcao(df[coluna_x].values[i], df[coluna_y].values[i])
        else: 
            z0 = funcao(df[coluna_x].values[i])
        list_out.append(z0)
    list_out = pd.DataFrame(list_out, columns = [nome_coluna])
    if nome_coluna in df.columns:
        df[nome_coluna] = list_out.replace(df[nome_coluna],df[nome_coluna])
    else:
        df = pd.concat([df, list_out], axis = 1)
    return df

# Carregamento da base de dados
data = pd.read_csv('C:/Users/sergi/.spyder-py3/Calibração/7498714_SPM06_E2292RON_6094_8.2625.txt', sep = '\t',
                   decimal = ',')

# Preprocessamento da base de dados
data = data.replace('**', np.nan)
data = data.drop(0, axis = 0)
data = data.dropna(axis = 1)
data = data.drop(['DATE', 'TIME', 'MOT_MAN'] , axis = 1)
data = data.drop(['PCV_ABNT_Max', 'CVVT_ENA'] , axis = 1)
colunas_str = []
for col in data.columns:
    if type(data[col].values[0]) == str:
        colunas_str.append(col)
data[colunas_str] = data[colunas_str].apply(lambda x: x.str.replace(',', '.'))
data = data.astype(float)
data = data.reset_index(drop=True)
    
# Carregamento dos mapas de calibração (Ainda não otimizados)
pqdpumpfl = pd.read_excel('pqdpumpfl.xlsx', header = None)
etaspmax_e22 = pd.read_excel('etaspmax_e22.xlsx', header = None)
etaspmax_e100 = pd.read_excel('etaspmax_e100.xlsx', header = None)
tbcmioff = pd.read_excel('tbcmioff.xlsx', header = None)
tbinvspec = pd.read_excel('tbinvspec.xlsx', header = None)
tbrdtqlam = pd.read_excel('tbrdtqlam.xlsx', header = None)
tcth2o = pd.read_excel('tcth2o.xlsx', header = None)

def get_df_name(df):
    name = [x for x in globals() if globals()[x] is df][0]
    return name



# Função de interpolação dos mapas 
f_pqdpumpfl = function(pqdpumpfl)
f_etaspmax_e22 = function(etaspmax_e22)
f_etaspmax_e100 = function(etaspmax_e100)
f_tbcmioff = function(tbcmioff)
f_tbinvspec = function(tbinvspec)
f_tbrdtqlam = function(tbrdtqlam)
f_tcth2o = function(tcth2o)

# Criação das colunas com os dados de saída dos mapas
data['PRENORM'] = data['PREM']/1013.25
data = out_df(data, f_etaspmax_e22, 'etaspmax_e22', 'RPM', 'PRENORM')
data = out_df(data, f_etaspmax_e100, 'etaspmax_e100', 'RPM', 'PRENORM')
data = out_df(data, f_tbcmioff, 'tbcmioff', 'RPM')
data = out_df(data, f_tbrdtqlam, 'tbrdtqlam', 'LAM_R')
data = out_df(data, f_tcth2o, 'tcth2o', 'RPM', 'TH2OC')
if data['AF'].mean() < 11:
    data['af'] = 8.7
    mapa = 'etaspmax_e100'
    AFCMI2PMI = 1094.59332
else:
    data['af'] = 13.2
    mapa = 'etaspmax_e22'
    AFCMI2PMI = 1660.74957
data = out_df(data, f_tbinvspec, 'tbinvspec', 'af', 'RPM')  
data = out_df(data, f_pqdpumpfl, 'pqdpumpfl', 'RPM', mapa)

# Check dos valores calculados
data['ecu_friction'] = data['CMFINT'] - data['PUMPTOT']*bar2nm/1000

def f_calculada(variavel, df):
    if variavel == 'cmie':
        df['calc_cmie'] = (df['tbinvspec']/AFCMI2PMI*df[var_ar] + 
                             df['tbcmioff'])*df['RDTQREEL']/100*df['tbrdtqlam']/100
    elif variavel == 'pumptot':   
        df['calc_pumptot'] = (df['PREEXH'] - df['PREM'] + df['pqdpumpfl'])/1000*bar2nm
    elif variavel == 'cmfint':   
        df['calc_cmfint'] =  (df['calc_pumptot']*bar2nm/1000 + 
                                df['tcth2o'])
    else: 
        df['calc_cmee'] = df['calc_cmie'] - df['calc_cmfint']
    
    return df

data = f_calculada('cmie', data)
data = f_calculada('pumptot', data)


data['erro_pumptot'] = (-data['IMEPL0']*bar2nm - data['calc_pumptot']) #Apagar depois

'''
Verificando o cálculo das variáveis

data['erro_calc_cmie'] = data['CMIE'] - data['calc_cmie']
data['erro_calc_cmfint'] = data['CMFINT'] - data['calc_cmfint']
data['erro_pumptot'] = (data['PUMPTOT'] - data['calc_pumptot'])/1000*bar2nm
data['erro_friction_ecu'] = data['ecu_friction'] - data['tcth2o']
data['erro_cmee'] = data['CMEE'] - data['calc_cmee']
'''

# Calculando o erro dos modelos
# data['erro_tq_indicado'] = data['CMIE'] - data['IMEPH0']*bar2nm
# data['erro_tq_eixo'] = data['CMEE'] - data['TORQUE_N']
# data['erro_calc_cmfint'] = data['CMFINT'] - data['calc_cmfint']
# data['erro_pumptot'] = (data['PUMPTOT'] - data['calc_pumptot'])/1000*bar2nm
# data['erro_friction_ecu'] = data['ecu_friction'] - data['tcth2o']


def otimizador(mapa, df_real, variavel, df, mapa_x, mapa_y = None):
    x, y = mapa.shape[0], mapa.shape[1]
    if x > 2:
        for i in range(8, x): 
            for j in range(2, y): 
                erro_0 = mean_squared_error(df_real, df['calc_'+variavel])
                parada = True
                lista_erros = [erro_0,0,0,0]
                lista_valor = [mapa.iloc[i,j], 0]
                delta0 = 10
                k = 1
                while parada:
                    valor0 = mapa.iloc[i,j]
                    mapa.iloc[i,j] += delta0
                    f_mapa = function(mapa)
                    df = out_df(df, f_mapa, get_df_name(mapa), mapa_x, mapa_y)  
                    df = f_calculada(variavel, df)
                    erro_mse = mean_squared_error(df_real,  df['calc_'+variavel])
                    lista_erros[k%4] = erro_mse
                    lista_valor[k%2] = mapa.iloc[i,j]
                    if erro_mse == erro_0:
                        parada = False
                        mapa.iloc[i,j] = valor0
                    delta_erro = lista_erros[k%4] - lista_erros[(k-1)%4]
                    delta_valor = lista_valor[k%2] - lista_valor[(k-1)%2]
                    delta0 = -delta_erro/delta_valor
                    if abs(delta_erro/delta_valor) < 0.001:
                        parada = False
                        mapa.iloc[i,j] = valor0
                    k += 1
                    if k%100 == 99:
                        print(erro_mse)
    else: 
        for j in range(1, y): 
            i = 1
            erro_0 = mean_squared_error(df_real, df['calc_'+variavel])
            parada = True
            lista_erros = [erro_0,0,0,0]
            lista_valor = [mapa.iloc[i,j], 0]
            delta0 = 10
            k = 1
            while parada:
                valor0 = mapa.iloc[i,j]
                mapa.iloc[i,j] += delta0
                f_mapa = function(mapa)
                df = out_df(df, f_mapa, get_df_name(mapa), mapa_x, mapa_y)  
                df = f_calculada(variavel, df)
                erro_mse = mean_squared_error(df_real,  df['calc_'+variavel])
                lista_erros[k%4] = erro_mse
                lista_valor[k%2] = mapa.iloc[i,j]
                if erro_mse == erro_0:
                    parada = False
                    mapa.iloc[i,j] = valor0
                delta_erro = lista_erros[k%4] - lista_erros[(k-1)%4]
                delta_valor = lista_valor[k%2] - lista_valor[(k-1)%2]
                delta0 = -delta_erro/delta_valor
                if abs(delta_erro/delta_valor) < 0.001:
                    parada = False
                    mapa.iloc[i,j] = valor0
                k += 1
                print(erro_mse)
                if k%100 == 99:
                    print(erro_mse)
    return mapa


tbcmioff = otimizador(tbcmioff, data['IMEPH0']*bar2nm, 'cmie', data, 'RPM')
tbinvspec = otimizador(tbinvspec, data['IMEPH0']*bar2nm, 'cmie', data, 'af', 'RPM')
# pqdpumpfl = otimizador(pqdpumpfl, -data['IMEPL0']*bar2nm, 'pumptot', data, 'RPM', mapa)




