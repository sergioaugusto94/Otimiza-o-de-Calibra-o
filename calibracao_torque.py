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
data = pd.read_csv('7498714_SPM06_E2292RON_6094_8.2625.txt', sep = '\t',
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
data['calc_cmie'] = (data['tbinvspec']/AFCMI2PMI*data[var_ar] + 
                     data['tbcmioff'])*data['RDTQREEL']/100*data['tbrdtqlam']/100
data['calc_pumptot'] = data['PREEXH'] - data['PREM'] + data['pqdpumpfl']
data['calc_cmfint'] =  (data['calc_pumptot']*bar2nm/1000 + 
                        data['tcth2o']) 
data['calc_cmee'] = data['calc_cmie'] - data['calc_cmfint'] 

'''
Verificando o cálculo das variáveis

data['erro_calc_cmie'] = data['CMIE'] - data['calc_cmie']
data['erro_calc_cmfint'] = data['CMFINT'] - data['calc_cmfint']
data['erro_pumptot'] = (data['PUMPTOT'] - data['calc_pumptot'])/1000*bar2nm
data['erro_friction_ecu'] = data['ecu_friction'] - data['tcth2o']
data['erro_cmee'] = data['CMEE'] - data['calc_cmee']
'''

# Calculando o erro dos modelos
data['erro_tq_indicado'] = data['CMIE'] - data['IMEPH0']*bar2nm
data['erro_tq_eixo'] = data['CMEE'] - data['TORQUE_N']
data['erro_calc_cmfint'] = data['CMFINT'] - data['calc_cmfint']
data['erro_pumptot'] = (data['PUMPTOT'] - data['calc_pumptot'])/1000*bar2nm
data['erro_friction_ecu'] = data['ecu_friction'] - data['tcth2o']




x, y = tbinvspec.shape[0], tbinvspec.shape[1]
for i in range(1, x):
    for j in range(1, 2): #Lembrar de trocar o 2 por j
        erro_0 = mean_squared_error(data['IMEPH0']*bar2nm, data['calc_cmie'])
        parada = True
        lista_erros = [erro_0,0,0,0]
        lista_valor = [tbinvspec.iloc[i,j], 0]
        delta0 = 10
        k = 1
        while parada:
            valor0 = tbinvspec.iloc[i,j]
            tbinvspec.iloc[i,j] += delta0
            f_tbinvspec = function(tbinvspec)
            data = out_df(data, f_tbinvspec, 'tbinvspec', 'af', 'RPM')  
            data['calc_cmie'] = (data['tbinvspec']/AFCMI2PMI*data[var_ar] + data['tbcmioff'])*data['RDTQREEL']/100*data['tbrdtqlam']/100
            
            erro_mse = mean_squared_error(data['IMEPH0']*bar2nm, data['calc_cmie'])
            lista_erros[k%4] = erro_mse
            lista_valor[k%2] = tbinvspec.iloc[i,j]
            if erro_mse == erro_0:
                parada = False
                tbinvspec.iloc[i,j] = valor0
            delta_erro = lista_erros[k%4] - lista_erros[(k-1)%4]
            delta_valor = lista_valor[k%2] - lista_valor[(k-1)%2]
            delta0 = -delta_erro/delta_valor
            if abs(delta_erro) < 0.00011:
                parada = False
                tbinvspec.iloc[i,j] = valor0
            k += 1
            if k%100 == 99:
                print(erro_mse)
