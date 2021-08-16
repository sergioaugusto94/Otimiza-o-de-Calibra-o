import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import (interp2d, interp1d)
import math

#Cria uma função de interpolação baseada no mapa passado.
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

#Cria uma coluna no df em função de uma função criada
def out_df(df, funcao, nome_coluna, coluna_x, coluna_y = None): 
    list_out = []
    for i in range (df.shape[0]):
        if type(funcao) == scipy.interpolate.interp2d:
            z0 = funcao(df[coluna_x].values[i], df[coluna_y].values[i])
        else: 
            z0 = funcao(df[coluna_x].values[i])
        list_out.append(z0)
    list_out = pd.DataFrame(list_out, columns = [nome_coluna])
    df = pd.concat([df, list_out], axis = 1)
    return df

# Carregamento da base de dados
data = pd.read_csv('7498714_SPM06_E2292RON_6094_8.2625.txt', sep = '\t',
                   decimal = ',')

# Preprocessamento da base de dados
data = data.replace('**', np.nan)
data = data.dropna(axis = 1)
data = data.drop(['DATE', 'TIME', 'MOT_MAN', 'DIN_TOT', 'PCV_ABNT_Max', 
                  'CVVT_ENA'] , axis = 1)
data = data.apply(lambda x: x.str.replace(',', '.'))
data = data.drop(0, axis = 0)
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
data = out_df(data, f_etaspmax_e22, 'etaspmax_e22', 'RPM', 'PREM')
data = out_df(data, f_etaspmax_e100, 'etaspmax_e100', 'RPM', 'PREM')
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
data['ecu_friction'] = data['CMFINT'] - data['PUMPTOT']*999/40/math.pi/1000
data['calc_cmie'] = (data['tbinvspec']/AFCMI2PMI*data['QACADV'] + 
                     data['tbcmioff'])*data['RDTQREEL']/100*data['RDTQLAM']/100  
