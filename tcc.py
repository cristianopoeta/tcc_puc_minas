import json as _json
import pandas as _pd
import pickle as _pickle
from pathlib import Path
from time import sleep as _sleep
from tqdm.notebook import tqdm as _tdqm
from datetime import datetime as _datetime

Path.with_stem = lambda self, stem: self.with_name(stem + self.suffix)
Path.with_stem.__doc__ = """Return a new path with the stem changed."""

Path.extend_stem = lambda self, text: self.with_stem(self.stem + text)
Path.extend_stem.__doc__ = """Return a new path with the stem extended."""

txt_nos_bricks = ('14716 22886 2356 2357 2453 2453a 2453b 2454 2456 2465 3001 3001f1 3001old '
    '3001oldb 3001oldf1 3001special 3002 3002f1 3002old 3003 3003f1 3003old 3004 3004f1 '
    '3004f2 3005 3005f1 3005f2 3005f3 3006 3007 30072 3008 3008f1 3008f2 3009 3009f1 3009f2 '
    '3009f3 3010 3010f1 3010f2 3010f3 30144 30145 30400 3065 3066 3067 3245b 3245c 3622 '
    '3622f1 3754 3755 4201 4202 4204 46212 49311 6111 6112 6212 6213 700 700e 700eD 700eD2 '
    '700eX 702 702old 733 733eX 772 bhol01 bhol02 bhol03 bhol04 bhol05 bhol06 crssprt01 '
    'crssprt02 crssprt03 crssprt04 x1214')

nm_dir_tcc = 'd:/jup_ws/tcc'
p_dir_tcc = Path(nm_dir_tcc)
p_dir_tcc.mkdir(parents=True, exist_ok=True)

nm_dir_bricklink = 'bricklink'
p_dir_bricklink = p_dir_tcc / nm_dir_bricklink
p_dir_bricklink.mkdir(parents=True, exist_ok=True)

nms_bricklink_df = ['catal_bricks', 'guia_precos', 'supersets', 
                    'lista_cores', 'lista_categs', 'paises', 'img_dscr',
                    'vendas_bricks', 'vendas_bricks_nb5', ]
for nm in nms_bricklink_df:
    exec(f'pckl_df_{nm} = p_dir_bricklink / "df_{nm}.pickle"')
    
nms_bricklink_dct = ['catal_bricks', 'guia_precos', 'supersets']
for nm in nms_bricklink_dct:
    exec(f'pckl_dct_{nm} = p_dir_bricklink / "dct_{nm}.pickle"')

nms_bricklink_cat = ['paises']
for nm in nms_bricklink_cat:
    exec(f'pckl_cat_{nm} = p_dir_bricklink / "cat_{nm}.pickle"')

nm_arq_autent = 'params_autent.json'
p_arq_autent = p_dir_bricklink / nm_arq_autent
with p_arq_autent.open('r') as f_autent:
    params_autent = _json.load(f_autent)

nm_arq_cores_std = 'StudioColorDefinition.txt'
p_arq_cores_std = p_dir_bricklink / nm_arq_cores_std

def ofuscar_params(dct):
    return {k:len(v)*'*' for k, v in dct.items()}

def frame2pickle(frame, p_pickle):
    frame.to_pickle(p_pickle)
    print(f'Salvo: {p_pickle}',
      f'Tamanho: {p_pickle.stat().st_size} bytes',
      f'Modifificado em: {_datetime.fromtimestamp(p_pickle.stat().st_mtime):%d/%m/%Y %H:%M:%S}',
      sep='\n')  

def pickle2frame(p_pickle):
    print(f'Lido: {p_pickle}',
      f'Tamanho: {p_pickle.stat().st_size} bytes',
      f'Modifificado em: {_datetime.fromtimestamp(p_pickle.stat().st_mtime):%d/%m/%Y %H:%M:%S}',
      sep='\n')
    return _pd.read_pickle(p_pickle)

def object2pickle(obj, p_pickle):
    with open(p_pickle, 'wb') as f_pickle:
        _pickle.dump(obj, f_pickle)
    print(f'Salvo: {p_pickle}',
      f'Tamanho: {p_pickle.stat().st_size} bytes',
      f'Modifificado em: {_datetime.fromtimestamp(p_pickle.stat().st_mtime):%d/%m/%Y %H:%M:%S}',
      sep='\n')  
        
def pickle2object(p_pickle):
    print(f'Lido: {p_pickle}',
      f'Tamanho: {p_pickle.stat().st_size} bytes',
      f'Modifificado em: {_datetime.fromtimestamp(p_pickle.stat().st_mtime):%d/%m/%Y %H:%M:%S}',
      sep='\n')
    with open(p_pickle, 'rb') as f_pickle:
        return _pickle.load(f_pickle)

def coletar_dict(dct_coleta, iter_tqdm, desc_tqdm, unit_tqdm,
                 fn_postfix, fn_chave, fn_coleta, sleep=0):
    pbar = _tdqm(iterable=iter_tqdm, desc=desc_tqdm, unit=unit_tqdm)
    for item in pbar:
        postfix = fn_postfix(item)
        chave = fn_chave(item)
        if chave in dct_coleta:
            postfix['situação'] = 'já coletado'
            pbar.set_postfix(postfix)
            _sleep(sleep)
        else:
            postfix['situação'] = 'coletando'
            pbar.set_postfix(postfix)
            dct_coleta[chave] = fn_coleta(item)
            postfix['situação'] = 'coleta concluída'
            pbar.set_postfix(postfix)              