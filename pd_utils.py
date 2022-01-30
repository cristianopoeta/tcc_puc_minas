'''Ferramentas auxiliares para pandas com Jupyter. '''

__version__ = '20210806'

# TODO:
#   - produzir docstrings
#   - versão de resumo de série que contemple a análise de tuplas formadas por colunas de
#       dataframe (índice do resumo será multinível)

import pandas as _pd
import numpy as _np
import datetime as _dttm
from random import sample as _sample
from IPython.display import display as _display
from collections import defaultdict as _defaultdict

class DisplayPandas:
    
    def __init__(self, head=None, sample=None, tail=None, filtros=None, func=None):
        self.head = head
        self.tail = tail
        self.sample = sample
        self.filtros = filtros
        self.func = func

    def _sanear(self, head, sample, tail, filtros):
        if not filtros:
            if not(head or tail or sample):
                sample = 1
        return (head, sample, tail, filtros)

    def display(self, obj, head=None, sample=None, tail=None, filtros=None):
        head = head if head else self.head
        sample = sample if sample else self.sample
        tail = tail if tail else self.tail
        filtros = filtros if filtros else self.filtros
        head, sample, tail, filtros = self._sanear(head, sample, tail, filtros)
        func = self.func
        if func:
            dspl = lambda o: _display(func(o))
        else:
            dspl = _display
        if self.filtros:
            filtros = self.filtros
            for filtro in filtros:
                dspl(obj.loc[filtro])
        else:
            objs = [
              obj.head(head) if head else None,
              obj.iloc[sorted(_sample(range(
                *slice(head if head else None, -tail if tail else None).indices(len(obj))
              ), sample ))] if sample else None,
              obj.tail(tail) if tail else None
            ]
            dspl(_pd.concat([o for o in objs if o is not None]))
        _display(obj.shape)
        
    def __call__(self, obj, head=None, sample=None, tail=None, filtros=None):
        self.display(obj, head=head, sample=sample, tail=tail, filtros=filtros)

def d_pd(obj, head=None, sample=None, tail=None, filtros=None):
    return DisplayPandas(
        head=head, sample=sample, tail=tail, filtros=filtros
    )(obj)

class Paleta:
    
    def __init__(self, cores):
        if isinstance(cores, str):
            self.cores = cores.split(' ')
        else:
            self.cores = list(cores)
            
    def __next__(self):
        raise NotImplementedError

class PaletaCircular(Paleta):
    _cores_default = '#ff9999 #ffcc99 #ffff99 #99ff99 #aaeeff #ccccff'
    
    def __init__(self, cores=None):
        if not cores:
            cores = self._cores_default
        super().__init__(cores)
        self.proxima = 0
        
    def __next__(self):
        cor = self.cores[self.proxima]
        self.proxima = (self.proxima + 1) % len(self.cores)
        return cor

class ExemploLinha:

    def __init__(self, loc=None, iloc=None, reset_index=False, cor_tipo=None, tit_orig='origem'):
        self.loc = loc
        self.iloc = iloc
        self.reset_index = reset_index
        self.tit_orig = tit_orig
        if cor_tipo is True:
            def func_cor_tipo(tipo):
                try:
                    func_cor_tipo.cores
                except AttributeError:
                    paleta = PaletaCircular()
                    default_factory = paleta.__next__
                    func_cor_tipo.cores = _defaultdict(default_factory)
                return func_cor_tipo.cores[tipo]
            self.cor_tipo = func_cor_tipo
        else:
            self.cor_tipo = cor_tipo

    def _frame_linha(self, frame, loc=None, iloc=None):
        if loc is None and iloc is None:
            loc = self.loc
            iloc = self.iloc
        if loc is None and iloc is None:
            return frame.sample(1)
        elif loc is not None and iloc is not None:
            raise RuntimeError('Parâmetros `loc` e `iloc` não podem ser usados simultaneamente.')
        elif loc is not None:
            return frame.loc[[loc]]
        elif iloc is not None:
            return frame.iloc[[iloc]]

    def _frame_detalhes(self, frame, loc=None, iloc=None, reset_index=None):
        if reset_index is None:
            reset_index = self.reset_index
        df_linha = self._frame_linha(frame, loc=loc, iloc=iloc)
        qtd_colunas = df_linha.shape[1]
        if reset_index:
            nm_indi = 'índice'
            nm_dado = 'dados'
            if isinstance(self.tit_orig, str):
                tit_orig = self.tit_orig
            else:
                tit_orig, nm_indi, nm_dado = self.tit_orig
            self._tit_orig, self._nm_indi, self._nm_dado = tit_orig, nm_indi, nm_dado
            qtd_niveis = df_linha.index.nlevels
            nivel_origem = qtd_niveis * [nm_indi] + qtd_colunas * [nm_dado]
            niveis = [nivel_origem]
            nms_niveis = [tit_orig]
            lst_dtypes = df_linha.index.to_frame(index=False).dtypes.to_list()
            df_linha = df_linha.reset_index()
        else:
            qtd_niveis = 0
            niveis = []
            nms_niveis = []
            lst_dtypes = []
        df = _pd.DataFrame({'valor_exemplo': df_linha.astype('O').iloc[0]})
        nivel_posicao = list(range(qtd_niveis)) + list(range(qtd_colunas))
        niveis.extend([nivel_posicao, df.index.to_list()])
        nms_niveis.extend(['i', 'nome_coluna'])
        indice = _pd.MultiIndex.from_arrays(niveis, names=nms_niveis)
        df.index = indice
        lst_dtypes.extend(frame.dtypes.to_list())
        df['dtype_coluna'] = [x.name for x in lst_dtypes]
        df['classe_valor'] = [df_linha.iat[0, col].__class__.__name__
                                  for col in range(df_linha.shape[1])]
        if reset_index:
            stlr = df.style.set_table_styles(
                 [{'selector':'th.row{0}, td.row{0}'.format(qtd_niveis),
                 'props':'border-top:1px solid black;'}] )
            return stlr
        else:
            return df
    
    def montar(self, frame, loc=None, iloc=None, reset_index=None):
        detalhes = self._frame_detalhes(
                        frame=frame, loc=loc, iloc=iloc, reset_index=reset_index)
        if self.cor_tipo:
            if isinstance(detalhes, _pd.io.formats.style.Styler):
                stlr = detalhes
            else:
                stlr = detalhes.style
            def cor_dtype_classe(sr):
                tipo = (sr['dtype_coluna'], sr['classe_valor'])
                cor = self.cor_tipo(tipo)
#                 print(tipo, cor)
                estilos = [f'background-color:{cor};' for val in sr]
                return estilos
            stlr.apply(cor_dtype_classe, axis=1)
            return stlr
        else:
            return detalhes
    
    def __call__(self, frame, loc=None, iloc=None, reset_index=None):
        return self.montar(frame=frame, loc=loc, iloc=iloc, reset_index=reset_index)
    
def exemplo_linha(frame, loc=None, iloc=None, reset_index=False, cor_tipo=None):
    return ExemploLinha(
        loc=loc, iloc=iloc, reset_index=reset_index, cor_tipo=cor_tipo
    )(frame)

class ResumoTipos:
    
    def __init__(self, reset_index=False, tit_indi='dtype, classe', 
                 tit_colu='nome coluna', tit_posi='i', tit_orig='origem', tit_tota='TOTAL'):
        self.reset_index = reset_index
        self.tit_indi = tit_indi
        self.tit_orig = tit_orig
        self.tit_posi = tit_posi
        self.tit_colu = tit_colu
        self.tit_tota = tit_tota
        
    def montar(self, obj):
        reset_index = self.reset_index
        tit_indi = self.tit_indi
        tit_posi = self.tit_posi
        tit_orig = self.tit_orig
        tit_colu = self.tit_colu
        tit_tota = self.tit_tota

        qtd_colunas = 1 if obj.ndim == 1 else obj.shape[1]
        if reset_index:
            nm_indi = 'índice'
            nm_dado = 'dados'
            if isinstance(self.tit_orig, str):
                tit_orig = self.tit_orig
            else:
                tit_orig, nm_indi, nm_dado = self.tit_orig
            self._tit_orig, self._nm_indi, self._nm_dado = tit_orig, nm_indi, nm_dado
            qtd_niveis = obj.index.nlevels
            nivel_origem = qtd_niveis * [nm_indi] + qtd_colunas * [nm_dado]
            niveis = [nivel_origem]
            nms_niveis = [tit_orig]
            obj = obj.reset_index()
            tpl_tota = ('', '', tit_tota)
        else:
            qtd_niveis = 0
            niveis = []
            nms_niveis = []
            tpl_tota = ('', tit_tota)
        if obj.ndim == 1:
            colunas = [obj]
            nms_cols = [obj.name]
            nms_dtps = [obj.dtype.name]
        else:
            colunas = [obj[nm] for nm in obj]
            nms_cols = obj.columns.to_list()
            nms_dtps = [dtp.name for dtp in obj.dtypes]
        nivel_posicao = list(range(qtd_niveis)) + list(range(qtd_colunas))
        niveis.extend([nivel_posicao, nms_cols])
        nms_niveis.extend([tit_posi, tit_colu])
        ind_cols = _pd.MultiIndex.from_arrays(niveis, names=nms_niveis)
        df = _pd.concat(
                [_pd.Series(
                    [f'{nm_dtp}, {col.iat[i].__class__.__name__}' 
                     for i in range(col.shape[0])]).value_counts(dropna=False)
                 for nm_col, nm_dtp, col in zip(nms_cols, nms_dtps, colunas)]
            , axis=1
        ).set_axis(ind_cols, axis=1).rename_axis(tit_indi).sort_index()
        df.loc[tit_tota] = df.sum(axis=0)
        df[tpl_tota] = df.sum(axis=1)
        shp = df.shape
        stlr = df.style
        (
            stlr
            .format(na_rep='', precision=0)
            .set_table_styles(
                [
                dict(selector=','.join([f'th.col{i},td.col{i}' for i in range(1, shp[1]-1)]),
                    props='border-left:1px solid lightgrey;'),
                dict(selector=f'th.col0,td.col0,th.col{shp[1]-1},td.col{shp[1]-1}', 
                    props='border-left:1px solid black;'),
                dict(selector=f'th.row{shp[0]-1},td.row{shp[0]-1}', 
                    props='border-top:1px solid black;'),
                dict(selector=f'thead > tr:nth-child({3+reset_index}) > th:nth-child(1)',
                      props='border-top:1px solid black;border-bottom:1px solid lightgrey;'), 
                ]
                +
                (
                [] if not reset_index else
                [
                dict(selector=f'th.col{qtd_niveis},td.col{qtd_niveis}', 
                    props='border-left:1px solid black;'),
                dict(selector='th.col_heading.level1',
                    props='border-top:1px solid lightgrey;'),
                dict(selector=f'th.level1.col{shp[1]-1}', props='border-top:none;'), 
                ] 
                )
            )
        )
        return stlr
    
    def __call__(self, obj):
        return self.montar(obj)
    
def resumo_tipos(obj, reset_index=False, tit_indi='dtype, classe', 
        tit_colu='nome_coluna', tit_posi='i', tit_orig='origem', tit_tota='TOTAL'):
    return ResumoTipos(
        reset_index=reset_index, tit_indi=tit_indi, 
        tit_colu=tit_colu, tit_posi=tit_posi, tit_orig=tit_orig, tit_tota=tit_tota
    )(obj)

class ResumoCateg:
    
    modelo_apply = '''{
    ('Qtd. distintos','Distintos'): lambda sr:
        2*['background-color:#ffff99;'] if sr['Qtd. distintos']<10 else 2*[None],
    ('Qtd. modas','F. moda','P. moda','Moda'): lambda sr:
        4*['background-color:#99ff99']
        if (sr['F. moda']>1 and sr['P. moda']<1 and (sr['Qtd. modas']*sr['P. moda'])>0.25)
        else 4*[None]
}'''
    modelo_applm = '''{
    'Qtd. distintos': lambda x: 'color:blue;font-weight:bold;' if x<10 else None,
    'P. não nulos': lambda x: 'color:red;font-weight:bold;background-color:#ffff99;'
        if x<=0.75 else None,
    'P. nulos': lambda x: 'color:red;font-weight:bold;background-color:#ffff99;'
        if x>0.25 else None
}'''

    def __init__(self, lim_distin=5, lim_modas=5, normalize=None,
            reset_index=False, percent=True, dct_apply=None, dct_applm=None,
            tit_vari='coluna', tit_dtyp='dtype', tit_posi='i', tit_orig='origem', 
            tit_esta='estatística'): 
        self.lim_distin  = lim_distin
        self.lim_modas   = lim_modas
        self.normalize   = normalize
        self.reset_index = reset_index
        self.percent     = percent
        if dct_apply is True:
            self.dct_apply = eval(self.modelo_apply)
        else:
            self.dct_apply   = dct_apply
        if dct_applm is True:
            self.dct_applm = eval(self.modelo_applm)
        else:
            self.dct_applm   = dct_applm
        self.tit_vari    = tit_vari
        self.tit_dtyp    = tit_dtyp
        self.tit_posi    = tit_posi
        self.tit_orig    = tit_orig
        self.tit_esta    = tit_esta

    def _colunas_indice(self, obj):
        reset_index = self.reset_index
        tit_vari = self.tit_vari
        tit_dtyp = self.tit_dtyp
        tit_posi = self.tit_posi
        tit_orig = self.tit_orig
        
        qtd_colunas = 1 if obj.ndim == 1 else obj.shape[1]
        if reset_index:
            nm_indi = 'índice'
            nm_dado = 'dados'
            if isinstance(self.tit_orig, str):
                tit_orig = self.tit_orig
            else:
                tit_orig, nm_indi, nm_dado = self.tit_orig
            self._tit_orig, self._nm_indi, self._nm_dado = tit_orig, nm_indi, nm_dado
            qtd_niveis = obj.index.nlevels
            nivel_origem = qtd_niveis * [nm_indi] + qtd_colunas * [nm_dado]
            niveis = [nivel_origem]
            nms_niveis = [tit_orig]
            obj = obj.reset_index()
        else:
            qtd_niveis = 0
            niveis = []
            nms_niveis = []
        if obj.ndim == 1:
            colunas = [obj]
            nms_cols = [obj.name]
            lst_dtypes = [obj.dtype]
        else:
            colunas = [obj[nm] for nm in obj]
            nms_cols = obj.columns.to_list()
            lst_dtypes = obj.dtypes.to_list()
        nivel_posicao = list(range(qtd_niveis)) + list(range(qtd_colunas))
        nivel_dtype = [x.name for x in lst_dtypes]
        niveis.extend([nivel_posicao, nms_cols, nivel_dtype])
        nms_niveis.extend([tit_posi, tit_vari, tit_dtyp])
        # for nm, nv in zip(nms_niveis, niveis):
        #     print(nm, len(nv), nv)
        indice = _pd.MultiIndex.from_arrays(niveis, names=nms_niveis)
        return (colunas, indice)

    def _linhas_estats(self, colunas):
        lim_distin = self.lim_distin
        lim_modas = self.lim_modas
        linhas = []
        for sr in colunas:
            tamanho = len(sr)
            nao_nulos = sr.notna().sum()
            nulos = sr.isna().sum()
            value_counts = sr.value_counts()
            qtd_distin = len(value_counts[value_counts > 0])
            if qtd_distin > 0:
                if qtd_distin == 1:
                    distintos = value_counts.index[0]
                elif qtd_distin <= lim_distin:
                    distintos = value_counts.index.to_list()
                else:
                    distintos = '+++'
                freq_moda = value_counts.iloc[0]
                modas = value_counts[value_counts == freq_moda]
                qtd_modas = len(modas)
                if qtd_modas == 1:
                    moda = modas.index[0]
                elif qtd_modas <= lim_modas:
                    moda = modas.index.to_list()
                else:
                    moda = '+++'
            else:
                distintos = _np.nan
                freq_moda = _np.nan
                qtd_modas = _np.nan
                moda = _np.nan
            linhas.append( (tamanho, qtd_distin, distintos, nao_nulos, nulos,
                            qtd_modas, freq_moda, moda) )
        return linhas
            
    def _frame_estats(self, linhas, indice):
        normalize = self.normalize
        tit_esta = self.tit_esta
        if linhas:
            ind_cols = _pd.Index(['Tamanho', 'Qtd. distintos', 'Distintos', 
                        'F. não nulos', 'F. nulos', 'Qtd. modas', 'F. moda', 'Moda']
                                 , name=tit_esta)
            df = _pd.DataFrame(linhas, columns=ind_cols, index=indice)
            if normalize is not False:
                df['P. não nulos'] = df['F. não nulos'] / df['Tamanho']
                df['P. nulos'] = df['F. nulos'] / df['Tamanho']
                df['P. moda'] = df['F. moda'] / df['F. não nulos']
                if normalize is True:
                    df = df[[nm_col.replace('F. ', 'P. ') 
                             for nm_col in ind_cols]]
                else:
                    df = df[[nm_col.replace(('~', 'F. ')[i], 'P. ')
                        for nm_col in ind_cols for i in range(1 + ('F. ' in nm_col))]]
            return df
        
    def _formatar_estats(self, frame):
        normalize = self.normalize
        reset_index = self.reset_index
        percent = self.percent
        dct_apply = self.dct_apply
        dct_applm = self.dct_applm
        formatar_percent = percent and (not normalize is False)
        formatar = (reset_index or formatar_percent or dct_apply or dct_applm)
        if not(formatar):
            return frame
        else:
            stlr = frame.style
            if reset_index:
                qtd_niveis = (frame.index.copy().droplevel(-1).droplevel(-1)
                              .to_list().index((self._nm_dado, 0)))
                stlr.set_table_styles(
                    [{'selector':'th.row{0}, td.row{0}'.format(qtd_niveis),
                    'props':'border-top:1px solid black;'}] )
            if formatar_percent:
                if percent is True:
                    percent = '{:.2%}'
                if isinstance(percent, str):
                    percent = percent.format
                cols_perce = [nm for nm in frame.columns if nm.startswith('P. ')]
                stlr = stlr.format(percent, subset=cols_perce)
            if dct_apply:
                for nms_cols, fn_stl in dct_apply.items():
                    ind_cols = _pd.Index(nms_cols)
                    if ind_cols.difference(frame.columns).empty:
                        stlr.apply(fn_stl, subset=ind_cols, axis=1)
            if dct_applm:
                for nm_col, fn_stl in dct_applm.items():
                    if nm_col in frame:
                        stlr = stlr.applymap(fn_stl, subset=nm_col)
            return stlr

    def montar(self, obj):
        colunas, indice = self._colunas_indice(obj)
        linhas = self._linhas_estats(colunas)
        frame = self._frame_estats(linhas, indice)
        return self._formatar_estats(frame)
    
    def __call__(self, obj):
        return self.montar(obj)

def resumo_categ(obj, lim_distin=5, lim_modas=5, normalize=None,
            reset_index=False, percent=True, dct_apply=None, dct_applm=None,
            tit_vari='coluna', tit_posi='i', tit_orig='origem', tit_esta='estatística'):
    return ResumoCateg(
        lim_distin=lim_distin, lim_modas=lim_modas, normalize=normalize, 
        reset_index=reset_index, percent=percent, dct_apply=dct_apply, dct_applm=dct_applm, 
        tit_vari=tit_vari, tit_posi=tit_posi, tit_orig=tit_orig, tit_esta=tit_esta
    )(obj)

class ResumoSerie:
    
    # TODO: parâmetro de limite de linhas para agrupamento das demais linhas em 'Diversos'?
    #       teria que considerar critério e sentido do ordenamento.
    def __init__(self, dropna=False, sort='F', ascending=False, normalize=None, percent=True, 
            nome_ind='Valor', nome_col='Freq.', nome_tot='Total', nome_ind_col=True):
        #self.lim_distin = lim_distin
        self.dropna = dropna
        self.sort = sort
        self.ascending = ascending
        self.normalize = normalize
        self.percent = percent
        self.nome_ind = nome_ind
        self.nome_col = nome_col
        self.nome_tot = nome_tot
        self.nome_ind_col = nome_ind_col
        
    def _frame_serie(self, serie):
        #lim_distin = self.lim_distin
        dropna = self.dropna
        sort = self.sort
        ascending = self.ascending
        normalize = self.normalize
        percent = self.percent
        nome_ind = self.nome_ind
        nome_col = self.nome_col
        nome_tot = self.nome_tot
        nome_ind_col = self.nome_ind_col
        
        # somente objetos não hashable precisariam de conversão para str
        # TODO: vale a pena testar se objetos não são hashable antes de converter?
        if serie.dtype.name == 'object':
            serie = serie.apply(str)
            
        sr_fa = serie.value_counts(dropna=dropna)
                                  # valor, índice
        if sort[0].upper() in ('V', 'I'):
            sr_fa = sr_fa.sort_index(ascending=ascending)
                                  # frequência, coluna, quantidade, percentual
        elif sort[0].upper() in ('Q', 'C', 'F', 'P'):
            sr_fa = sr_fa.sort_values(ascending=ascending)
        
        # TODO: necessário try/except depois que se garante que valores da série são hashable?
        try:
            sr_fa.loc[nome_tot] = sr_fa.sum()
        except:
            sr_fa.index = sr_fa.index.apply(str)
            sr_fa.loc[nome_tot] = sr_fa.sum()

        if normalize is False:
            srs = {nome_col: sr_fa}
        else:
            sr_fr = sr_fa / sr_fa.iloc[-1]
            if normalize is True:
                srs = {nome_col: sr_fr}
            elif normalize is None:
                if isinstance(nome_col, str):
                    srs = {f'{nome_col} abs.': sr_fa,
                        f'{nome_col} rel.': sr_fr}
                else:
                    srs = {nm: sr for nm, sr in zip (nome_col, [sr_fa, sr_fr])}
            else:
                raise TypeError('Parâmetro `normalize` deve ser True, False ou None. ')
        dtfr = _pd.DataFrame(srs)
        dtfr.index.name = nome_ind
        if nome_ind_col is True:
            nome_ind_col = serie.name
        elif callable(nome_ind_col):
            nome_ind_col = nome_ind_col(serie.name)
        if nome_ind_col:
            dtfr.columns.name = nome_ind_col
        stlr = dtfr.style
        if not normalize is False and percent:
            if percent is True:
                percent = '{:.2%}'
            if isinstance(percent, str):
                percent = percent.format
            stlr.format(percent, subset=dtfr.columns[-1])
        stlr.set_table_styles([
            dict(
                selector=f'th.row{dtfr.shape[0]-1},td.row{dtfr.shape[0]-1}',
                props='border-top:1px solid black;'
            )
        ])
        return stlr
    
    def montar(self, obj):
        if obj.ndim == 1:
            return self._frame_serie(obj)
        else:
            resumos = []
            for nm_col, col in obj.iteritems():
                try:
                    resumo = self._frame_serie(col)
                except Exception as e:
                    print(e)
                else:
                    resumos.append(resumo)
            if resumos:
                return resumos
        
    def __call__(self, obj):
        return self.montar(obj)
    
def resumo_serie(obj, dropna=False, sort='F', ascending=False, normalize=None, percent=True, 
            nome_ind='Valor', nome_col='Freq.', nome_tot='Total', nome_ind_col=True):
    return ResumoSerie(
        dropna = dropna, sort = sort, ascending = ascending, normalize = normalize, 
        percent = percent, nome_ind = nome_ind, nome_col = nome_col, nome_tot = nome_tot, 
        nome_ind_col = nome_ind_col
    )(obj)

#--------------------------------------------------------------------------------------------------

def read_sql(conexao, sql, max_sql=100, cols_sort=None, sort_asc=True, max_rows=5, **kwargs):
    print(_dttm.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    sql_print = ' '.join(sql.replace('\n',' ').split())
    if max_sql:
        sql_print = sql_print[:max_sql] + ('(...)' if len(sql_print)>max_sql else '')
    print('[', sql_print, ']', sep='')
    df = _pd.read_sql(sql, conexao, **kwargs)
    if cols_sort:
        df.sort_values(by=cols_sort, ascending=sort_asc, inplace=True)
    _pd.options.display.max_rows = max_rows
    _display(df)
    print(_dttm.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    return df

# TODO:
#    - Criar um parâmetro tit_serie que nomeará o índice dos rótulos de linha/coluna do
# dataframe original (ambos viram linha no dataframe resumo).
# Problema: Quando índice do dataframe tem vários níveis, teria que haver vários nomes.
# Solução: Quando met_resumo==None, .apply(pd.value_counts) transforma índice multinível
# em um nível com tuplas. Daria para fazer o mesmo quando met_resumo<>None.
def resumo_df_bool(frame, axis=0, met_resumo='all', tit_resumo='Resumo',
                   met_total='all', tit_total='\u03A3', fmt_lbl=True):
    if met_resumo:
        mthd_resu = getattr(frame, met_resumo)
        sr_resu = mthd_resu(axis=axis)
        if fmt_lbl:
            sr_resu.index = ['[{}]'.format(lbl.replace('_',' ')) for lbl in sr_resu.index]
        if met_total:
            mthd_geral = getattr(sr_resu, met_total)
            sr_resu[tit_total] = mthd_geral()
        df_out = _pd.DataFrame({tit_resumo: sr_resu})
    else:
        sigma = '\u03A3'
        lst_TFS = [True, False, sigma]
        if axis in (0, 'index'):
            df_out = _pd.DataFrame(frame.apply(
                        _pd.value_counts, axis=axis), index=lst_TFS).T.fillna(0).astype(int)
        elif axis in (1, 'columns'):
            df_out = _pd.DataFrame(frame.apply(
                        _pd.value_counts, axis=axis), columns=lst_TFS).fillna(0).astype(int)
        df_out[sigma] = df_out.sum(axis=1)
        df_out.columns.name = tit_resumo
        if fmt_lbl:
            df_out.index = ['[{}]'.format(lbl.replace('_',' ')) for lbl in df_out.index]
        if tit_total:
            df_out.loc[tit_total] = df_out.sum(axis=0)
    return df_out

def compara_cols_dentro_dfs(dct_frames, lst_compars, erro=0.000001, resumo=False,
                           resumo_comp='Todos', resumo_dtfr='Todos'):
    sigma = '\u03A3';
    delta = '\u0394'.format(erro)
    delta_lt_erro = '|\u0394|<{}'.format(erro)
    cols_slice = list(set([col for comp in lst_compars for col in comp]))
    #dct_dfs = {nm: df[cols_slice].fillna(0) for nm, df in dct_frames.items()}
    dct_dfs = {'[{}]'.format(nm): df[cols_slice].fillna(0) for nm, df in dct_frames.items()}
    lst_df_out = []
    for comparacao in lst_compars:
        colunas = list(comparacao)
        while colunas:
            col1 = colunas.pop(0)
            for col2 in colunas:
                lst_lins_dfs = []
                nome_comp = '[{}]-[{}]'.format(col1, col2)
                for nome_df, dfr in dct_dfs.items():
                    sr = abs(dfr[col1] - dfr[col2]) < erro
                    if resumo:
                        lst_lins_dfs.append([sr.all()])
                    else:
                        lst_lins_dfs.append(sr.value_counts())
                if resumo:
                    df_comp = _pd.DataFrame.from_records(
                    lst_lins_dfs,
                        index=_pd.Index(dct_dfs.keys(), name='Dataframe'),
                        columns=[nome_comp])
                    df_comp.columns.name = delta_lt_erro
                else:
                    df_comp = _pd.DataFrame.from_records(
                        lst_lins_dfs, index=_pd.Index(dct_dfs.keys(), name='Dataframe'),
                        columns=[True, False, sigma]).fillna(0).astype(int)
                    df_comp[sigma] = df_comp.sum(axis=1)
                    df_comp = _pd.concat([df_comp], axis=1, keys=[nome_comp],
                                    names=[delta, delta_lt_erro])
                lst_df_out.append(df_comp)
    d_out = _pd.concat(lst_df_out, axis=1)
    if resumo:
        if resumo_comp:
            d_out.loc[resumo_comp] = d_out.all()
        if resumo_dtfr:
            d_out[resumo_dtfr] = d_out.all(axis=1)
    return d_out.T

def compara_cols_entre_dfs(dct_frames, lst_cols, erro=0.000001, resumo=True,
                           resumo_colu='Todos', resumo_comp='Todos'):
    sigma = '\u03A3'
    delta = '\u0394'.format(erro)
    delta_lt_erro = '|\u0394|<{}'.format(erro)
    dct_rename = {col: '[{}]'.format(col.replace('_',' ')) for col in lst_cols}
    dct_dfs = {nm: df[lst_cols].fillna(0).rename(dct_rename, axis=1)
                for nm, df in dct_frames.items()}
    nomes_dfs = list(dct_dfs.keys())
    lst_df_out = []
    while nomes_dfs:
        nome_df1 = nomes_dfs.pop(0)
        for nome_df2 in nomes_dfs:
            nome_comp = '[{}]-[{}]'.format(nome_df1, nome_df2)
            df_bool = abs(dct_dfs[nome_df1]-dct_dfs[nome_df2])<erro
            if resumo:
                lst_df_out.append(_pd.DataFrame.from_records(
                                    [df_bool.all()], index=[nome_comp]))
            else:
                df_count = _pd.DataFrame(df_bool.apply(_pd.value_counts),
                                    index=[True, False, sigma]).fillna(0).astype(int)
                df_count.loc[sigma] = df_count.sum()
                lst_df_out.append(_pd.concat([df_count], keys=[nome_comp]))
    d_out = _pd.concat(lst_df_out)
    d_out.columns = d_out.columns.copy()
    d_out.columns.name = 'Colunas'
    nome_ind_out = delta_lt_erro if resumo else (delta, delta_lt_erro)
    if resumo:
        d_out.index.set_names(nome_ind_out, inplace=True)
        if resumo_colu:
            d_out.loc[resumo_colu] = d_out.all()
        if resumo_comp:
            d_out[resumo_comp] = d_out.all(axis=1)
    else:
        d_out.index.set_names((delta, delta_lt_erro), inplace=True)
    return d_out

'''
class Navegador(object):

    def __init__(self, titulo, frames, qtd_linhas, posicao):
        self.titulo = titulo
        self.frames = frames
        self.qtd_linhas = qtd_linhas
        self.posicao = posicao

    def mostrar(self, passo=1, posicao=None):
        _pd.options.display.max_rows = self.qtd_linhas
        if posicao:
            self.posicao = posicao
        print('[{}] Atual posição: {}'.format(self.titulo, self.posicao))
        lst_display = []
        lst_posi_max = []
        for df in self.frames:
            lst_display.append(
                df.iloc[self.posicao * self.qtd_linhas :
                       (self.posicao + 1) * self.qtd_linhas])
            lst_posi_max.append(df.shape[0]//self.qtd_linhas)
        posi_max = max(lst_posi_max)
        _display(*lst_display)
        self.posicao = (self.posicao + passo) % posi_max
        print('[{}] Próxima posição: {}'.format(self.titulo, self.posicao))
'''