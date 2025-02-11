# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Wed Jan  1 16:27:40 2025

@author: Francesco Zammori PhD
University of Parma (Italy)
"""

from typing import Any, Optional, Iterable, Callable, Generator, TypeVar
from copy import deepcopy
import math
import string
import itertools
import random
from functools import partial

class Val_Doppio(Exception):
    pass

class Val_Mancante(Exception):
    pass

class Prob_Incompatibili(Exception):
    pass

class Prob_Errate(Exception):
    pass

 
class Network:
    """ rappresenta le catene di probabilità 
            tramite dizionari annidati, di cui il primo
                può avere solo una chiave, prodotto trainante 
    
    es. {1:{2:{3:{},4:{5:{}}}, 6:{7:{}}}}

    """   
    def __init__(self, start_val:Optional[int] = None):
        self.nw:dict[int, dict[...]] = {} 
        if start_val is not None: self.nw[start_val] = {} 
    
    def __repr__(self) -> str:
        return f'Nw - {self.n_nodes}'
    
    def __str__(self) -> str:
        return repr(self.nw)
    
    @property 
    def n_nodes(self) -> int:
        return len(tuple(self.all_keys()))
    
    def all_keys(self) -> Generator[int, None, None]:
        """ restituisce tutte le chiavi 
                    dei dizionari annidati """
        def _all_k(dct:dict):
            for k, val in dct.items():
                yield k
                yield from _all_k(val)
        
        yield from _all_k(self.nw)
    
    def add_x(self, key:int = 1, at:Optional[int] = None):
        """ aggiunge una coppia chiave dizionario vuoto
                in una certa posizione """
        if self.nw == {}: self.nw = {key:{}} # inseriamo il primo
        else:
            if key in self.all_keys(): raise Val_Doppio("Valore doppio inammissibile")
            sub_ntw = self.find_x(at) # potrebbe essere None, errore gestito dopo ...
            try: sub_ntw[key] = {}
            except TypeError: raise Val_Mancante("Dizionario d'inserimento mancante")
            
    def _add(self, new_val:int|dict, key: int, at:int):
        sub_ntw = self.find_x(at) # potrebbe essere None, errore gestito dopo ...
        try: sub_ntw[key] = new_val 
        except TypeError: raise Val_Mancante("Dizionario d'inserimento mancante")
        
    def __add__(self, other:Network) -> Network:
        """ aggiunge una sotto network al 
                    network principale """
        def _check(obj_1, obj_2, exc:bool) -> bool:
            key = tuple(obj_2.nw.keys())[0]
            # i valori di obj_2 non devono figurare in obj_1
            cond_1 = all(x not in obj_1.all_keys() for x in obj_2.all_keys() if x != key)
            cond_2 = key in obj_1.all_keys()
            if cond_1 and cond_2: return True 
            if not exc: return False 
            if not cond_1: raise Val_Doppio("Valore doppio inammissibile")
            if not cond_2: Val_Mancante("Dizionario d'inserimento mancante")
        
        new = Network()
        if self.nw == {}: new.nw = deepcopy(other.nw)
        else:
            
            if _check(self, other, False):
                obj1, obj2 = self, other
            else: 
                # controlla scambiando l'ordine, se ancora False eccezione
                _ = _check(other, self, True) 
                obj1, obj2 = other, self
            
            new.nw = deepcopy(obj1.nw)
            key = tuple(obj2.nw.keys())[0]
            new.find_x(key).update(deepcopy(obj2.nw[key]))
            return new

    def find_x(self, x:int) -> dict|None:
        """ restituisce il dizionario associato 
                    alla chiave x """
        def _find_one(x, net):
            if x in net.keys(): return net[x]
            # if net == {}: return None # arrivati alla fine
            for y in net.values(): 
                out = _find_one(x, y)
                if out is not None: return out
            else: return None
        
        if x not in self.all_keys(): return None
        return _find_one(x, self.nw)
                        
    def fl_successors(self, x) -> tuple[int]|None:
        """ trova i successori di primo livello """
        sc = self.find_x(x)
        if sc is None: return None
        return tuple(sc.keys())

    def all_successors(self, x:int)-> Generator[list[int], None, None]:
        """ restituisce tutti i successori 
                raggruppati per livello """
        
        def _succ(vals:list) -> list:
            out = []
            for x in vals:
                sc = self.find_x(x)
                out += list(sc.keys())
            return out
            
        x = [x]
        net = self.nw
        while True:
            x = _succ(x)
            if x == []: break
            yield x

    def all_predecessors(self, x, net:Optional[dict] = None, pr = ())-> tuple[int]:
        """ restituisce tutti i predecessori """
        if net is None: net = self.nw
        if x in net.keys(): return pr
        if x == {}: return None
        for key in net.keys():
            out = self.all_predecessors(x, net[key], pr + (key, ))
            if out is not None: return out


lc_letters = string.ascii_lowercase
Sk = TypeVar('Sk')
m_prob =  dict[int, float] # probabilità marginale es 1: 0.43
c_prob = dict[tuple[int, int|tuple[int,int]], float] # probabilità condizionata es P(1|2) -> (1, 2): 0.8, P(1|2,3) -> (1, (2,3))
gen_function = Callable[[float, int], Generator[float, None, None]]

def gen_prob(perc_max = 0.75, max_val = 0.6):
    """ genera probabilità casuali, in modo
         da saturare la probabilità restate Pr. 
            Ciascuna è minore o uguale di Pr*perc_max """
    def inner(res_pr:float, n_pr:int):
        for i in range(n_pr - 1):
            pr = random.uniform(0, min(res_pr*perc_max, max_val))
            yield pr
            res_pr -= pr
        yield min(res_pr, max_val)
    return inner


def gen_network(prob:Iterable[tuple[int, int]]):
    """ generiamo il network a partire dalle 
            probabilità """
    # raggruppiamo per chiave 
    p_ragg = {}
    for ps, pd in prob: p_ragg.setdefault(pd, []).append((ps, pd))
    # creiamo i sotto network
    networks = [Network(key) for key in p_ragg.keys()]
    for pr, nw in zip(p_ragg.values(), networks):
        for p in pr: nw.add_x(p[0], p[1])
    # uniamo i nwtwork in un network unico
    while True:
       obj1 = networks[0]
       if len(networks) == 1: return obj1
       for obj2 in networks[1:]:
           try: 
               obj3 = obj1 + obj2
               networks.remove(obj1)
               networks.remove(obj2)
               networks.insert(0, obj3)
               break
           except: continue
       else: raise Prob_Incompatibili("Probabilità in input errate")


def flat_tuple(to_flat:tuple[tuple[...]]) -> Generator[Any, None, None]:
    for val in to_flat:
        try: 
            _ = val[0]
            yield from flat_tuple(val)
        except: yield val

def cycle_list(vals:Iterable[Any]) -> Generator[list[Any], None, None]:
    it = list(vals)
    while True:
        yield it
        it = it[1:] + [it[0]]

class Generatore:
    def __init__(self, n_skus:int, l_list:int):
        """ le sku sono numeri da 1 a n
                l_list è la lunghezza delle liste
                    d'ordini da generare """
        # progressivo e sku
        self.ns:int = n_skus
        self.nl:int = l_list
        # numero di cluster (prodotti correlati positivamente)
        self.nc:int = 0
        # probabilità marginale 
        self.pr: m_prob = {}
        # probabilità condizionate di ogni cluster
        self.cpr: dict[int, c_prob] = {}
        # il network delle probabilità di ogni cluster
        self.nw: dict[int, Network] = {} 
        
        """ attributi che verranno valorizzati con il calcolo degli ordini """
        # ordini generati
        self.orders: dict[int, list[int]] = {}
        # sintesi sku e ordini nei quali figura
        self.sku_orders: dict[int, set[int]] = {}
        # altre coocorrenze interessanti che sono state trovate
        self.discovered: dict[int, dict[int|tuple, float]] = {} 
        
    @property
    def relevant_skus(self) -> tuple[int]:
        """ le sku per le quali sono state definite le probabilità """
        rs = set()
        for nw in self.nw.values():
            rs.update(nw.all_keys())
        return tuple(rs)

    def leader_product(self, c_idx:int)-> int:
        """ il prodotto radice
                nelle catene di dipendenze """
        return tuple(self.nw[c_idx].all_keys())[0]
    
    def sku_clusters(self, sku:int) -> tuple[int]:
        """ restituisce il numero di cluster
                in cui figura una sku """
        cl = tuple()
        for idx, nw in self.nw.items():
            if sku in tuple(nw.all_keys()): cl += (idx,)
        return cl  
    
    def add_prob(self, prob: Iterable[float]|m_prob, gen_prob:gen_function):
        """ aggiunge le probabilitò delle sku, per le sku 
             non significative, le probabilità vengono generate in maniera random """
        def normalize_to(n_val:float, values:Iterable[float]):
            tot = sum(values)
            return tuple(n_val*v/tot for v in values)
            
        if type(prob) != dict: prob = {(i + 1):pr for i,pr in enumerate(prob)}
        # la probabilità totale è pari alla lunghezza della lsita
        # qui calcoliamo il residuo
        res_prob = self.nl - sum(pr for pr in prob.values()) 
        nr = self.ns - len(prob)
        if res_prob != 0: 
            if res_prob < 0: raise Prob_Errate("La somma delle probabilità deve essere <= della lunghezza della lista")
            if res_prob/nr >= 0.9: raise  Prob_Errate("Le probabilità impostate sono troppo basse")
            # calcoliamo in maniera casuale le probabilità restanti
            # e le normalizziamo al valore coerente
            new_prob = normalize_to(res_prob, (tuple(gen_prob(res_prob, nr))))
            prob.update({i:pr for i, pr in zip((i for i in range(1, self.ns + 1) if i not in prob.keys()), new_prob)})
        self.pr = prob
        
    def add_cprob(self, *cprobs:c_prob):
        """ si aggiungono le probabilità
                condizionate di un cluster.
                   Devono essere definite in maniera coerente 
                       relativamente a un prodotto leader """
        for cprob in cprobs:
            self.nc += 1
            self.nw[self.nc] = gen_network(tuple(cprob.keys())) # crea il network
            self.cpr[self.nc] = cprob # registra la probabilità
            # aggiungiamo le probabilità di secondo livello
            self._sl_cprob(self.nc) 
    
    def _sl_cprob(self, cl_number:int):
        """ probabilità di secondo livello """
        
        def _add_2l_cprob(x:int):
            """ aggiunge la probabilità
                    condizionate di 
                       secondo livello """
            # gli ultimi due predecessori
            cpr = self.cpr[cl_number]
            pred_2, pred_1 = tuple(nw.all_predecessors(x))[-2:]
            # calcolo approssimato --> vedi file txt
            cpr[x,(pred_1,pred_2)] = self.pr[pred_1]*cpr[x,pred_1]/cpr[pred_1,pred_2]
            
        nw = self.nw[cl_number]
        lead = tuple(nw.all_keys())[0]
        succ = tuple(nw.all_successors(lead))[1:] # solo quelli di secondo livello!!!
        for sc in flat_tuple(succ): _add_2l_cprob(sc)
        
    def make_sub_set(self, sequence:tuple[int], n_list:int, start_val: int = 0) -> tuple[dict[int, list[int]]]:
        """ sequence è l'ordine con cui vengono considerati i cluster,
            n_list è il numero di liste che vanno create """
     
        def _place_sku(sku:int, orders_id:Iterable[int]):
            for id_or in orders_id:
                orders[id_or].append(sku) # aggiungiamo la sku alla lista
                if sku in sku_order.keys(): # le non rilevanti non ci sono
                    sku_order[sku].add(id_or) # registriamo il fatto che la lista ha la sku
                if len(orders[id_or]) >= self.nl: full_orders.add(id_or)
            n_sku[sku] -= len(orders_id)
            
        def _place_first_level(sku:int):
            """ piazza i prodotti leader, 
                    non è contemplato il fatto che un 
                        prodotto leader possa dipendere da un altro """
            open_orders = tuple(id_or for id_or in orders.keys() if id_or not in full_orders)
            n_open = len(open_orders)
            orders_id = open_orders if n_open <= n_sku[sku] else random.sample(open_orders, n_sku[sku])
            _place_sku(sku, orders_id)
            
        def _place_other_level(sku:int, last_cluster:bool = False):
            """ piazza le sku che hanno dipendenze di primo
                        e di secondo livello, 
                last cluster indica se stiamo considerando l'ultima 
                    classe di prodotti correlati """
            
            def _single_level(pred_1, to_place = None):
                """ considero solo dipendenze
                        singole es. C|A """
                if to_place is None: to_place = min(n_sku[sku], int(round(len(sku_order[pred_1])*pr_given_1, 0)))
                open_orders =  set(sku_order[pred_1]) # quelli in cui c'è il predecessore
                open_orders -= sku_order[sku] # meno quelli eventuali in cui è già presente
                open_orders -= full_orders
                n_open = len(open_orders) # numero di ordini
                orders_id = tuple(open_orders) if n_open <= to_place else random.sample(tuple(open_orders), to_place)
                _place_sku(sku, orders_id)
            
            def _double_level(pred_1, pred_2) -> int:
                """ piazza considerando dipendenze
                        a coppie es. C|A,B  
                           pred_1 è quello più vicino
                    restituisce le sku residue da piazzare """
                # ordini che hanno entrambi i predecessori e non sono pieni:
                open_orders =  sku_order[pred_1] & sku_order[pred_2]
                open_orders -= full_orders
                n_open = len(open_orders) # numero di ordini
                
                # sku da piazzare
                n_sku_now = min(n_sku[sku], int(round(n_open*pr_given_1_2, 0)))
                n_sku_next = min(n_sku[sku], int(round(len(sku_order[pred_1])*pr_given_1, 0))) # quelle che andrebbero messe dopo
                
                orders_id = tuple(open_orders) if n_open <= n_sku_now else random.sample(tuple(open_orders), n_sku_now)
                _place_sku(sku, orders_id)
    
                # ritorniamo quante ancora devono essere piazzate
                return max(0, n_sku_next - len(orders_id))
            
            predecessors = tuple(self.nw[c_idx].all_predecessors(sku))[-2:]
            pred_2, *pred_1 = predecessors # sono o uno o due
            # le probabilità
            pr = self.pr[sku]
        
            try:
                pred_1 = pred_1[0] # il primo e l'unico
                pr_given_1 = self.cpr[c_idx][sku, pred_1]
                pr_given_1_2 = self.cpr[c_idx][sku, (pred_1, pred_2)]
                # piazziamo in base a Pr(C|B,A) e vediamo quante ancora ne devono essere piazzate
                n_sku_remaining = _double_level(pred_1, pred_2)
        
            except IndexError: 
                pred_1 = pred_2
                pr_given_1 = self.cpr[c_idx][sku, pred_1]
                n_sku_remaining = None
            finally:
                _single_level(pred_1, n_sku_remaining)
                """ se non compare in altri cluster aggiungo le residue casualmente """
                if len(self.sku_clusters(sku)) == 1 or last_cluster: 
                    random_allocation(sku) 
            
        def random_allocation(sku):
            """ alloca le residue in maniera random """
        
            def allocate(pred_1:int = None):
                open_orders = set(id_or for id_or in orders.keys() if id_or not in full_orders)
                # quando piazziamo le non rilevanti non sono presenti in sku_order, darebbe errore
                if sku in sku_order.keys():
                    open_orders -= sku_order[sku] # meno quelli eventuali in cui è già presente
                    if pred_1 is not None:
                        open_orders -= sku_order[pred_1] # allocazione random, ma non dobbiamo metterli dove c'è il predecessore
               
                n_open = len(open_orders)
                orders_id = open_orders if n_open <= n_sku[sku] else random.sample(tuple(open_orders), n_sku[sku])
                _place_sku(sku, orders_id)
            
            try: pred_1 = tuple(self.nw[c_idx].all_predecessors(sku))[-1]
            except: pred_1 = None
            allocate(pred_1) 
            # se per i prodotti significativi è rimasto qualcosa, proviamo a piazzarli in maniera totalmente random
            if n_sku[sku] > 0 and pred_1: allocate(None)
               
                
        # le sku da piazzare
        n_sku = {sku:int(round(self.pr[sku]*n_list, 0)) for sku in range(1, self.ns + 1)}
        # liste effettive 
        n_list = math.ceil(sum(ns for ns in n_sku.values())/self.nl)
        # gli ordini
        orders = {idx:[] for idx in range(start_val, start_val + n_list)}
        # accoppiamento sku e indice ordine in cui figura (solo per le sku d'interesse)
        sku_order = {sku:set() for sku in self.relevant_skus}
        # liste piene 
        full_orders = set()
        tot_i = len(sequence)
        for i in range(tot_i):
            c_idx = sequence[i]
            leader_sku = self.leader_product(c_idx)
            _place_first_level(leader_sku)
            # consideriamo i successori, raggruppati per primo, secondo, terzo livello ecc.
            for other_sku in self.nw[c_idx].all_successors(leader_sku):
                # ordiniamo per probabilità crescente
                other_sku.sort(key = lambda sku: self.pr[sku], reverse = True)
                for sku in other_sku: 
                    _place_other_level(sku, last_cluster = (i == tot_i - 1))
        
        for remaining_sku in (sk for sk in n_sku.keys() if sk not in self.relevant_skus): 
            random_allocation(remaining_sku)
        
        return orders, sku_order
    
    def __iter__(self) -> Generator[list[int], None, None]:
        """ restituisce un ordine alla volta """
        for Ord in self.orders.values():
            if len(Ord) == self.nl: yield Ord

    def gen_det(self, n_batch:int = 1000, n_rep = 5):
        """ genera tutti gli ordini procedendo 
                con generazioni successive da n_batch 
                    ordini ciascuna, ripetuta n_rep 
                        volte per ogni cluster """
        self.orders = {}
        self.sku_orders = {sku:set() for sku in self.relevant_skus}
        
        for i, sequence in zip(range(n_rep*self.nc), cycle_list(tuple(self.nw.keys()))):
            print(f'making sub_set {i}-th')
            order, sku_order = self.make_sub_set(sequence = sequence, n_list = n_batch, start_val = i*n_batch)
            self.orders.update(order)
            for sku, new_set in sku_order.items():
                self.sku_orders[sku] |= new_set
    
    def gen_rnd(self, n_list:int = 5_000, record_all:bool = False, print_every:int = 1_000):
        """ genera gli ordini usando un 
                approccio generativo di tipo
                    stocastico """
    
        def normalize_prob(probs, sku_taboo:list) -> dict:
            tot_pr = sum(pr for sk, pr in probs.items() if sk not in sku_taboo)
            return {sk:pr/tot_pr for sk, pr in probs.items() if sk not in sku_taboo}
        
        def update_sku_and_successors(sku, probs) -> dict:
            probs[sku] = 0
            # se ha dei successori ne modifichiamo la prob con quella condizionata
            for cl in self.sku_clusters(sku): # tutti i cluster ai quali appartiene la sku
                successors = self.nw[cl].fl_successors(sku) # gli eventuali successori per i quali c'è prob. condizionata
                if successors is None: successors = tuple()
                for sc in successors: 
                    probs[sc] = self.cpr[cl][sc, sku] # la probabilità condizionata
            return probs
        
        def extract_one(or_id, probs) -> int:
            sku = random.choices(tuple(probs.keys()), tuple(probs.values()))[0]
            self.orders[or_id].append(sku)  
            if record_all or (sku in relevant_skus):
                self.sku_orders.setdefault(sku, set()).add(or_id)
            return sku
        
        self.orders = {}
        self.sku_orders = {sku:set() for sku in self.relevant_skus}
        relevant_skus = self.relevant_skus
        i_print = print_every - 1
        
        for or_id in range(n_list): # per ogni lista che vogliamo generare
            if or_id == i_print: 
                print(f'Generati altri {print_every} ordini')
                i_print += print_every
            
            original_sku_probs = self.pr.copy()
            self.orders[or_id] = []
            for idx in range(self.nl): # per ogni elemento che dobbiamo mettere nell'ordine
                # normalizziamo le probabilità
                nr_probs = normalize_prob(original_sku_probs, self.orders[or_id])
                sku = extract_one(or_id, nr_probs)
                # mettiamo a zero la probabilità dell'sku estratta
                # sostituiamo la prob.marginale con quella condizionata 
                # di tutti gli eventuali successori
                original_sku_probs = update_sku_and_successors(sku, original_sku_probs)
                # e dopo normalizziamo
                
    def ideal_prob(self, sku:int)-> float:
        """ restituisce la probabilità impostata"""
        return self.pr[sku]
       
    def emp_prob(self, sku:int) -> float:
        """ restituisce la probabilità empirica 
                calcolata sul campione generato """
        if sku in self.relevant_skus: 
            return round(len(self.sku_orders[sku])/len(self.orders),4)
        else:
            n_sku, n_tot = 0, 0
            for order in self:
                n_tot += 1
                if sku in order: n_sku += 1
            try: return round(n_sku/n_tot, 4)
            except ZeroDivisionError: return 0
            
    def ideal_cprob(self, sku:int, given_sku) -> float:
        """ restituisce la probabilità condizionata
                di primo livello, impostata"""
        for cl in self.sku_clusters(sku):
            try: return self.cpr[cl][sku, given_sku]
            except:continue
        else: return math.nan
    
    def emp_cprob(self, sku:int, given_sku) -> float:
        """ restituisce la probabilità condizionata
                di primo livello, impostata"""
        if sku in self.relevant_skus and given_sku in self.relevant_skus: 
            both_sku = self.sku_orders[sku] & self.sku_orders[given_sku]
            return round(len(both_sku)/len(self.sku_orders[given_sku]),4)
        else:
            n_other, n_both = 0, 0
            for order in self:
                if given_sku in order: 
                    n_other += 1
                    if sku in order: n_both += 1  
            try: return round(n_both/n_other, 4)
            except ZeroDivisionError: return 0
    
    def find_new(self, skus:Optional[Iterable[Sk]] = None):
        """ trova nuove relazioni """
        self.discovered = {}
        if skus is None: skus = self.pr.keys()
        for sku in skus:
            p_sku = self.emp_prob(sku)
            for oth_sku in self.pr.keys():
                if sku != oth_sku:
                    cp_sku = self.emp_cprob(sku, oth_sku)
                    if cp_sku > p_sku: self.discovered.setdefault(sku, {sku:p_sku}).update({(sku, oth_sku):cp_sku})
    
    def show_new(self, round_to:int = 2) -> Generator[dict, None, None]:
        def convert(key:int|tuple, val: float) -> str:
            try: 
                x1, x2 = key
                x = str(x1) + "|" + str(x2)
            except: x = str(key)
            x = 'P' + '(' + x + ')'
            return x + "=" + str(round(val, round_to))
        
        for d_prob in self.discovered.values():
            out = [convert(key, val) for key, val in d_prob.items()]
            #x = {''.join(('P', str(key))):val for key, val in d_prob.items()}
            yield ', '.join(out)
            
    def __repr__(self) -> str:
        if self.orders == {}: return "Empty generator"
        out = ""
        
        for sku in self.relevant_skus:
            """ probabilità di base"""
            ideal_prob = round(self.ideal_prob(sku), 4)
            emp_prob = round(self.emp_prob(sku), 4)
            delta = round(abs(ideal_prob - emp_prob)/ideal_prob, 4)
            out += f'P[{sku}]:{ideal_prob}, eP[{sku}]:{emp_prob}, dp:{delta}\n'
            for cl in self.sku_clusters(sku): # tutti i cluster ai quali appartiene la sku
                """ probabilità condizionate """
                predecessors = self.nw[cl].all_predecessors(sku)
                try:
                    pred = predecessors[-1]
                    ideal_prob = round(self.cpr[cl][sku, pred], 4) 
                    emp_prob = round(self.emp_cprob(sku, pred), 4)
                    delta = round(abs(ideal_prob - emp_prob)/ideal_prob, 4)
                    out += f'P[{sku}|{pred}]:{ideal_prob}, eP[{sku}|{pred}]:{emp_prob}, dcp:{delta}\n'
                except: out += ""
        return out



        
        
random.seed(1)

""" 
THE PROPOSED EXAMPLE
20 SKUs, 2 independent clusters each made of 5 SKUs, 
order list with L = 5  
"""
# marginal probabilities 
prob = {1:0.75,  # first cluster
            2:0.5, 
                3:0.25, 
                4:0.2, 
            5: 0.3,
        6:0.6, # second cluster
            7: 0.4, 
            8: 0.3,
            9: 0.2,
            10: 0.1,
        # free SKUs
        11:0.3, 12:0.25, 13:0.2, 14:0.2, 15:0.15, 16:0.12, 17:0.10, 18:0.05, 19:0.02, 20:0.01}

# conditional probabilities cluster 1
cpr_1 = {(2, 1):0.65, (5, 1):0.4, 
         (3, 2):0.35, (4, 2):0.3}

# conditional probabilities cluster 2
cpr_2 = {(7,6):0.5, (8,6): 0.35, (9,6):0.25, (10,6):0.15}

g = Generatore(20, 5)

g.add_prob(prob, lambda x: None) # there is no need to add "random" probabilities to the free SKUS, so lambda x: None
g.add_cprob(cpr_1)
g.add_cprob(cpr_2)

g.gen_det(n_batch  = 1000, n_rep = 10) # constructive algorithm N = 10, B = 1000

"""
# use this to generate the order list using the stochastic approach 
g.gen_rnd(10000)
"""

g.find_new() # to show the new discovered probabilities
for x in g.show_new():
    print(x)

"""

"""

