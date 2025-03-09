import networkx as nx
from copy import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from collections import Counter
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from torch.autograd import Variable
import collections
from time import time


EMBEDDING_DIM = 128
PRINT_EVERY = 100
EPOCHS = 100
EPOCHS4Struct = 0
BATCH_SIZE = 1000
BATCHS=100
N_SAMPLES = 20
LR=0.005
LR4Struct=0.005
device = 'cuda'


class WLAlign:
    def __init__(self, config):
        self.ratio = config['train_ratio']

    def run(self, G1, G2, alignment):
        ratio = self.ratio
        anchor = list(alignment.keys())[:int(len(alignment) * 0.01 * ratio)]
        anchor = {x: alignment[x] for x in anchor}
        nx_G, anchor_list, recover_dict = build_graph(G1, G2, anchor, alignment)
        # print(nx_G.nodes())
        ouput_filename_networkx = "x.txt"
        ouput_filename_networky = "y.txt"

        G_anchor = get_graph_anchor(nx_G)

        network = networkC(G_anchor, nx_G, True, 1, 1, anchor_list)
        network_all = networkC(nx_G, G_anchor, True, 1, 1, anchor_list)

        print(network.num_nodes())
        print('length of all nodes:', len(list(network.G.nodes())))
        print('length of all nodes:', len(list(network.G_all.nodes())))

        print(device)
        all_candate = []
        un_candate_pair_list = []

        start_time = time()

        # Initialize training function
        # trainerI = Trainer(EMBEDDING_DIM, 0.0001, EPOCHS4Struct, BATCHS, BATCH_SIZE, N_SAMPLES, network_all, device)
        trainerT = Trainer_T(EMBEDDING_DIM, LR4Struct, 8000, BATCHS, BATCH_SIZE, N_SAMPLES, network_all, device)

        y = 1
        num_mark = 0
        acc_list = []

        # Training process
        while (True):
            print(len(network.vocab2int))
            # Initialize aggregate function and node label
            agg_model = AggregateLabel(len(network.vocab2int), len(network.vocab2int), device).to(device)
            onehot_label = get_graph_label(network, len(network.vocab2int), device).to(device)
            # if y==1:
            #     pre_layer_label=torch.zeros(len(network_all.vocab2int),EMBEDDING_DIM).to(device)
            # else:
            #     pre_layer_label=embedding_I.detach()
            # print(pre_layer_label.shape,'============================================')

            # label aggregate=================================================================
            layers_label = agg_model(onehot_label.weight.data, network.adj_real.to(device), 1)

            # Judge the colored node pair by similarity=================================================================
            candate_pair, un_candate_pair, result_old_list = get_candate_pair(layers_label[0], network)
            candate_pair_self, un_candate_pair_self, result_old_list_self = get_candate_pair_self(layers_label[0],
                                                                                                  network)

            y += 1

            # Judge whether convergence
            if not network.is_convergence():
                network_tmp = copy(network)
                network_tmp.vocab2int = copy(network.vocab2int)
                network_tmp.int2vocab = copy(network.int2vocab)
                candate_pair = list(set(candate_pair))
                all_candate.extend(candate_pair)
                all_candate = list(set(all_candate))

                # embedding_I, network_all = trainerI.train_anchor(network_tmp, all_candate, layers_label[0],pre_layer_label, nx_G,network.mark_pair,0)
                #
                embedding_T, network_all = trainerT.train_anchor(network_all, all_candate, un_candate_pair_list,
                                                                 layers_label[0], network.mark_pair, 50)
                # If there is no convergence, remap the label
                network.remark_node(candate_pair)
                network.reset_anchor(candate_pair, get_graph_anchorBy_mark, len(candate_pair))

                print(network.is_mark_finished())
                network.reset_edges(candate_pair, get_graph_anchorBy_mark, candate_pair_self)
                num_mark = network.num_mark()
                continue
            else:
                num_mark = network.num_mark()
                candate_pair = list(set(candate_pair))
                all_candate = remark(all_candate, network)
                all_candate.extend(candate_pair)

                embedding_T, network_all = trainerT.train_anchor(network_all, all_candate, un_candate_pair_list,
                                                                 layers_label[0], network.mark_pair,
                                                                 )
                print(embedding_T.shape)
                writeFile(embedding_T, network_all, ouput_filename_networkx + ".number_T", "_foursquare")
                writeFile(embedding_T, network_all, ouput_filename_networky + ".number_T", "_twitter")

                break
        print('finished!')
        #
        f_networkx = open(ouput_filename_networkx + ".number_T")
        f_networky = open(ouput_filename_networky + ".number_T")
        network_x=dict()
        network_y=dict()
        line=f_networkx.readline()
        while line:
            listx = []
            line = line.replace("|\n", "")
            sp = line.split(" ", 1)
            vector_array = sp[1].split("|", 10000)
            for x in vector_array:
                listx.append(x)
            listx = list(map(float, listx))
            vector = change2tensor(listx)
            i = 0
            if sp[0].split('_')[0] in anchor_list:
                line = f_networkx.readline()
                continue
            network_x[sp[0]] = vector
            line = f_networkx.readline()
        f_networkx.close()

        line = f_networky.readline()
        while line:
            listy = []
            line = line.replace("|\n", "")
            sp = line.split(" ", 1)
            vector_array = sp[1].split("|", 10000)
            for y in vector_array:
                listy.append(y)
            listy = list(map(float, listy))
            vector = change2tensor(listy)
            if sp[0].split('_')[0] in anchor_list:
                line = f_networky.readline()
                continue
            network_y[sp[0]] = vector
            line = f_networky.readline()
        f_networky.close()

        emb1 = None
        emb2 = None
        for x in recover_dict:
            name = str(x) + "_twitter"
            if emb1 is None:
                emb1 = network_y[name]
                emb1 = emb1.unsqueeze(dim=0)
            else:
                emb1 = torch.cat((emb1, network_y[name].unsqueeze(dim=0)), dim=0)

        for x in range(len(network_x)):
            name = str(x) + "_foursquare"
            if emb2 is None:
                emb2 = network_x[name]
                emb2 = emb2.unsqueeze(dim=0)
            else:
                emb2 = torch.cat((emb2, network_x[name].unsqueeze(dim=0)), dim=0)

        S = torch.mm(emb1, emb2.t())
        test_list = list(map(int, alignment.keys()))[int(len(alignment) * 0.01 * ratio):]
        S_test = S[test_list]
        rank_matrix = (-1 * S_test).argsort().tolist()
        align_links = [[], []]

        align_links[0] = test_list
        align_links[1] = S_test.argmax(dim=1)

        end_time = time()

        return align_links, rank_matrix, end_time - start_time


def getpAtN(network_x, network_y, test_list):
    pAtN_x_map = dict()
    al = 0
    print('-------------------------')
    for array_edge in test_list:
        target = 0
        array_edge = str(array_edge)
        y = array_edge + "_twitter"
        x = array_edge + "_foursquare"
        if x in network_x.keys() and y in network_y.keys():
            sam = torch.cosine_similarity(network_x[x], network_y[y], dim=0)
            for key,value in network_y.items():
                if key==y:
                    continue
                if (torch.cosine_similarity(network_x[x], value, dim=0).double() >= sam.double()):
                    target += 1
        else:
            print('c')
            al -= 1

        pAtN_x_map[array_edge]=target
        al += 1

    return pAtN_x_map, al



class networkC:
    def __init__(self, nx_G,G_all, is_directed, p, q,anchor_list):
        self.G = nx_G
        self.G_all=G_all
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.anchor_list=anchor_list
        self.max_mark = 0
        self.mark_pair = self.get_mark_pair()
        self.vocab_prepare()
        self.edge_prepare()
        self.node_mark=self.dict_mark()
        self.converge_distribution=[]
        self.converge_distribution_pre=[]

    def get_G_anchor(self,G_anchor):
        self.G_anchor=G_anchor

    def get_mark_pair(self):
        mark_pair=dict()
        for node in self.G.nodes():
            if node.endswith('_anchor'):
                mark_pair[node]=self.max_mark
                self.max_mark+=1
        return mark_pair

    def is_convergence(self):
        converge_dict=dict()
        for key,value in self.mark_pair.items():
            if value not in converge_dict.keys():
                converge_dict[value]=1
            else:
                converge_dict[value]+=1
        converge_distribution=sorted(list(converge_dict.values()))

        if len(self.converge_distribution)!=len(converge_distribution) and len(self.converge_distribution_pre)!=len(converge_distribution):
            self.converge_distribution_pre = self.converge_distribution
            self.converge_distribution=converge_distribution
            return False
        else:
            if len(self.converge_distribution)==len(converge_distribution):
                for a,b in zip(self.converge_distribution,converge_distribution):
                    if a!=b:
                        self.converge_distribution_pre = self.converge_distribution
                        self.converge_distribution = converge_distribution
                        return False
            else:
                for a,b in zip(self.converge_distribution_pre,converge_distribution):
                    if a!=b:
                        self.converge_distribution_pre = self.converge_distribution
                        self.converge_distribution = converge_distribution
                        return False
        return True

    def dict_mark(self):
        node_mark=dict()
        for node in self.G_all.nodes():
            if node.endswith('_anchor'):
                node_mark[node]=True
            else:
                node_mark[node]=False
        return node_mark

    def remark_node(self,result_dict):
        for pair in result_dict:
            pairs=pair.split('-')
            self.node_mark[pairs[0]]=True
            self.node_mark[pairs[1]]=True

    def is_mark_finished(self):
        for key,value in self.node_mark.items():
            if value==False:
                return value
        return True
    def num_mark(self):
        num=0
        for key,value in self.node_mark.items():
            if value==True:
                num+=1

        return num

    def hash_mark(self):
        hash_dict=dict()
        for key,value in self.mark_pair.items():
            if value not in hash_dict.keys():
                hash_dict[value]=[key]
            else:
                hash_dict[value].append(key)
        mark_pair=dict()
        for i,value in enumerate(hash_dict.values()):
            for v in value:
                mark_pair[v]=i
        self.mark_pair=mark_pair

    def reset_anchor(self, result_list, get_graph_anchor, is_change):
        for node in result_list:
            nodes = node.split('-')
            if nodes[0] in self.mark_pair.keys() and nodes[1] not in self.mark_pair.keys():
                self.mark_pair[nodes[1]] = self.mark_pair[nodes[0]]
            elif nodes[1] in self.mark_pair.keys() and nodes[0] not in self.mark_pair.keys():
                self.mark_pair[nodes[0]] = self.mark_pair[nodes[1]]
            elif nodes[0] in self.mark_pair.keys() and nodes[1] in self.mark_pair.keys():
                if self.mark_pair[nodes[0]] == self.mark_pair[nodes[1]]:
                    pass
                else:
                    self.mark_pair[nodes[0]] = self.max_mark
                    self.mark_pair[nodes[1]] = self.max_mark
                    self.max_mark += 1
            else:
                self.mark_pair[nodes[0]] = self.max_mark
                self.mark_pair[nodes[1]] = self.max_mark
                self.max_mark += 1
        # print(self.mark_pair)
        self.hash_mark()
        self.G = get_graph_anchor(self.G, self.mark_pair, self.num_mark())

        self.vocab_prepare()
        self.edge_prepare()

    def reset_edges(self,result_dict,get_graph_anchor,candate_pair_self):
        # for node in result_dict:
        #     nodes = node.split('-')
        #     if nodes[0] in self.mark_pair.keys() and nodes[1] not in self.mark_pair.keys():
        #         self.mark_pair[nodes[1]] = self.mark_pair[nodes[0]]
        #     elif nodes[1] in self.mark_pair.keys() and nodes[0] not in self.mark_pair.keys():
        #         self.mark_pair[nodes[0]] = self.mark_pair[nodes[1]]
        #     elif nodes[0] in self.mark_pair.keys() and nodes[1] in self.mark_pair.keys():
        #         pass
        #     else:
        #         self.mark_pair[nodes[0]] = self.max_mark
        #         self.mark_pair[nodes[1]] = self.max_mark
        #         self.max_mark += 1
        for node in candate_pair_self:
            nodes = node.split('-')
            if nodes[0] in self.mark_pair.keys() and nodes[1] not in self.mark_pair.keys():
                self.mark_pair[nodes[1]] = self.mark_pair[nodes[0]]
            elif nodes[1] in self.mark_pair.keys() and nodes[0] not in self.mark_pair.keys():
                self.mark_pair[nodes[0]] = self.mark_pair[nodes[1]]
            else:
                pass
        self.G = get_graph_anchor(self.G_all, self.mark_pair, self.num_mark())

        self.vocab_prepare()
        self.edge_prepare()

    def reset_anchor2(self,result_dict,get_graph_anchor):
        for node in result_dict.keys():
            nodes=node.split('-')
            self.G.remove_node(nodes[0])
            self.G.remove_node(nodes[1])

        self.G = get_graph_anchor(self.G_all,self.mark_pair)

        self.vocab_prepare()
        self.edge_prepare()

    def reset_graph(self,graph_all,init_emb,device):
        print('reset label')
        init_emb = F.normalize(init_emb)
        self.vocab_Re_prepare(graph_all,init_emb,device)
        self.edge_prepare()
        return self.init_emb
    def get_same_label(self,all_candate,mark_pair):
        left=[]
        right=[]
        for node in all_candate:
            nodes = node.split('-')
            left_node=self.vocab2int[nodes[0]]
            right_node=self.vocab2int[nodes[1]]


            left.append(left_node)
            right.append(right_node)
        return left,right

    def set_closure_mark(self,all_candate):
        all_nodes = list(self.G.nodes())
        graph_x = nx.Graph()
        for edge in all_candate:
            nodes = edge.split('-')
            graph_x.add_edge(nodes[0], nodes[1])

        closure_list = []
        closure_noded_all = []
        nodes_left = []
        for closure in nx.connected_components(graph_x):
            closure_list.append(list(closure))
            closure_noded_all.extend(list(closure))
        for node in all_nodes:
            if node not in closure_noded_all:
                nodes_left.append(node)
        closure_list.append(nodes_left)

        closure_dict=dict()
        for i in range(len(closure_list)):
            for closure in closure_list[i]:
                closure_dict[closure]=i
        self.closure_list=closure_list
        self.closure_dict=closure_dict
        self.num_label=len(closure_list)

    def vocab_Re_prepare(self,graph_all,init_emb,device):
        self.G=graph_all
        bias=len(list(self.G.nodes()))-init_emb.shape[1]
        emb_bias=torch.zeros(init_emb.shape[0],bias).to(device)
        init_emb = torch.cat((init_emb,emb_bias),dim=1)
        eye_emb=torch.eye(len(list(self.G.nodes()))).to(device)
        emb_list=[]
        for tensor in init_emb:
            emb_list.append(tensor)
        num_node = len(list(self.vocab2int.keys()))
        for node in self.G.nodes():
            if node not in self.vocab2int.keys():
                self.vocab2int[node]=num_node
                self.int2vocab[num_node]=node
                emb_list.append(eye_emb[num_node])
                num_node+=1
        self.init_emb=torch.stack(emb_list,dim=0)
        print(len(list(self.vocab2int.keys())))
        print(self.init_emb.shape)
    def vocab_prepare(self):
        self.vocab2int = {w: c for c, w in enumerate(self.G.nodes())}
        self.int2vocab = {c: w for c, w in enumerate(self.G.nodes())}
        F_intlist=[]
        T_intlist=[]
        mark_F_intlist=[]
        mark_T_intlist = []
        for key,value in self.vocab2int.items():
            if key not in self.mark_pair.keys():
                if key.endswith('_foursquare'):
                    F_intlist.append(value)
                if key.endswith('_twitter'):
                    T_intlist.append(value)
            else:
                if key.endswith('_foursquare'):
                    mark_F_intlist.append(value)
                if key.endswith('_twitter'):
                    mark_T_intlist.append(value)
        all_list=list(self.vocab2int.values())
        intlist_p=[]
        for i in all_list:
            if i not in F_intlist+T_intlist:
                intlist_p.append(i)
        self.all_intlist_p=intlist_p
        self.all_intlist_n=F_intlist+T_intlist

        self.F_intlist=torch.LongTensor(F_intlist)
        self.T_intlist=torch.LongTensor(T_intlist)
        self.mark_F_intlist=torch.LongTensor(F_intlist+mark_F_intlist)
        self.mark_T_intlist = torch.LongTensor(T_intlist + mark_T_intlist)
        self.only_mark_F_intlist = torch.LongTensor(mark_F_intlist)
        self.only_mark_T_intlist = torch.LongTensor(mark_T_intlist)
        #self.all_intlist_n=torch.LongTensor(F_intlist+T_intlist)

    def get_intlist_p(self,length):
        p_list=[]
        for i in range(length):
            p_list.append(random.choices(self.all_intlist_p)[0])
        return torch.LongTensor(p_list)

    def get_intlist_n(self,length):
        n_list=[]
        for i in range(length):
            n_list.append(random.choices(self.all_intlist_n)[0])
        return torch.LongTensor(n_list)

    def edge_prepare(self):
        source_list = []
        target_list = []
        adj_real = torch.eye(len(self.vocab2int), len(self.vocab2int))
        for edge in self.G.edges():
            source_list.append(self.vocab2int[edge[0]])
            target_list.append(self.vocab2int[edge[1]])
            adj_real[self.vocab2int[edge[0]]][self.vocab2int[edge[1]]] = float(self.G[edge[0]][edge[1]]['weight'])
            adj_real[self.vocab2int[edge[1]]][self.vocab2int[edge[0]]] = float(self.G[edge[0]][edge[1]]['weight'])
        adjacency_matrix = [source_list, target_list]
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.long)
        self.adjacency_matrix = adjacency_matrix
        self.adj_real = adj_real

    def num_nodes(self):
        return len(self.vocab2int)

    def get_embedding_index(self,embedding):
        self.embedding=embedding

        embedding_dict=dict()
        vocab2embedding=dict()
        for key in self.vocab2int.keys():
            embedding_dict[key]=torch.tensor([self.vocab2int[key]])
            vocab2embedding[key]=self.embedding[embedding_dict[key]]

        self.embedding_index=embedding_dict
        self.vocab2embedding=vocab2embedding

    def embedding_rebuild(self):
        embedding_list=[]
        i=0
        self.vocab2int=dict()
        self.int2vocab=dict()
        for key,value in self.vocab2embedding.items():
            embedding_list.append(value)
            self.vocab2int[key]=i
            self.int2vocab[i]=key
            i+=1
        self.embedding=torch.stack(embedding_list, 0)

    def get_emb_dict_by_1file(self,file,anchor_list,pix):
        f_network = open(file)

        network_dict = dict()

        line = f_network.readline()
        i=0
        while line:
            listx = []
            line = line.replace("|\n", "")
            sp = line.split(" ", 1)
            vector_array = sp[1].split("|", 127)
            for x in vector_array:
                listx.append(x)
            listx = list(map(float, listx))
            vector = change2tensor(listx)
            if sp[0].replace(pix,'') in anchor_list:
                sp[0]=sp[0].replace(pix,"_anchor")
                i+=1
            network_dict[sp[0]] = vector
            line = f_network.readline()
        f_network.close()

        self.vocab2embedding=network_dict
        self.embedding_rebuild()
        return i

    def get_emb_dict_by_2file(self,file1,file2,anchor_list):
        f_networkx = open(file1)
        f_networky = open(file2)

        network_dict = dict()

        line = f_networkx.readline()
        while line:
            listx = []
            line = line.replace("|\n", "")
            sp = line.split(" ", 1)
            vector_array = sp[1].split("|", 127)
            for x in vector_array:
                listx.append(x)
            listx = list(map(float, listx))
            vector = change2tensor(listx)
            if sp[0].replace("_foursquare",'') in anchor_list:
                sp[0]=sp[0].replace("_foursquare","_anchor")
            network_dict[sp[0]] = vector
            line = f_networkx.readline()
        f_networkx.close()


        line = f_networky.readline()
        while line:
            listy = []
            line = line.replace("|\n", "")
            sp = line.split(" ", 1)
            vector_array = sp[1].split("|", 127)
            for y in vector_array:
                listy.append(y)
            listy = list(map(float, listy))
            vector = change2tensor(listy)
            if sp[0].replace("_twitter",'') in anchor_list:
                sp[0]=sp[0].replace("_twitter","_anchor")
            network_dict[sp[0]] = vector
            line = f_networky.readline()
        f_networky.close()
        self.vocab2embedding=network_dict
        self.embedding_rebuild()

    def get_anchor_index(self,anchor_list):
        anchor_index_list=[]
        for anchor in anchor_list:
            anchor+='_anchor'
            index=self.vocab2int[anchor]
            anchor_index_list.append(index)
        self.anchor_index=torch.tensor(anchor_index_list)

        return self.anchor_index


    def get_training_set(self,batch_size):
        source = []
        target = []
        edges_list = []
        mark_list = []
        for i in range(10):
            edges = random.sample(list(self.G_anchor.edges()), batch_size)
            edges_list += edges
        # edges=[random.choice(list(self.G.edges())) for _ in range(batch_size)]
        for edge in edges_list:
            mark = 0
            x = edge[0]
            y = edge[1]
            if edge[0].endswith("_anchor"):
                x = random.sample(list(self.G_anchor.successors(edge[0])), 1)[0]
                mark += 1
            if edge[1].endswith("_anchor"):
                y = random.sample(list(self.G_anchor.predecessors(edge[1])), 1)[0]
                mark += 2

            mark_list.append(mark)
            source.append(x)
            target.append(y)
            if mark != 0:
                source.append(edge[0])
                target.append(edge[1])

        return source, target,mark_list

    def get_training_set2(self,batch_size):
        source = []
        target = []
        edges_list = []
        mark_list = []
        for i in range(10):
            edges = random.sample(list(self.G.edges()), batch_size)
            edges_list += edges
        # edges=[random.choice(list(self.G.edges())) for _ in range(batch_size)]
        for edge in edges_list:
            mark = 0
            x = edge[0]
            y = edge[1]

            source.append(x)
            target.append(y)

        return source, target,mark_list

    def get_training_set_by_anchor(self,batch_size):
        source = []
        target = []
        edges_list = []
        mark_list = []
        for i in range(10):
            edges = random.sample(list(self.G_anchor.edges()), batch_size)
            edges_list += edges
            edges = random.sample(list(self.G_B.edges()), batch_size)
            edges_list += edges
        # edges=[random.choice(list(self.G.edges())) for _ in range(batch_size)]
        for edge in edges_list:
            x = edge[0]
            y = edge[1]
            source.append(x)
            target.append(y)

        return source, target,mark_list

    def get_all_freq(self):
        walk = []
        for edge in self.G.edges():
            walk.append(edge[0])
            walk.append(edge[1])
        return walk

        return walk
    def get_all_freqX(self,source,target):
        walk=[]
        for x,y in zip(source,target):
            walk.append(x)
            walk.append(y)
        return walk



class Trainer_T:
    def __init__(self,EMBEDDING_DIM,LR,EPOCHS,BATCHS,BATCH_SIZE,N_SAMPLES,network,device,isanchor=False):
        self.Embedding_dim=EMBEDDING_DIM
        self.lr=LR
        self.Epochs=EPOCHS
        self.Batchs=BATCHS
        self.Batch_size=BATCH_SIZE
        self.n_samples=N_SAMPLES
        self.isanchor=isanchor
        self.model=EmbeddingModel(len(network.vocab2int), EMBEDDING_DIM).to(device)

    def train_anchor(self,network,all_candate, all_closure,layer_label,mark_pair,epoches=None):
        # device = 'cuda'
        # network.reset_graph(all_candate)
        acc_list=[]
        # network.set_closure_mark(all_candate, all_closure)
        # prepare for the cosine loss=====================================================================
        int_left, int_right = network.get_same_label(all_candate, mark_pair)
        mark_tensor = torch.LongTensor([1] * len(int_left)).to(device)
        mark_tensor_noise = torch.LongTensor([0] * len(int_left)).to(device)
        int_left = torch.LongTensor(int_left).to(device)
        int_right = torch.LongTensor(int_right).to(device)

        all_nodes = torch.LongTensor([network.vocab2int[i] for i in mark_pair.keys()]).to(device)
        all_labels = torch.LongTensor([mark_pair[node] for node in mark_pair.keys()]).to(device)
        #================================================================

        G_anchor = get_graph_anchor(network.G)
        network.get_G_anchor(G_anchor)
        node_noise = get_graph_noise(network.G, G_anchor)
        EMBEDDING_DIM, LR, EPOCHS, BATCHS, BATCH_SIZE, N_SAMPLES\
            =self.Embedding_dim,self.lr,self.Epochs,self.Batchs,self.Batch_size,self.n_samples

        if epoches:
            EPOCHS=epoches

        print('network length:', len(network.vocab2int))
        model = self.model
        criterion = NegativeSamplingLoss()
        cos = nn.CosineEmbeddingLoss(margin=0)

        # =======================================================================================
        noise_dist=self.noise_get(network.get_all_freq(),network).to(device)
        noise_dist2 = self.noise_get(node_noise,network).to(device)
        if len(noise_dist2)==0:
            noise_dist2=noise_dist
        # =======================================================================================

        steps = 0
        sqrs = []
        vs = []
        for param in model.parameters():
            sqrs.append(torch.zeros_like(param.data))
            vs.append(torch.zeros_like(param.data))
        for e in range(EPOCHS):

            i = 0
            if self.isanchor:
                sourcex, targetx, mark_list = network.get_training_set_by_anchor(BATCHS)
            else:
                sourcex, targetx, mark_list = network.get_training_set(BATCHS)
            for source, target in get_batch(sourcex, targetx, BATCH_SIZE):
                steps += 1
                int_source = [network.vocab2int[w] for w in source]
                int_target = [network.vocab2int[w] for w in target]
                inputs, targets = torch.LongTensor(int_source).to(device), torch.LongTensor(int_target).to(device)

                target_input_vectors = model.forward_input(targets)
                source_output_vectors = model.forward_output(inputs)
                self_in_vectors = model.forward_self(inputs)
                self_out_vectors = model.forward_self(targets)

                self_left=model.forward_self(int_left)
                self_right=model.forward_self(int_right)


                self_all_nodes=model.forward_self(all_nodes)

                size, _ = target_input_vectors.shape
                noise_vectors_self, noise_vectors_input, noise_vectors_output = model.forward_noise(size, N_SAMPLES,
                                                                                                    device, noise_dist)

                noise_vectors_self2, noise_vectors_input2, noise_vectors_output2 = model.forward_noise2(size, N_SAMPLES,
                                                                                                        device,
                                                                                                        noise_dist2)

                loss = criterion(self_in_vectors, self_out_vectors, target_input_vectors, source_output_vectors,
                                 noise_vectors_self, noise_vectors_input, noise_vectors_output,
                                 noise_vectors_self2, noise_vectors_input2, noise_vectors_output2)
                loss += cos(self_left, self_right, mark_tensor)

                model.zero_grad()
                loss.backward()
                adam(model.parameters(), vs, sqrs, LR, steps)

                i += 1

            progress(e / EPOCHS * 100, loss)
        print(acc_list)
        return model.self_embed.weight.data,network


    def noise_get(self,noise_list,network):
        all_walks = noise_list
        int_all_words = [network.vocab2int[w] for w in all_walks]

        int_word_counts = Counter(int_all_words)
        total_count = len(int_all_words)
        word_freqs = {w: c / (total_count) for w, c in int_word_counts.items()}

        word_freqs = np.array(list(word_freqs.values()))
        unigram_dist = word_freqs / word_freqs.sum()
        noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
        return noise_dist



def adam(parameters, vs, sqrs, lr, t, beta1=0.9, beta2=0.999):
    eps = 1e-8
    for param, v, sqr in zip(parameters, vs, sqrs):
        v[:] = beta1 * v + (1 - beta1) * param.grad.data
        sqr[:] = beta2 * sqr + (1 - beta2) * param.grad.data ** 2
        v_hat = v / (1 - beta1 ** t)
        s_hat = sqr / (1 - beta2 ** t)
        param.data = param.data - lr * v_hat / torch.sqrt(s_hat + eps)


def Other_label(labels,num_classes):
    index=torch.randint(num_classes, (labels.shape[0],)).to(labels.device)
    other_labels=labels+index
    other_labels[other_labels >= num_classes]=other_labels[other_labels >= num_classes]-num_classes
    return other_labels


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,self_in_vectors,self_out_vectors, target_input_vectors, source_output_vectors,
                noise_vectors_self,noise_vectors_input,noise_vectors_output,
                noise_vectors_self2, noise_vectors_input2, noise_vectors_output2):
        BATCH_SIZE, embed_size = target_input_vectors.shape

        self_in_vectors_T=self_in_vectors.view(BATCH_SIZE,embed_size,1)
        input_vectors_I=target_input_vectors.view(BATCH_SIZE,1,embed_size)
        output_vectors_I=source_output_vectors.view(BATCH_SIZE,1,embed_size)
        self_out_vectors_T=self_out_vectors.view(BATCH_SIZE,embed_size,1)

        input_vectors_T = target_input_vectors.view(BATCH_SIZE, embed_size, 1)
        output_vectors_T = source_output_vectors.view(BATCH_SIZE, embed_size, 1)

        p1_loss=torch.bmm(input_vectors_I,self_in_vectors_T).sigmoid().log()
        p2_loss=torch.bmm(output_vectors_I,self_out_vectors_T).sigmoid().log()

        p1_noise_loss = torch.bmm(noise_vectors_input.neg() ,self_in_vectors_T).sigmoid().log()
        p2_nose_loss = torch.bmm(noise_vectors_self.neg(),output_vectors_T).sigmoid().log()
        p1_noise_loss = p1_noise_loss.squeeze().sum(1)
        p2_nose_loss=p2_nose_loss.squeeze().sum(1)

        p1_noise_loss_ex = torch.bmm(noise_vectors_self.neg(), input_vectors_T).sigmoid().log()
        p2_nose_loss_ex = torch.bmm(noise_vectors_output.neg(), self_out_vectors_T).sigmoid().log()
        p1_noise_loss_ex = p1_noise_loss_ex.squeeze().sum(1)
        p2_nose_loss_ex = p2_nose_loss_ex.squeeze().sum(1)

        p1_noise_loss2 = torch.bmm(noise_vectors_input2.neg(), self_in_vectors_T).sigmoid().log()
        p2_nose_loss2 = torch.bmm(noise_vectors_self2.neg(), output_vectors_T).sigmoid().log()
        p1_noise_loss2 = p1_noise_loss2.squeeze().sum(1)
        p2_nose_loss2 = p2_nose_loss2.squeeze().sum(1)

        p1_noise_loss_ex2 = torch.bmm(noise_vectors_self2.neg(), input_vectors_T).sigmoid().log()
        p2_nose_loss_ex2 = torch.bmm(noise_vectors_output2.neg(), self_out_vectors_T).sigmoid().log()
        p1_noise_loss_ex2 = p1_noise_loss_ex2.squeeze().sum(1)
        p2_nose_loss_ex2 = p2_nose_loss_ex2.squeeze().sum(1)

        return -(p1_loss+p2_loss+p1_noise_loss+p2_nose_loss+p1_noise_loss_ex+p2_nose_loss_ex+
                 p1_noise_loss2+p2_nose_loss2+p1_noise_loss_ex2+p2_nose_loss_ex2).mean()



class EmbeddingModel(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed

        self.self_embed = nn.Embedding(n_vocab,n_embed)
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        self.self_embed.weight.data.uniform_(-1,1)
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    def forward_self(self,words):
        self_vectors=self.self_embed(words)
        return self_vectors

    def forward_noise(self, size, N_SAMPLES,device,noise_dist):
        noise_dist = noise_dist

        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLES,
                                        replacement=True)
        noise_vectors_self = self.self_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_in = self.in_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_out = self.out_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        return noise_vectors_self, noise_vectors_in, noise_vectors_out

    def forward_noise2(self, size, N_SAMPLES,device,noise_dist):
        noise_dist = noise_dist.to(device)

        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLES,
                                        replacement=True)
        noise_vectors_self = self.self_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_in = self.in_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_out = self.out_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        return noise_vectors_self, noise_vectors_in, noise_vectors_out




class GraphConvolution(Module):
    def __init__(self, in_features, out_features,W=True, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight.data.uniform_(-1,1)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.W = W

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        if self.W:
            support = torch.mm(x, self.weight)
        else:
            support = x

        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class AggregateLabel(torch.nn.Module):
    def __init__(self,input_dim,emb_dim,device):
        super(AggregateLabel, self).__init__()
        self.aggregater = GraphConvolution(input_dim, emb_dim,W=False,bias=False)
        self.device=device

    def forward(self, x, edge_index,layer):
        label_list=[]
        for i in range(layer):
            one_layer_label = self.aggregater(x,edge_index)
            one_layer_label_arr=one_layer_label.cpu().numpy()
            one_layer_label=torch.FloatTensor(np.int64(one_layer_label_arr > 0)).to(self.device)
            label_list.append(one_layer_label-x)
            x=one_layer_label

        return label_list


def save_candate(candate_pair, file_path):
    fileObject = open(file_path, 'w')
    for ip in candate_pair:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()


def load_candate(file_path):
    f = open(file_path, "r")
    table = f.read()
    f.close()
    return table

def get_noise_dist(network, all_walks):
    int_all_words = [network.vocab2int[w] for w in all_walks]
    # Count node frequency
    int_word_counts = Counter(int_all_words)
    total_count = len(int_all_words)
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
    # node distribution
    word_freqs = np.array(list(word_freqs.values()))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
    return noise_dist


def FTsplit(int_word_counts, network):
    int_word_counts_f = copy.copy(int_word_counts)
    int_word_counts_t = copy.copy(int_word_counts)
    count_f = 0
    count_t = 0
    for key in int_word_counts.keys():
        if network.int2vocab[key].endswith("_foursquare"):
            int_word_counts_t[key] = 0
            count_f += int_word_counts[key]
        if network.int2vocab[key].endswith("_twitter"):
            int_word_counts_f[key] = 0
            count_t += int_word_counts[key]
    return int_word_counts_f, int_word_counts_t, count_f, count_t


def progress(percent, loss, width=50):
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # concat string
    print('\r%s %d%% loss:%f' % (show_str, percent, loss), end='')


def get_G_rank(G_anchor, nx_G):
    g_result = nx.DiGraph()
    for edge in nx_G.edges():
        if edge[0] in list(G_anchor.nodes()) or edge[1] in list(G_anchor.nodes()):
            g_result.add_edge(edge[0], edge[1], weight=1)
    return g_result


def get_target(words, idx, window_size):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx + 1:stop + 1]

    return list(target_words)


def get_batch(source, targets, BATCH_SIZE):
    for idx in range(0, len(source), BATCH_SIZE):
        batch_x, batch_y = [], []
        x = source[idx:idx + BATCH_SIZE]
        y = targets[idx:idx + BATCH_SIZE]

        for i in range(len(x)):
            batch_x.append(x[i])
            batch_y.append(y[i])

        yield batch_x, batch_y


def get_neighbor(anchor_node, network):
    anchor_node = network.int2vocab[anchor_node]

    seed = random.randint(0, 9)
    if seed % 2 == 0:
        try:
            node_f = random.sample(network.input_link_f[anchor_node], 1)
        except KeyError:
            node_f = random.sample(network.output_link_f[anchor_node], 1)
        try:
            node_t = random.sample(network.input_link_t[anchor_node], 1)
        except KeyError:
            node_t = random.sample(network.output_link_t[anchor_node], 1)
    else:
        try:
            node_f = random.sample(network.output_link_f[anchor_node], 1)
        except KeyError:
            node_f = random.sample(network.input_link_f[anchor_node], 1)
        try:
            node_t = random.sample(network.output_link_t[anchor_node], 1)
        except KeyError:
            node_t = random.sample(network.input_link_t[anchor_node], 1)

    node_f = node_f[0]
    node_t = node_t[0]
    return network.vocab2int[node_f], network.vocab2int[node_t]


def writeFile(out_emb, network, ouput_filename_network, pix):
    f = open(ouput_filename_network, 'w')
    f.seek(0)
    vectors = out_emb
    for k, v in network.int2vocab.items():
        if v.endswith('_label'):
            continue
        if v.endswith('_anchor'):
            if '-' in v:
                # print(v)
                # v=v.replace('_anchor','')
                # vs=v.split('-')
                # if pix=='_foursquare':
                #     v=vs[0]+'_anchor'
                # else:
                #     v=vs[1]+'_anchor'
                continue
            v = v.replace('_anchor', pix)

        if v.endswith(pix):
            f.write(v)
            f.write(" ")
            value = torch.Tensor.cpu(vectors[k])
            value = value.data.numpy()
            for val in value:
                f.write(str(val))
                f.write("|")
            f.write("\n")
    print(len(vectors))
    f.flush()
    f.close()


def record_embedding(out_emb, network):
    vectors = out_emb
    embedding_dict = dict()
    for k, v in network.int2vocab.items():
        value = torch.Tensor.cpu(vectors[k])
        embedding_dict[v] = value
    return embedding_dict


def readData(file_name, pix, anchor, graph, graph_another):
    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line.split(" ", 1)

            array_edge[1] = array_edge[1].replace("\n", "")
            array_edge_0 = array_edge[0] + pix
            array_edge_1 = array_edge[0] + pix
            if array_edge[0] in anchor:
                array_edge[0] += '_anchor'
            else:
                array_edge[0] += pix
            if array_edge[1] in anchor:
                array_edge[1] += '_anchor'
            else:
                array_edge[1] += pix

            graph.add_node(array_edge[0])
            graph.add_node(array_edge[1])
            graph.add_edge(array_edge[0], array_edge[1], weight=1)

            graph_another.add_node(array_edge_0)
            graph_another.add_node(array_edge_1)
            graph_another.add_edge(array_edge_0, array_edge_1, weight=1)

    del anchor
    f.close()


def getAnchors(network, anchor_file):
    answer_list = []
    file_name = anchor_file
    # read anchor file
    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line
            array_edge = array_edge.replace("\n", "")
            answer_list.append(array_edge)
            array_edge = array_edge + '_anchor'
            network.add_node(array_edge)
    print(len(answer_list))
    return answer_list


def remove_node(nx_G):
    closure_list = []
    graph_x = nx.Graph()
    for edge in nx_G.edges():
        graph_x.add_edge(edge[0], edge[1])
    for closure in nx.connected_components(graph_x):
        closure_list.append(closure)
    if len(closure_list[0]) > len(closure_list[1]):
        for node in list(closure_list[1]):
            nx_G.remove_node(node)
    else:
        for node in list(closure_list[0]):
            nx_G.remove_node(node)
    return nx_G


def read_graph(filex, filey, anchor_file):
    '''
    Reads the input network in networkx.
    '''
    networkx_file = filex
    networky_file = filey
    graph = nx.DiGraph()
    graph_f = nx.DiGraph()
    graph_t = nx.DiGraph()

    anchor_list = getAnchors(graph, anchor_file)
    readData(networkx_file, "_foursquare", anchor_list, graph, graph_f)
    readData(networky_file, "_twitter", anchor_list, graph, graph_t)
    return graph, graph_f, graph_t, anchor_list


def read_graph_one(file, pix):
    '''
        Reads the input network in networkx.
    '''
    graph = nx.DiGraph()

    anchor_list = getAnchors(graph)
    readData(file, pix, anchor_list, graph)
    return graph


def get_graph_anchorBy_mark(G, mark_pair, num_mark):
    graph_anchor = nx.DiGraph()
    for edge in G.edges():
        if edge[0] in mark_pair.keys() or edge[1] in mark_pair.keys():
            graph_anchor.add_edge(edge[0], edge[1], weight=1)

    return graph_anchor


def get_graph_anchor(nx_G):
    graph_anchor = nx.DiGraph()
    for edge in nx_G.edges():
        if edge[0].endswith("_anchor") or edge[1].endswith("_anchor"):
            graph_anchor.add_edge(edge[0], edge[1], weight=1)
    return graph_anchor


def get_graph_noise(nx_G, G_anchor):
    list_n = list(nx_G.nodes())
    list_a = list(G_anchor.nodes())
    G_noise = list(set(list_n).difference(set(list_a)))
    return G_noise


def change2tensor(list):
    list = torch.Tensor(list)
    list = list.squeeze()
    list = Variable(list)
    return list


def test_candate_pair(candate_pair, what, test_anchor_list, mark_pair):
    mark = 0
    miss = 0
    for l in candate_pair:
        ls = l.split('-')
        lsf = ls[0].split('_')
        lst = ls[1].split('_')
        if lsf[0] == lst[0]:
            mark += 1

    print('\n', what, mark, ',all all:', len(candate_pair))
    return mark, miss


def get_test_anchor(file_name):
    answer_list = []
    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line
            array_edge = array_edge.replace("\n", "")
            answer_list.append(array_edge)
    return answer_list

def remark(all_candate,network):
    result_candate=[]
    for can in all_candate:
        cans=can.split('-')
        if network.mark_pair[cans[0]]==network.mark_pair[cans[1]]:
            result_candate.append(can)
    result_candate=list(set(result_candate))
    return result_candate


def get_graph_label(network,device):
    init_embed = nn.Embedding(len(network.vocab2int), len(network.vocab2int)).to(device)
    init_emb = torch.eye(len(network.vocab2int)).to(device)
    i = 0
    for node in network.G.nodes():
        if node.endswith('_anchor') or node.endswith('_mark'):
            init_embed.weight.data[network.vocab2int[node]] = init_emb[i]
            i += 1
        else:
            init_embed.weight.data[network.vocab2int[node]] = torch.zeros_like(init_emb[i])
    return init_embed


def get_graph_label(network,init_dim,device):
    init_embed = nn.Embedding(len(network.vocab2int), init_dim).to(device)
    init_emb = torch.eye(init_dim).to(device)
    i = 0
    for node in network.G.nodes():
        if node in network.mark_pair.keys():
            init_embed.weight.data[network.vocab2int[node]] = init_emb[network.mark_pair[node]]
            i += 1
        else:
            init_embed.weight.data[network.vocab2int[node]] = torch.zeros_like(init_emb[-1])
    return init_embed



def get_candate_pair(layers_embedding,network):
    layers_embedding_0 = F.normalize(layers_embedding)
    embedding_f_anchor = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor = layers_embedding_0[network.mark_T_intlist]


    embedding_f_anchor_mark = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor_mark = layers_embedding_0[network.mark_T_intlist]
    F2T = torch.mm(embedding_f_anchor, embedding_t_anchor_mark.T)
    T2F = torch.mm(embedding_t_anchor, embedding_f_anchor_mark.T)

    if np.all(F2T.cpu().detach().numpy()==0):
        return [],[],[]

    # embedding_f_anchor= layers_embedding[0][network.F_intlist]
    # embedding_t_anchor= layers_embedding[0][network.T_intlist]
    #
    # F2T = F.cosine_similarity(embedding_f_anchor.unsqueeze(1),embedding_t_anchor.unsqueeze(0),dim=2)
    # T2F = F.cosine_similarity(embedding_t_anchor.unsqueeze(1),embedding_f_anchor.unsqueeze(0),dim=2)

    idx2t = np.argmax(F2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(T2F.cpu().detach().numpy(), axis=1)
    idx2t_n = np.max(F2T.cpu().detach().numpy(), axis=1)
    idx2f_n = np.max(T2F.cpu().detach().numpy(), axis=1)
    print()
    print(idx2f_n)
    print(idx2t_n)
    for i in range(len(idx2t_n)):
        F2T[i][idx2t[i]]=0
    for i in range(len(idx2f_n)):
        T2F[i][idx2f[i]]=0

    idx2t_sec = np.argmax(F2T.cpu().detach().numpy(), axis=1)
    idx2f_sec = np.argmax(T2F.cpu().detach().numpy(), axis=1)
    idx2t_n_sec = np.max(F2T.cpu().detach().numpy(), axis=1)
    idx2f_n_sec = np.max(T2F.cpu().detach().numpy(), axis=1)
    # print(idx2f_n_sec)
    # print(idx2t_n_sec)


    allign_list1 = []
    allign_list2 = []
    result_old_list = []

    for t, f in zip(idx2t, network.mark_F_intlist):
        # print(network.int2vocab[int(f)],network.int2vocab[int(network.T_intlist[t])])
        allign_list1.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.mark_T_intlist[t])])
        if network.int2vocab[int(network.mark_T_intlist[t])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.mark_T_intlist[t])])

    for f, t in zip(idx2f, network.mark_T_intlist):
        # print(network.int2vocab[int(network.F_intlist[f])],network.int2vocab[int(t)])
        allign_list2.append(network.int2vocab[int(network.mark_F_intlist[f])] + '-' + network.int2vocab[int(t)])
        if network.int2vocab[int(network.mark_F_intlist[f])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(network.mark_F_intlist[f])] + '-' + network.int2vocab[int(t)])
    #print('result old list:',result_old_list)

    result_list=[]
    for a in allign_list1:
        if a in allign_list2:
            result_list.append(a)

    result_list2=list(set(allign_list1+allign_list2))

    result_list_un=[]
    for re in result_list2:
        if re not in result_list:
            result_list_un.append(re)
    result_list_new,_,_=get_candate_pair_new(layers_embedding,network)

    result_list.extend(result_list_new)
    return result_list,result_list_un,result_old_list



def get_candate_pair_self(layers_embedding,network):
    layers_embedding_0 = F.normalize(layers_embedding)
    embedding_f_anchor = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor = layers_embedding_0[network.mark_T_intlist]


    embedding_f_anchor_mark = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor_mark = layers_embedding_0[network.mark_T_intlist]
    T2T = torch.mm(embedding_t_anchor, embedding_t_anchor_mark.T)
    F2F = torch.mm(embedding_f_anchor, embedding_f_anchor_mark.T)

    if np.all(T2T.cpu().detach().numpy()==0):
        return [],[],[]

    # embedding_f_anchor= layers_embedding[0][network.F_intlist]
    # embedding_t_anchor= layers_embedding[0][network.T_intlist]
    #
    # F2T = F.cosine_similarity(embedding_f_anchor.unsqueeze(1),embedding_t_anchor.unsqueeze(0),dim=2)
    # T2F = F.cosine_similarity(embedding_t_anchor.unsqueeze(1),embedding_f_anchor.unsqueeze(0),dim=2)

    idx2t = np.argmax(T2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(F2F.cpu().detach().numpy(), axis=1)
    idx2t_n = np.max(T2T.cpu().detach().numpy(), axis=1)
    idx2f_n = np.max(F2F.cpu().detach().numpy(), axis=1)
    print()
    print(idx2f_n)
    print(idx2t_n)
    for i in range(len(idx2t_n)):
        T2T[i][idx2t[i]]=0
    for i in range(len(idx2f_n)):
        F2F[i][idx2f[i]]=0

    idx2t = np.argmax(T2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(F2F.cpu().detach().numpy(), axis=1)
    # print(idx2f_n_sec)
    # print(idx2t_n_sec)


    allign_list1 = []
    allign_list2 = []
    result_old_list = []

    for f_t, f in zip(idx2f, network.mark_F_intlist):
        # print(network.int2vocab[int(f)],network.int2vocab[int(network.T_intlist[t])])
        allign_list1.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.mark_F_intlist[f_t])])
        if network.int2vocab[int(network.mark_F_intlist[f_t])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.mark_F_intlist[f_t])])

    for t_f, t in zip(idx2t, network.mark_T_intlist):
        # print(network.int2vocab[int(network.F_intlist[f])],network.int2vocab[int(t)])
        allign_list2.append(network.int2vocab[int(network.mark_T_intlist[t_f])] + '-' + network.int2vocab[int(t)])
        if network.int2vocab[int(network.mark_T_intlist[t_f])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(network.mark_T_intlist[t_f])] + '-' + network.int2vocab[int(t)])
    #print('result old list:',result_old_list)
    allign_count1=dict(collections.Counter(allign_list1))
    allign_list1=[]
    for key,value in allign_count1.items():
        if value>1:
            allign_list1.append(key)
    allign_count2 = dict(collections.Counter(allign_list2))
    allign_list2 = []
    for key, value in allign_count2.items():
        if value > 1:
            allign_list2.append(key)
    result_list=list(set(allign_list1+allign_list2))

    return result_list,[],result_old_list




def get_candate_pair_new(layers_embedding,network):
    layers_embedding_0 = F.normalize(layers_embedding)
    embedding_f_anchor = layers_embedding_0[network.F_intlist]
    embedding_t_anchor = layers_embedding_0[network.T_intlist]


    embedding_f_anchor_mark = layers_embedding_0[network.F_intlist]
    embedding_t_anchor_mark = layers_embedding_0[network.T_intlist]
    F2T = torch.mm(embedding_f_anchor, embedding_t_anchor_mark.T)
    T2F = torch.mm(embedding_t_anchor, embedding_f_anchor_mark.T)

    # embedding_f_anchor= layers_embedding[0][network.F_intlist]
    # embedding_t_anchor= layers_embedding[0][network.T_intlist]
    #
    # F2T = F.cosine_similarity(embedding_f_anchor.unsqueeze(1),embedding_t_anchor.unsqueeze(0),dim=2)
    # T2F = F.cosine_similarity(embedding_t_anchor.unsqueeze(1),embedding_f_anchor.unsqueeze(0),dim=2)
    if np.all(F2T.cpu().detach().numpy()==0):
        return [],[],[]
    idx2t = np.argmax(F2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(T2F.cpu().detach().numpy(), axis=1)
    idx2t_n = np.max(F2T.cpu().detach().numpy(), axis=1)
    idx2f_n = np.max(T2F.cpu().detach().numpy(), axis=1)
    print()
    # print(idx2f_n)
    # print(idx2t_n)
    for i in range(len(idx2t_n)):
        F2T[i][idx2t[i]]=0
    for i in range(len(idx2f_n)):
        T2F[i][idx2f[i]]=0



    allign_list1 = []
    allign_list2 = []
    result_old_list = []

    for t, f in zip(idx2t, network.F_intlist):
        # print(network.int2vocab[int(f)],network.int2vocab[int(network.T_intlist[t])])
        allign_list1.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.T_intlist[t])])
        if network.int2vocab[int(network.T_intlist[t])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.T_intlist[t])])

    for f, t in zip(idx2f, network.T_intlist):
        # print(network.int2vocab[int(network.F_intlist[f])],network.int2vocab[int(t)])
        allign_list2.append(network.int2vocab[int(network.F_intlist[f])] + '-' + network.int2vocab[int(t)])
        if network.int2vocab[int(network.F_intlist[f])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(network.F_intlist[f])] + '-' + network.int2vocab[int(t)])
    # print('result old list:',result_old_list)

    result_list=[]
    for a in allign_list1:
        if a in allign_list2:
            result_list.append(a)

    result_list2=list(set(allign_list1+allign_list2))

    result_list_un=[]
    for re in result_list2:
        if re not in result_list:
            result_list_un.append(re)

    return result_list,result_list_un,result_old_list


def build_graph(G1, G2, anchor, alignment):
    alignment = {int(x): alignment[x] for x in alignment}
    edge1 = G1.edge_index.t().cpu().numpy().tolist()
    edge2 = G2.edge_index.t().cpu().numpy().tolist()
    node = [-1] * (G1.edge_index.max() + 1)
    idx_set = set(list(range(G1.edge_index.max() + 1))) - set(alignment.values())
    idx_list = list(idx_set)
    idx_use_count = 0
    for idx in range(G1.edge_index.max() + 1):
        if idx in alignment.keys():
            node[idx] = alignment[idx]
        else:
            node[idx] = idx_list[idx_use_count]
            idx_use_count += 1

    added_edges = set()
    nx_G = nx.DiGraph()
    for edge in edge1:
        s = node[edge[0]]
        t = node[edge[1]]
        if s in anchor.values():
            s = str(s) + "_anchor"
        else:
            s = str(s) + "_twitter"

        if t in anchor.values():
            t = str(t) + "_anchor"
        else:
            t = str(t) + "_twitter"

        nx_G.add_node(s)
        nx_G.add_node(t)

        edge = (s, t)
        reverse_edge = (t, s)

        if edge not in added_edges:
            nx_G.add_edge(*edge, weight=1)
            added_edges.add(edge)

        if reverse_edge not in added_edges:
            nx_G.add_edge(*reverse_edge, weight=1)
            added_edges.add(reverse_edge)


    for edge in edge2:
        s = edge[0]
        t = edge[1]
        if s in anchor.values():
            s = str(s) + "_anchor"
        else:
            s = str(s) + "_foursquare"

        if t in anchor.values():
            t = str(t) + "_anchor"
        else:
            t = str(t) + "_foursquare"

        nx_G.add_node(s)
        nx_G.add_node(t)

        edge = (s, t)
        reverse_edge = (t, s)

        if edge not in added_edges:
            nx_G.add_edge(*edge, weight=1)
            added_edges.add(edge)

        if reverse_edge not in added_edges:
            nx_G.add_edge(*reverse_edge, weight=1)
            added_edges.add(reverse_edge)

    recover_dict = {x: y for y, x in enumerate(node)}

    return nx_G, anchor.values(), recover_dict