import align_utils
from Bio import SeqIO
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import seq_utils
from itertools import combinations
from math import log
import pickle as pkl


def run_prc(train_csv, test_csv, blastx_out, sga_out, num_class,
            pred_dir, num_iter=500):
    """Run PRC model and predict the results.
    """
    if num_class == 18:
        pred_to_label = {0: 'Amarillovirales', 1: 'Articulavirales', 2: 'Bunyavirales', 3: 'Cryppavirales', 
                         4: 'Durnavirales', 5: 'Ghabrivirales', 6: 'Hepelivirales', 7: 'Martellivirales', 
                         8: 'Mononegavirales', 9: 'Nidovirales', 10: 'Nodamuvirales', 11: 'Patatavirales', 
                         12: 'Picornavirales', 13: 'Reovirales', 14: 'Sobelivirales', 15: 'Stellavirales', 
                         16: 'Tolivirales', 17: 'Tymovirales'}
    elif num_class == 3:
        pred_to_label = {0: 'Bunyavirales', 1: 'Mononegavirales', 2: 'Picornavirales'}

    # label_to_pred = {j: i for i, j in pred_to_label.items()}
    train_df = pd.read_csv(train_csv, names=["label", "sequence"])
    train_idx_to_label = {f"train_{idx}":label for idx, label in enumerate(train_df["label"].values)}
    
    test_df = pd.read_csv(test_csv, names=["label", "sequence"])
    # test_idx_to_label = {f"test_{idx}":label for idx, label in enumerate(test_df["label"].values)}
    
    G = nx.Graph()

    # load the blastx output into the graph
    with open(blastx_out) as test_f:
        for l in test_f:
            l_info = l.split()
            if float(l_info[10]) < 1e-5:
                G.add_edge(l_info[0], l_info[1], weight = 1)

    # load the sga output into the graph
    with open(sga_out) as test_f:
        for l in test_f:
            if l[: 2] == 'ED':
                l_info = l.split()
                G.add_edge(l_info[1], l_info[2], weight = 1)

    label_prob_dict = {}
    test_list = []
    for node in G.nodes():
        if 'train' in node:
            temp_arr = np.array([0. for i in range(num_class)])
            temp_arr[train_idx_to_label[node]] = 1.0
            label_prob_dict[node] = temp_arr
        else:
            label_prob_dict[node] = np.array([1/num_class for i in range(num_class)])
            test_list.append(node)

    # remove the subgraph without label
    graph_groups_list = [G.subgraph(c) for c in nx.connected_components(G)]
    print("The number of subgraph:", len(graph_groups_list))
    
    remove_node = set()
    # with open(sub_graph_out, 'w') as tg_out:
    for g in graph_groups_list:
        temp_set = set()
        for n_id in g:
            # print(n_id)
            if 'train' in n_id:
                temp_set = set()
                break
            else:
                temp_set.add(n_id)
        remove_node = set.union(remove_node, temp_set)
    
    print(remove_node, len(remove_node))

    # remove the isolated nodes from graph and testing list
    for s_id in remove_node:
        try:
            G.remove_node(s_id)
        except nx.exception.NetworkXError:
            pass

    test_list = [i for i in test_list if i not in remove_node]

    # update the node's probability
    i = 0
    temp_test_list = test_list.copy()
    while(i < num_iter):
        random.shuffle(temp_test_list)
        if i % 10 == 0:
            print(i)
        for test_node in temp_test_list:
            neigh = [n for n in G.neighbors(test_node)]
            temp_arr = np.array([0. for i in range(num_class)])
            for n in neigh:
                temp_arr += label_prob_dict[n]
            label_prob_dict[test_node] = temp_arr/ len(neigh)
            
            if np.max(label_prob_dict[test_node]) >= 0.95:
                temp_test_list.remove(test_node)
        i += 1

    pred_out_path = os.path.join(pred_dir, 'prediction.csv')
    pred_array_out = open(f"{pred_out_path}.out", 'w')
    with open(pred_out_path, 'w') as pout:
        for n in test_list:
            label = np.argmax(label_prob_dict[n])
            pout.write(f"{n},{pred_to_label[label]}\n")
            pred_array_out.write(f"{n}\t{label_prob_dict[n]}\n")


def get_prc_prot(reads_path, blast_prot_path, prc_rst, sga_rst, num_thread):
    """Get the protein of the PRC reads.
    """
    reads_index = SeqIO.index(reads_path, 'fasta')
    prc_reads_id_list = pd.read_csv(prc_rst, sep=',', header=None)[0]

    # load the sga results
    sga_dict = {}
    with open(sga_rst) as test_f:
        for l in test_f:
            if l[: 2] == 'ED':
                l_info = l.split()

                if l_info[1] not in sga_dict:
                    sga_dict[l_info[1]] = {l_info[2]}
                else:
                    sga_dict[l_info[1]].add(l_info[2])
                
                if l_info[2] not in sga_dict:
                    sga_dict[l_info[2]] = {l_info[1]}
                else:
                    sga_dict[l_info[2]].add(l_info[1])

    # load the blastx protein
    blast_prot_index = SeqIO.index(blast_prot_path, 'fasta')
    blast_prot_id_list = [s for s in blast_prot_index]

    # get the corresponding dna
    temp_dir_path = f"{os.path.dirname(reads_path)}/prc_temp"
    num_prc_out = 0
    os.makedirs(temp_dir_path)
    with open(f"{temp_dir_path}/prc_seq.dna.fasta", 'w') as prc_dna:
        for s_id in prc_reads_id_list:
            if s_id not in blast_prot_id_list:
                prc_dna.write(f">{s_id}\n{reads_index[s_id].seq}\n")
                num_prc_out += 1

    # see if prc gets results
    if num_prc_out == 0:
        with open(f"{temp_dir_path}/prc_seq.prc.prot.fasta", 'w') as prc_prot:
            pass
        return "no_prc_graph"
    
    # translate the dna to protein
    # seq_utils.translate2protein(f"{temp_dir_path}/prc_seq.dna.fasta", num_thread, True)
    seq_utils.translate2protein_transeq(f"{temp_dir_path}/prc_seq.dna.fasta", num_thread)
    with open(f"{temp_dir_path}/prc_seq.dna.fasta.protein", 'a') as ps_a:
        for s in blast_prot_index:
            ps_a.write(f">{s}_x\n{blast_prot_index[s].seq}\n")

    # run all-all blastp with overlap > 80/3
    align_utils.make_diamond_db(fn_in=f"{temp_dir_path}/prc_seq.dna.fasta.protein", fn_out=f"{temp_dir_path}/prc_seq", cpu=6)
    align_utils.run_diamond(aa_fp=f"{temp_dir_path}/prc_seq.dna.fasta.protein", 
                db_fp=f"{temp_dir_path}/prc_seq", 
                cpu=6, diamond_out_fn=f"{temp_dir_path}/prc_seq.blastp", e_value=1)

    blastp_dict = {}
    with open(f"{temp_dir_path}/prc_seq.blastp") as bp:
        for l in bp:
            l = l.split()
            if int(l[3]) >= int(80/ 3):
                if l[0] != l[1]:
                    if l[0][: -2] not in blastp_dict:
                        blastp_dict[l[0][: -2]] = {l[0][-1]: [l[1][: -2]]}
                    else:
                        if l[0][-1] not in blastp_dict[l[0][: -2]]:
                            blastp_dict[l[0][: -2]][l[0][-1]] = [l[1][: -2]]
                            # blastp_dict[l[0][: -2]] = {l[0][-1]: 1}
                        else:
                            blastp_dict[l[0][: -2]][l[0][-1]].append(l[1][: -2])

    prc_prot_list = []
    for k, v in blastp_dict.items():
        if len(v) > 1:
            temp_frame = 0
            temp_num_inter = 0
            for f in v:
                num_inter = len(set.intersection(set(v[f]), sga_dict[k]))
                if num_inter > temp_num_inter:
                    temp_frame = f
                    temp_num_inter = num_inter
                elif num_inter == temp_num_inter:
                    temp_frame = '*'
        
            prc_prot_list.append(f"{k}_{temp_frame}")
        else:
            for f in v:
                prc_prot_list.append(f"{k}_{f}")

    prc_prot_index = SeqIO.index(f"{temp_dir_path}/prc_seq.dna.fasta.protein", 'fasta')
    with open(f"{temp_dir_path}/prc_seq.prc.prot.fasta", 'w') as prc_prot:
        for s_id in prc_prot_list:
            # if '*' not in s_id:
            #     prc_prot.write(f">{s_id[: -2]}\n{prc_prot_index[s_id].seq}\n")
            # else:
            #     prc_prot.write(f">{s_id[: -2]}\n{'X'*66}\n")
            
            if '*' not in s_id:
                prc_prot.write(f">{s_id}\n{prc_prot_index[s_id].seq}\n")
            else:
                prc_prot.write(f">{s_id}\n{'X'*66}\n")

        out_seq_list = [i[: -2] for i in prc_prot_list]
        for s in prc_reads_id_list:
            if s not in out_seq_list:
                prc_prot.write(f">{s}_n\n{'X'*66}\n")

    return "prc_graph"


def build_graph(train_embed, valid_embed, test_embed, train_blastp_out,
                train_csv, valid_csv, test_fasta, fimo_out, blastx_out, sga_out,
                motif_GCN_temp_dir_path, pseudo_pred_path, e_thres=1, num_class=18):
    """Build CNN_GCN graph.
    """
    if num_class == 18:
        pred_to_label = {0: 'Amarillovirales', 1: 'Articulavirales', 2: 'Bunyavirales', 3: 'Cryppavirales', 
                         4: 'Durnavirales', 5: 'Ghabrivirales', 6: 'Hepelivirales', 7: 'Martellivirales', 
                         8: 'Mononegavirales', 9: 'Nidovirales', 10: 'Nodamuvirales', 11: 'Patatavirales', 
                         12: 'Picornavirales', 13: 'Reovirales', 14: 'Sobelivirales', 15: 'Stellavirales', 
                         16: 'Tolivirales', 17: 'Tymovirales'}
    elif num_class == 3:
        pred_to_label = {0: 'Bunyavirales', 1: 'Mononegavirales', 2: 'Picornavirales'}

    label_to_pred = {j: i for i, j in pred_to_label.items()}
    
    # initialization
    print("Loading the pseudo label dict ... ...")
    pseudo_dict = {}
    with open(pseudo_pred_path) as ppp:
        for l in ppp:
            l_info = l.strip().split(',')
            if l_info[1] in label_to_pred:
                pseudo_dict[l_info[0]] = label_to_pred[l_info[1]]
    
    print("Loading data ... ...")
    # load the reads embedding vector
    train_embed = np.load(train_embed)
    valid_embed = np.load(valid_embed)
    test_embed = np.load(test_embed)
    num_seq = train_embed.shape[0] + valid_embed.shape[0] + test_embed.shape[0]

    print(f"train_embed: {train_embed.shape}\nvalid_embed: {valid_embed.shape}\ntest_embed: {test_embed.shape}\n")

    # relu function processing
    train_embed[train_embed < 0] = 0.0
    valid_embed[valid_embed < 0] = 0.0
    test_embed[test_embed < 0] = 0.0

    # load the index and label
    train_df = pd.read_csv(train_csv, names=["label", "sequence"])
    train_idx_to_label = {idx: label for idx, label in enumerate(train_df["label"].values)}
    
    valid_df = pd.read_csv(valid_csv, names=["label", "sequence"])
    valid_idx_to_label = {idx: label for idx, label in enumerate(valid_df["label"].values)}

    # test_df = pd.read_csv(test_csv, names=["label", "sequence"])
    # test_idx_to_label = {idx: label for idx, label in enumerate(test_df["label"].values)}
    test_id2idx = {s.id: s_index for s_index, s in enumerate(SeqIO.parse(test_fasta, 'fasta'))}

    print("Load all motifs ... ...")
    fimo_df = pd.read_csv(fimo_out, sep='\t', header=None)
    motif_list = set(fimo_df[0])
    # print(motif_list)
    print('The number of motifs:', len(motif_list))

    # build the dictionary to map the motif id to the corresponding label
    motif_dict = {m: f'filter_{i}' for i, m in enumerate(motif_list)}
    filter2label_dict = {f: label_to_pred[m.split('_')[1]] for m, f in motif_dict.items()}

    seq_motif_dict = {}
    for m, s in zip(fimo_df[0], fimo_df[1]):
        if s not in seq_motif_dict:
            seq_motif_dict[s] = set([m])
        else:
            seq_motif_dict[s].add(m)

    # building edges
    print("Building graph ... ...")
    G = nx.Graph()
    print("Build graph based on the blastp of training - training samples ... ...")
    with open(train_blastp_out) as file_in:
        for line in file_in.readlines():
            tmp = line[:-1].split()
            node1 = tmp[0]
            node2 = tmp[1]
            # weight = float(tmp[2])
            # aln_len = int(tmp[2])
            weight = float(tmp[10])
            if weight < e_thres:  # the threshold of evalue is 1 (used in building edges)
                if ('test' not in node1) and ('test' not in node2):
                    G.add_edge(node1, node2, weight = 1)
    
    print(f"Build graph based on the fimo p-value ... ...")
    with open(fimo_out) as fo:
        for l in fo:
            ls = l.split()
            node1 = ls[0]
            node2 = ls[1]
            G.add_edge(motif_dict[node1], node2, weight=1)

    # build edge using the evalue file
    print(f"Add edges based on the blastx e-value ... ...")
    with open(blastx_out) as file_in:
        for line in file_in.readlines():
            tmp = line[:-1].split()
            node1 = tmp[0]
            node2 = tmp[1]
            # weight = float(tmp[2])
            # aln_len = int(tmp[2])
            weight = float(tmp[10])
            if weight < e_thres:  # the threshold of evalue is 1 (used in building edges)
                G.add_edge(node1, node2, weight = 1)
                # calculate the weights log(1/e-value)
                # G.add_edge(node1, node2, weight=log10(1/ weight))

    # build edge using the evalue between testing sequences
    with open(sga_out) as file_in:
        for l in file_in:
            if l[: 2] == 'ED':
                tmp = l.split()
                node1 = tmp[1]
                node2 = tmp[2]

                if (node1 in pseudo_dict) and (node2 in pseudo_dict):
                    G.add_edge(node1, node2, weight=1)

    # build edge between filters
    print("Calculating pmi ... ...")
    filter_pair_count = {frozenset([i, j]): 0 for i, j in combinations(motif_list, 2)}  # calculate the dictionary to store the pairs and corresponding number
    filter_freq = {m: 0 for m in motif_list}
    
    for seqs in seq_motif_dict.values():
        seqs = list(seqs)
        if len(seqs) > 1:
            for (i, j) in combinations(seqs, 2):
                filter_pair_count[frozenset([i, j])] += 1
            for s in seqs:
                filter_freq[s] += 1

    print("Building edges between motifs ... ...")
    for key, count in filter_pair_count.items():
        if count > 0:
            i = list(key)[0]
            j = list(key)[1]
            
            freq_i = filter_freq[i]
            freq_j = filter_freq[j]
            pmi = log((1.0 * count / num_seq) /
                    (1.0 * freq_i * freq_j/(num_seq * num_seq)))
            
            if pmi > 0.0:
                G.add_edge(motif_dict[i], motif_dict[j], weight = pmi)

    # build data set
    print("Generating training data ... ...")
    train_idx = []
    valid_idx = []
    test_idx = []
    all_label = []
    feature = []

    test_to_id = {}
    cnt = 0
    zero_filter = 0
    for node in G.nodes():
        if 'train' in node:
            idx = int(node.split("train_")[1])
            all_label.append(train_idx_to_label[idx])
            train_idx.append(cnt)
            feature.append(train_embed[idx])
            cnt += 1

        elif 'valid' in node:
            idx = int(node.split("validation_")[1])
            all_label.append(valid_idx_to_label[idx])
            valid_idx.append(cnt)
            feature.append(valid_embed[idx])
            cnt += 1

        elif 'test' in node:
            if node in pseudo_dict:
                idx = test_id2idx[node]
                all_label.append(pseudo_dict[node])
                train_idx.append(cnt)
                feature.append(test_embed[idx])
                cnt += 1
            else:
                idx = test_id2idx[node]
                all_label.append(-1)
                test_idx.append(cnt)
                feature.append(test_embed[idx])
                test_to_id[node] = cnt
                cnt += 1

        elif 'filter' in node:
            idx = int(node.split("filter_")[1])
            # not assign the label to the motifs
            # all_label.append(-1)
            # assign the label to the motif
            all_label.append(filter2label_dict[node])
            train_idx.append(cnt)
            neighs = [n for n in G.neighbors(f"filter_{idx}")]
            num_avg_filter = len([n for n in neighs if 'filter' not in n])
            if num_avg_filter > 0:
                temp_f = np.zeros((num_avg_filter, train_embed.shape[1]))
                temp_f_idx = 0
                train_neigh_idx = [int(n.split('train_')[1]) for n in neighs if 'train' in n]
                # print(train_neigh_idx)
                valid_neigh_idx = [int(n.split('validation_')[1]) for n in neighs if 'valid' in n]
                # print(valid_neigh_idx)
                # test_neigh_idx = [int(n.split('test_')[1]) for n in neighs if 'test' in n]
                test_neigh_idx = [test_id2idx[n] for n in neighs if 'test' in n]
                # print(test_neigh_idx)
                for i in train_neigh_idx:
                    temp_f[temp_f_idx] = train_embed[i]
                    temp_f_idx += 1
                for i in valid_neigh_idx:
                    temp_f[temp_f_idx] = valid_embed[i]
                    temp_f_idx += 1
                for i in test_neigh_idx:
                    temp_f[temp_f_idx] = test_embed[i]
                    temp_f_idx += 1
                # print(temp_f, num_avg_filter, np.max(temp_f))
                avg_feature = np.average(temp_f, axis=0)
                # print(avg_feature, avg_feature.shape)
                # exit(0)
            else:
                avg_feature = np.zeros((train_embed.shape[1]))
                zero_filter += 1
            
            feature.append(avg_feature)
            cnt += 1

        else:
            print(f'Unknown node: {node}')
            exit(0)

    print(zero_filter)

    # motif_GCN_temp_dir_path = os.path.join(work_dir, 'motif_GCN_temp')
    # os.makedirs(motif_GCN_temp_dir_path)
    pkl.dump(train_idx, open(f"{motif_GCN_temp_dir_path}/read.train_idx", "wb"))
    pkl.dump(valid_idx, open(f"{motif_GCN_temp_dir_path}/read.valid_idx", "wb"))
    pkl.dump(test_idx, open(f"{motif_GCN_temp_dir_path}/read.test_idx", "wb"))
    adj = nx.adjacency_matrix(G)
    pkl.dump(adj, open(f"{motif_GCN_temp_dir_path}/read.graph", "wb"))
    pkl.dump(all_label, open(f"{motif_GCN_temp_dir_path}/read.label", "wb"))
    pkl.dump(test_to_id, open(f"{motif_GCN_temp_dir_path}/read.dict", "wb"))
    feature = np.array(feature)
    pkl.dump(feature, open(f"{motif_GCN_temp_dir_path}/read.feature", "wb"))

    pkl.dump(G, open(f"{motif_GCN_temp_dir_path}/graph.nx", "wb"))


# def extract_meme(orig_dir, target_dir):
#     """Copy the meme files from original directory to the target directory.
#     """

#     os.makedirs(f"{target_dir}/motif")
#     bmp_list = ['Bunyavirales', 'Mononegavirales', 'Picornavirales']
#     file_list = ['meme.html', 'meme.txt', 'meme.xml']

#     for order in bmp_list:
#         os.makedirs(f"{target_dir}/motif/{order}")
#         for f in file_list:
#             shutil.copyfile(f"{orig_dir}/motif/{order}/{f}", f"{target_dir}/motif/{order}/{f}")


