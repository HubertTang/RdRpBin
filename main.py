import ana_utils
from Bio import SeqIO
import argparse
from collections import Counter
import os
import pandas as pd
from itertools import combinations
from math import log
import multiprocessing
from multiprocessing import Pool
from motif_gcn_scripts.data import load_data, preprocess_features, preprocess_adj, sample_mask
from motif_gcn_scripts import model
from motif_gcn_scripts.config import  args
from motif_gcn_scripts.utils import masked_loss, masked_acc
import meta_tools
import networkx as nx
import numpy as np
import pickle as pkl
import random
import rdrp_dataset
import rdrp_model
import shutil
import subprocess
import  scipy.sparse as sp
from scipy.special import softmax
import torch
from    torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def rdrpbin_cmd():
    parser = argparse.ArgumentParser(description="ARGUMENTS")

    # argument for dataset
    parser.add_argument(
        'input_file',
        type=str,
        help="Path of input file."
    )

    parser.add_argument(
        "--database",
        type=str,
        default="RdRpBin_db",
        help="Database."
        )

    parser.add_argument(
        "--format",
        default="fasta",
        type=str,
        help="Format of input file (fasta (default), fastq)")

    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='RdRpBin_beta'
    )

    args = parser.parse_args()

    assert (args.format in ['fasta', 'fastq'])

    return args


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def csv2fasta(in_csv, prefix):
    """Assign the label to the sequences and write the sequences into fasta file.
    """
    in_seq_df = pd.read_csv(in_csv, sep=',', header=None)
    with open(f"{in_csv}.index.fasta", 'w') as out_fasta:
        for s_index, s in enumerate(in_seq_df[1]):
            out_fasta.write(f">{prefix}_{s_index}\n{s}\n")


def fasta2csv_pre(fasta_file, in_format, label):
    """Convert fasta file into csv file and index-fasta file.
    """
    fasta_dir = os.path.dirname(fasta_file)
    s_index = 0
    with open(f"{fasta_dir}/test_rdrp_sim.csv", 'w') as csv:
        with open(f"{fasta_dir}/test_rdrp_sim.csv.index.fasta", 'w') as fasta:
            for s in SeqIO.parse(fasta_file, in_format):
                csv.write(f"{label},{str(s.seq)}\n")
                fasta.write(f">{s.id}_test_{s_index}\n{s.seq}\n")
                s_index += 1


def merge_text(input_dir, out_path):
    """Merge the text file in the input directory
    """

    text_file_list = os.listdir(input_dir)
    text_file_list.sort()
    with open(out_path, 'w') as outfile:
        for fname in text_file_list:
            with open(f"{input_dir}/{fname}") as infile:
                outfile.write(infile.read())


def run_blastx(database_path, train_csv, test_csv, fam_dict_path, 
               query_path, blast_out, pred_out_dir):
    """Run blastx and analyze the result.
    """
    os.system(f"diamond blastx -d {database_path} -q {query_path} -o {blast_out} -p 6 -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qframe --very-sensitive ")
    
    # load the dictionary which convert the train id into label
    train_index2label = {}
    with open(train_csv) as train:
        for l_index, l in enumerate(train):
            train_index2label[l_index] = l.split(',')[0]

    # load the family dictionary
    pfam_dict = {}
    pfam_df = pd.read_csv(fam_dict_path, sep=',', header=None)
    for order, order_label in zip(pfam_df[0], pfam_df[1]):
        pfam_dict[str(order_label)] = order
    pfam_dict['other'] = len(pfam_dict)

    # save the result into a dictionary
    test_rst_dict = {}
    with open(blast_out) as test_f:
        for l in test_f:
            l_info = l.split()
            # if float(l_info[10]) < 10:
            if float(l_info[10]) < 1e-5:
                train_label = train_index2label[int(l_info[1].split('_')[1])]
                if l_info[0] in test_rst_dict:
                    test_rst_dict[l_info[0]].append(train_label)
                else:
                    test_rst_dict[l_info[0]] = [train_label]

    # majority voting and save the prediction
    with open(os.path.join(pred_out_dir, 'prediction.csv'), 'w') as out:
        for k, v in test_rst_dict.items():
            c = Counter(v)
            v_major, _ = c.most_common()[0]
            out.write(f"{k},{pfam_dict[v_major]}\n")

    # # analyze the result
    # ana_utils.analyze_predict_result(pred_rst=os.path.join(pred_out_dir, 'prediction.csv'), 
    #                                  truth_csv=test_csv, fam_path=fam_dict_path, 
    #                                  out_dir=pred_out_dir, num_class=18)


def trans_spec_frame(dna_seq, spec_frame):
    """Translate all DNA sequences in fasta file
    """
    frame_dict = {1: 0, -1: 1, 2: 2, -2: 3, 3: 4, -3: 5}

    dna_seq = dna_seq.seq
    # use both fwd and rev sequences
    dna_seqs = [dna_seq, dna_seq.reverse_complement()]

    # generate all translation frames
    aa_seqs = [s[i:].translate(stop_symbol="X") for i in range(3) for s in dna_seqs]
    # for aas in aa_seqs:
    #     print(aas)
    # exit(0)
    return aa_seqs[frame_dict[spec_frame]]
    

def get_blastx_right_frame(reads_path, blastx_out, blastx_prot_out, blastx_nucl_out):
    """Get the correct frame of sga.
    """
    reads_index = SeqIO.index(reads_path, 'fasta')
    seq_dict = {}
    with open(blastx_out) as blo:
        for l in blo:
            l = l.strip().split()
            if l[0] not in seq_dict:
                seq_dict[l[0]] = int(l[12])

    with open(blastx_prot_out, 'w') as bpo:
        for s_id in seq_dict:
            temp_prot = trans_spec_frame(reads_index[s_id], seq_dict[s_id])
            bpo.write(f">{s_id}\n{temp_prot}\n")

    with open(blastx_nucl_out, 'w') as bno:
        for s_id in seq_dict:
            bno.write(f">{s_id}\n{reads_index[s_id].seq}\n")


def run_sga(reads_file, tar_f):
    """Run SGA on all the reads to build the edges.
    """
    reads_dir = os.path.dirname(reads_file)
    os.system(f"cd {reads_dir}\nsga preprocess {reads_file} > {reads_file}.sga.prep")
    os.system(f"cd {reads_dir}\nsga index -t 6 -a ropebwt {reads_file}.sga.prep")
    os.system(f"cd {reads_dir}\nsga preprocess {tar_f} > {tar_f}.sga.prep")
    os.system(f"cd {reads_dir}\nsga index -t 6 -a ropebwt {tar_f}.sga.prep")
    os.system(f"cd {reads_dir}\nsga overlap -t 6 -m 80 -e 0.01 -d 2 -f {tar_f}.sga.prep --exhaustive {reads_file}.sga.prep")
    os.system(f"cd {reads_dir}\ngunzip -f test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg.gz")


def run_prc(train_csv, test_csv, blastx_out, sga_out, num_class,
            pred_dir, fam_path, num_iter=500):
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
    elif num_class == 9:
        pred_to_label = {0: 'Cryppavirales', 1: 'Durnavirales', 2: 'Ghabrivirales', 3: 'Hepelivirales', 
                         4: 'Nidovirales', 5: 'Nodamuvirales', 6: 'Sobelivirales', 7: 'Stellavirales', 
                         8: 'Tolivirales'}
    label_to_pred = {j: i for i, j in pred_to_label.items()}
    train_df = pd.read_csv(train_csv, names=["label", "sequence"])
    train_idx_to_label = {f"train_{idx}":label for idx, label in enumerate(train_df["label"].values)}
    
    test_df = pd.read_csv(test_csv, names=["label", "sequence"])
    test_idx_to_label = {f"test_{idx}":label for idx, label in enumerate(test_df["label"].values)}
    
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
        # for e in g.edges():
        #     print(e)
        #     for n_id in e:
        #         if 'train' in n_id:
        #             break
        #         else:
        #             temp_set.add(n_id)
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

    # # analyze the result
    # ana_utils.analyze_predict_result(pred_rst=pred_out_path, truth_csv=test_csv, 
    #                                  fam_path=fam_path, out_dir=pred_dir, 
    #                                  num_class=num_class)


def translate2protein(input_file_path, filter=True):
    """Translate the DNA to protein.
    """
    work_dir = os.path.dirname(input_file_path)
    input_file_name = input_file_path.split('/')[-1]
    
    # translate and filter out the reads with stop codon
    num_procr = multiprocessing.cpu_count()
    meta_tools.split_fasta(fasta_file=input_file_path, num_split=num_procr)
    
    pool = Pool(processes=num_procr)
    for temp_id in range(num_procr):
        pool.apply_async(meta_tools.trans_6_frame_all, [f"{input_file_path}.{temp_id}", filter])
    pool.close()
    pool.join()

    # move all the file into a temp directory
    os.makedirs(f"{work_dir}/temp")
    for i in range(num_procr):
        os.remove(f"{input_file_path}.{i}")
        shutil.move(f"{input_file_path}.{i}.protein", 
                    f"{work_dir}/temp/{input_file_name}.{i}.protein")
    merge_text(input_dir=f"{work_dir}/temp",
               out_path=f"{work_dir}/{input_file_name}.protein")
    
    shutil.rmtree(f"{work_dir}/temp")


def translate2protein_transeq(input_file_path):
    """Translate the DNA to protein.
    """
    work_dir = os.path.dirname(input_file_path)
    input_file_name = input_file_path.split('/')[-1]
    
    # translate and filter out the reads with stop codon
    num_procr = multiprocessing.cpu_count()
    meta_tools.split_fasta(fasta_file=input_file_path, num_split=num_procr)

    pool = Pool(processes=num_procr)
    for temp_id in range(num_procr):
        pool.apply_async(meta_tools.transeq, [f"{input_file_path}.{temp_id}"])
    pool.close()
    pool.join()
    
    # move all the file into a temp directory
    os.makedirs(f"{work_dir}/temp")
    for i in range(num_procr):
        os.remove(f"{input_file_path}.{i}")
        os.remove(f"{input_file_path}.{i}.pep")
        shutil.move(f"{input_file_path}.{i}.protein", 
                    f"{work_dir}/temp/{input_file_name}.{i}.protein")
    merge_text(input_dir=f"{work_dir}/temp",
               out_path=f"{work_dir}/{input_file_name}.protein")
    
    shutil.rmtree(f"{work_dir}/temp")


def make_diamond_db(fn_in, fn_out, cpu: int):
    """Build Diamond blastp database.
    """
    os.system(f"./diamond makedb --threads {cpu} --in {fn_in} -d {fn_out}")


def run_diamond(aa_fp, db_fp, cpu: int, diamond_out_fn, e_value):
    """Run Diamond to blastp
    """
    os.system(f"./diamond blastp --threads {cpu} --sensitive -d {db_fp} -q {aa_fp} -o {diamond_out_fn} -e {e_value}")


def get_prc_prot(reads_path, blast_prot_path, prc_rst, sga_rst):
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
    os.makedirs(temp_dir_path)
    with open(f"{temp_dir_path}/prc_seq.dna.fasta", 'w') as prc_dna:
        for s_id in prc_reads_id_list:
            if s_id not in blast_prot_id_list:
                prc_dna.write(f">{s_id}\n{reads_index[s_id].seq}\n")
    
    # translate the dna to protein
    translate2protein(f"{temp_dir_path}/prc_seq.dna.fasta", True)
    with open(f"{temp_dir_path}/prc_seq.dna.fasta.protein", 'a') as ps_a:
        for s in blast_prot_index:
            ps_a.write(f">{s}_x\n{blast_prot_index[s].seq}\n")

    # run all-all blastp with overlap > 80/3
    make_diamond_db(fn_in=f"{temp_dir_path}/prc_seq.dna.fasta.protein", fn_out=f"{temp_dir_path}/prc_seq", cpu=6)
    run_diamond(aa_fp=f"{temp_dir_path}/prc_seq.dna.fasta.protein", 
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


def reads_shift_trans(read_2, s1, e1, s2, e2, f1, r):
    """read_1 and read_2 are in Bio.SeqRecord froamt.
    """
    frame_index_p = [1, 2, 3] * 100
    frame_index_n = [-3, -2, -1] * 100

    if r == 1:
        read_2 = read_2[::-1]
        s2 = len(read_2) - e2
        e2 = len(read_2) - s2

    if f1 > 0:
        frame_dict = {3: 2, 2: 3, 1: 1}
        if s1 > 0:
            f2 = frame_dict[frame_index_p[f1-1: ][s1]]
            s2_prot = trans_spec_frame(read_2, spec_frame=f2)
            
        else:
            if frame_index_p[s1] == f1:
                f2 = 1
            elif frame_index_p[1: ][s1] == f1:
                f2 = 2
            else:
                f2 = 3
            s2_prot = trans_spec_frame(read_2, spec_frame=frame_dict[f2])
    
    else:
        frame_dict = {-3: -2, -2: -3, -1: -1}
        if s1 > 0:
            if frame_index_n[e2 - len(read_2)] == f1:
                f2 = -1
            elif frame_index_n[: -1][e2 - len(read_2)] == f1:
                f2 = -2
            else:
                f2 = -3
            s2_prot = trans_spec_frame(read_2, spec_frame=frame_dict[f2])
        
        else:
            if f1 < -1:
                frame_index_n = frame_index_n[: f1+1]
            f2 = frame_dict[frame_index_n[s2]]

    return s2_prot


def get_prc_prot_v2(reads_path, blastx_rst, sga_rst):
    """Get the corecte frame of reads based on the SGA result.
    """
    reads_index = SeqIO.index(reads_path, 'fasta')

    # parse the blastx result and only consider the frame with the smallest e-value
    blastx_frame_dict = {}
    with open(blastx_rst) as b_rst:
        for l in b_rst:
            l = l.strip().split()
            if l[0] not in blastx_frame_dict:
                blastx_frame_dict[l[0]] = int(l[-1])
    
    # parse the sga result
    sga_rst_dict = {}
    with open(sga_rst) as s_rst:
        for l in s_rst:
            if l[: 2] == 'ED':
                l = l.strip().split()


def process_before_fimo(all_trans_path, prc_prot_path, blastx_prot_path):
    """Remove the repeated blastx/ prc protein and add blastx protein with stop codon.
    """
    prc_prot_index = SeqIO.index(prc_prot_path, 'fasta')
    prc_blastx_list = [s[: -2] for s in prc_prot_index]

    with open(f"{all_trans_path}.full", 'w') as atp:
        for s in SeqIO.parse(blastx_prot_path, 'fasta'):
            if s.id not in prc_blastx_list:
                atp.write(f">{s.id}\n{s.seq}\n")
                prc_blastx_list.append(s.id)
        
        for s in SeqIO.parse(all_trans_path, 'fasta'):
            if s.id[: -2] not in prc_blastx_list:
                atp.write(f">{s.id}\n{s.seq}\n")

        for s in prc_prot_index:
            atp.write(f">{s[: -2]}\n{prc_prot_index[s].seq}\n")


def fimo(meme_file, fasta_file, output_dir):
    """Run fimo.
    """
    os.system("fimo --oc %s --verbosity 1 --thresh 1.0E-4 %s %s" % (output_dir, meme_file, fasta_file))


def run_fimo(motif_dir, fasta_file_path, out_dir, num_class=18):
    """Run fimo in parallel.
    """
    if num_class == 18:
        order_list = ['Amarillovirales', 'Articulavirales', 'Bunyavirales', 'Cryppavirales', 
                    'Durnavirales', 'Ghabrivirales', 'Hepelivirales', 'Martellivirales', 
                    'Mononegavirales', 'Nidovirales', 'Nodamuvirales', 'Patatavirales', 
                    'Picornavirales', 'Reovirales', 'Sobelivirales', 'Stellavirales', 
                    'Tolivirales', 'Tymovirales']
    elif num_class == 3:
        order_list = ['Bunyavirales', 'Mononegavirales', 'Picornavirales']
    elif num_class == 9:
        order_list = ['Cryppavirales', 'Durnavirales', 'Ghabrivirales', 'Hepelivirales', 
                      'Nidovirales', 'Nodamuvirales', 'Sobelivirales', 'Stellavirales', 
                      'Tolivirales']
    
    num_procr = multiprocessing.cpu_count()
    pool = Pool(processes=num_procr)
    for order in order_list:
        meme_path = os.path.join(motif_dir, order, 'meme.txt')
        out_path = os.path.join(out_dir, order)
        pool.apply_async(fimo, (meme_path, fasta_file_path, out_path))
    pool.close()
    pool.join()

    # remove the temporary file
    for order in order_list:
        for f in os.listdir(os.path.join(out_dir, order)):
            if f[: 5] == 'cisml':
                os.remove(os.path.join(out_dir, order, f))


def dna_csv2protein_csv(dna_csv_path, protein_fasta_path, protein_csv_out):
    """Convert the dna csv into the protein csv.
    """
    dna_csv_df = pd.read_csv(dna_csv_path, sep=',', header=None)
    protein_seq_dict = {s.id[: -2]: s.seq for s in SeqIO.parse(protein_fasta_path, 'fasta')}
    
    with open(protein_csv_out, 'w') as pco:
        for seq_index, label in enumerate(dna_csv_df[0]):
            if f"test_{seq_index}" in protein_seq_dict:
                seq = protein_seq_dict[f"test_{seq_index}"]
                pco.write(f"{label},{seq}\n")
            else:
                seq = 'X' * 66
                pco.write(f"{label},{seq}\n")


# def fasta2csv(fasta_file, in_format, output_len, label):
def fasta2csv(fasta_file, in_format, label, out_len=66):
    """Convert fasta file into csv file.
    """
    with open(f"{fasta_file}.csv", 'w') as csv:
        for s in SeqIO.parse(fasta_file, in_format):
            # csv.write(f"{label},{str(s.seq)[: output_len].ljust(output_len, 'X')}\n")
            csv.write(f"{label},{str(s.seq)[: out_len]}\n")


def pred_fc(data_loader, model, num_hidden, num_data):
    model.eval()
    result = np.ones((num_data, num_hidden))

    before_index = 0

    for (datas, labels) in tqdm(data_loader):
        num_s = datas.shape[0]
        datas = datas.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(datas)

            temp_npy = outputs.cpu().numpy()

            result[before_index: before_index + num_s, :] = temp_npy
            before_index += num_s

    return result


def run_CNN_pred(database_name, test_csv, num_class=18, seq_len=66, num_token=21, batch_size=1024):
    """Run CNN and output the FC1 vector.
    """
    model_path_dict = {"cdhit4_613_c1": "log/06-24_15-18/06-24_15-18.pt",
                       "cdhit4_613_c5": "log/06-23_17-19/06-23_17-19.pt",
                       "cdhit6_613_c1": "log/06-24_15-20/06-24_15-20.pt",
                       "cdhit6_613_c5": "log/06-29_21-37/06-29_21-37.pt",
                       "BMP_cdhit4_613_c1": "log/07-14_11-42/07-14_11-42.pt",
                       "BMP_cdhit4_613_c5": "log/07-14_20-18/07-14_20-18.pt",
                       "BMP_cdhit6_613_c1": "log/07-14_11-46/07-14_11-46.pt",
                       "BMP_cdhit6_613_c5": "log/07-14_20-22/07-14_20-22.pt",
                       "BMP_camisim_cdhit4": "log/07-15_12-47/07-15_12-47.pt",
                    #    "cdhit95_database": "log/07-26_21-12/07-26_21-12.pt"}
                       "cdhit95_database": "log/07-28_23-17/07-28_23-17.pt",
                       "cdhit95_9_database": "log/07-29_16-56/07-29_16-56.pt",
                       "cdhit95100_db_66": "log/09-26_14-50/09-26_14-50.pt"}
    
    # from history.csv import the parameters of the model
    df = pd.read_csv("log/history.csv", sep=',')
    model_name = model_path_dict[database_name].split('/').pop().split('.').pop(0)
    for index, n in enumerate(df['Name']):
        if n == model_name:
            row_id = index
            break

    model = df['Model(or trained)'][row_id]
    encoding = df['Encoding'][row_id]
    filter_size = [int(i) for i in df['Filter_size'][row_id].strip().strip('[]').split()]
    num_filter = int(df['Num_filters'][row_id])
    dropout = float(df['Dropout'][row_id])
    num_hidden = int(df['Num_hidden'][row_id])

    # load the testing dataset
    test_data = eval(f"rdrp_dataset.{encoding}")(file=test_csv, seq_len=seq_len)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6)

    # load the trained model
    trained_model = eval(f"rdrp_model.{model}")(num_class=num_class, num_token=num_token, 
                         seq_len=seq_len, kernel_nums=num_filter, kernel_sizes=filter_size, 
                         dropout=dropout, num_fc=num_hidden, out_logit=False).to(device)
    trained_model.load_state_dict(torch.load(model_path_dict[database_name]))
    
    num_test = int(subprocess.check_output(f'wc -l {test_csv}', shell=True).split()[0])

    temp_arr = pred_fc(test_loader, trained_model, num_hidden, num_test)
    print(f"Test embedding vector size: {temp_arr.shape}")
    np.save(f"{test_csv}.FC1", temp_arr)


def run_merge_fimo_out(train_csv, fimo_out_path, train_meme_dir, 
                       test_meme_dir, thres=1e-5, num_class=18):
    """Merge the fimo output.
    """
    if num_class == 18:
        pred_to_label = {0: 'Amarillovirales', 1: 'Articulavirales', 2: 'Bunyavirales', 3: 'Cryppavirales', 
                         4: 'Durnavirales', 5: 'Ghabrivirales', 6: 'Hepelivirales', 7: 'Martellivirales', 
                         8: 'Mononegavirales', 9: 'Nidovirales', 10: 'Nodamuvirales', 11: 'Patatavirales', 
                         12: 'Picornavirales', 13: 'Reovirales', 14: 'Sobelivirales', 15: 'Stellavirales', 
                         16: 'Tolivirales', 17: 'Tymovirales'}
    elif num_class == 3:
        pred_to_label = {0: 'Bunyavirales', 1: 'Mononegavirales', 2: 'Picornavirales'}
    elif num_class == 9:
        pred_to_label = {0: 'Cryppavirales', 1: 'Durnavirales', 2: 'Ghabrivirales', 3: 'Hepelivirales', 
                         4: 'Nidovirales', 5: 'Nodamuvirales', 6: 'Sobelivirales', 7: 'Stellavirales', 
                         8: 'Tolivirales'}

    label_to_pred = {j: i for i, j in pred_to_label.items()}
    
    # load the dictionary which can convert the index to label
    train_df = pd.read_csv(train_csv, names=["label", "sequence"])
    train_idx_to_label = {f"train_{idx}":pred_to_label[label] for idx, label in enumerate(train_df["label"].values)}
    
    # valid_df = pd.read_csv(valid_csv, names=["label", "sequence"])
    # valid_idx_to_label = {f"validation_{idx}":pred_to_label[label] for idx, label in enumerate(valid_df["label"].values)}

    # test_df = pd.read_csv(test_csv, names=["label", "sequence"])
    # test_idx_to_label = {f"test_{idx}":pred_to_label[label] for idx, label in enumerate(test_df["label"].values)}
    
    # idx2label = {**train_idx_to_label, **test_idx_to_label, **valid_idx_to_label}
    idx2label = train_idx_to_label

    fo = open(fimo_out_path, 'w')
    motif_dict = {}
    for o in label_to_pred:
        motif_set= set()
        train_fimo_df = pd.read_csv(f"{train_meme_dir}/{o}/fimo.tsv", sep='\t')
        test_fimo_df = pd.read_csv(f"{test_meme_dir}/{o}/fimo.tsv", sep='\t')
        
        for m, s, p in zip(train_fimo_df['motif_id'], train_fimo_df['sequence_name'], train_fimo_df['p-value']):
            # remove the uncorrect train reads
            if m[0] == '#':
                continue

            if p < thres:
                if 'train' in s:
                    if o == idx2label[s]:
                        fo.write(f"{m}_{o}\t{s}\t{p}\n")
                        motif_set.add(m)
                elif 'validation' in s:
                    fo.write(f"{m}_{o}\t{s}\t{p}\n")
                    motif_set.add(m)

        for m, s, p in zip(test_fimo_df['motif_id'], test_fimo_df['sequence_name'], test_fimo_df['p-value']):
            # remove the uncorrect train reads
            if m[0] == '#':
                continue
            if p < thres:
                fo.write(f"{m}_{o}\t{s}\t{p}\n")
                motif_set.add(m)

        # print(o, motif_set, len(motif_set))
        motif_dict[o] = len(motif_set)
    
    print(motif_dict)
    return motif_dict


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
    elif num_class == 9:
        pred_to_label = {0: 'Cryppavirales', 1: 'Durnavirales', 2: 'Ghabrivirales', 3: 'Hepelivirales', 
                         4: 'Nidovirales', 5: 'Nodamuvirales', 6: 'Sobelivirales', 7: 'Stellavirales', 
                         8: 'Tolivirales'}
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
            # if node in pseudo_dict:
            #     idx = test_id2idx[node]
            #     all_label.append(pseudo_dict[node])
            #     train_idx.append(cnt)
            #     feature.append(test_embed[idx])
            #     cnt += 1
            # else:
            #     idx = test_id2idx[node]
            #     all_label.append(-1)
            #     test_idx.append(cnt)
            #     feature.append(test_embed[idx])
            #     test_to_id[node] = cnt
            #     cnt += 1

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


def train_motif_GCN(motif_gcn_temp_dir, train_csv_path, pred_out_path):
    """Train the motif GCN model.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    else:
        print("Running with cpu")

    adj        = pkl.load(open(f"{motif_gcn_temp_dir}/read.graph",'rb'))
    labels     = pkl.load(open(f"{motif_gcn_temp_dir}/read.label",'rb'))
    features   = pkl.load(open(f"{motif_gcn_temp_dir}/read.feature",'rb'))
    test_to_id = pkl.load(open(f"{motif_gcn_temp_dir}/read.dict",'rb'))
    train_idx  = pkl.load(open(f"{motif_gcn_temp_dir}/read.train_idx",'rb'))
    valid_idx  = pkl.load(open(f"{motif_gcn_temp_dir}/read.valid_idx",'rb'))
    test_idx   = pkl.load(open(f"{motif_gcn_temp_dir}/read.test_idx",'rb'))

    # load labels
    train_idx = np.array(train_idx)
    valid_idx = np.array(valid_idx)
    test_idx = np.array(test_idx)
    labels = np.array(labels)

    y_train = np.zeros(labels.shape)
    y_valid = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    train_mask = sample_mask(train_idx, labels.shape[0])
    valid_mask = sample_mask(valid_idx, labels.shape[0])
    test_mask = sample_mask(test_idx, labels.shape[0])

    y_train[train_mask] = labels[train_mask]
    y_valid[valid_mask] = labels[valid_mask]
    y_test[test_mask] = labels[test_mask]

    # load features
    features = sp.csc_matrix(features)

    print('adj:', adj.shape)
    print('features:', features.shape)
    print('y:', y_train.shape, y_valid.shape, y_test.shape) # y_val.shape, 
    print('mask:', train_mask.shape, valid_mask.shape, test_mask.shape) # val_mask.shape

    features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
    supports = preprocess_adj(adj)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda')
        num_classes = max(labels)+1
        train_label = torch.from_numpy(y_train).long().to(device)
        valid_label = torch.from_numpy(y_valid).long().to(device)
        test_label = torch.from_numpy(y_test).long().to(device)
        train_mask = torch.from_numpy(train_mask.astype(np.bool)).to(device)
        valid_mask = torch.from_numpy(valid_mask.astype(np.bool)).to(device)
        test_mask = torch.from_numpy(test_mask.astype(np.bool)).to(device)

        i = torch.from_numpy(features[0]).long().to(device)
        v = torch.from_numpy(features[1]).to(device)
        feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float().to(device)

        i = torch.from_numpy(supports[0]).long().to(device)
        v = torch.from_numpy(supports[1]).to(device)
        support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

    else:
        train_label = torch.from_numpy(y_train).long()
        num_classes = max(labels)+1
        train_mask = torch.from_numpy(train_mask.astype(np.bool))
        test_label = torch.from_numpy(y_test).long()
        test_mask = torch.from_numpy(test_mask.astype(np.bool))

        i = torch.from_numpy(features[0]).long()
        v = torch.from_numpy(features[1])
        feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float()

        i = torch.from_numpy(supports[0]).long()
        v = torch.from_numpy(supports[1])
        support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float()

    print('x :', feature)
    print('sp:', support)
    num_features_nonzero = feature._nnz()
    feat_dim = feature.shape[1]

    # calculate the loss weights
    all_data = pd.read_csv(train_csv_path, header=None)[0].values
    all_label = [i for i in range(num_classes)]
    num_each_class = np.array([np.count_nonzero(all_data == l) for l in all_label])
    class_weights = num_each_class.max()/ num_each_class

    def accuracy(out, mask):
        pred = np.argmax(out, axis = 1)
        mask_pred = np.array([pred[i] for i in range(len(labels)) if mask[i] == True])
        mask_label = np.array([labels[i] for i in range(len(labels)) if mask[i] == True])
        return np.sum(mask_label == mask_pred)/len(mask_pred)

    net = model.GCN(feat_dim, num_classes, num_features_nonzero)

    def evaluate():
        net.eval()
        with torch.no_grad():
            eval_out = net((feature, support))
            eval_loss = masked_loss(eval_out, valid_label, valid_mask, class_weights)
            eval_acc = accuracy(eval_out.detach().cpu().numpy(), valid_mask.detach().cpu().numpy())
        
        return eval_loss, eval_acc

    # load the model trained from stage one
    if args.trained_model == "None":
        pass
    else:
        net.load_state_dict(torch.load(args.trained_model))

    if torch.cuda.is_available():
        net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)#args.learning_rate

    val_losses = []
    # _ = net.train()
    for epoch in range(args.epochs):
        _ = net.train()
        # forward pass
        out = net((feature, support))
        #out = out[0]
        loss = masked_loss(out, train_label, train_mask, class_weights)
        loss += args.weight_decay * net.l2_loss()
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # calculating the acc
        _ = net.eval()
        out = net((feature, support))
        if torch.cuda.is_available():
            acc_train = accuracy(out.detach().cpu().numpy(), train_mask.detach().cpu().numpy())
        else:
            acc_train = accuracy(out.detach().numpy(), train_mask.detach().numpy())
        #acc_test = accuracy(out.detach().cpu().numpy(), test_mask.detach().cpu().numpy())
        # evaluation
        val_loss, val_acc = evaluate()
        val_losses.append(val_loss)

        print(f"Epoch: {epoch}, train_loss={loss.item()}, train_acc={acc_train}, val_loss={val_loss}, val_acc={val_acc}")
        
        # # early stopping
        # # if epoch > args.early_stopping and val_losses[-1] > np.mean(val_losses[-(args.early_stopping+1):-1]):
        # if epoch > args.early_stopping and val_losses[-1] > torch.mean(torch.stack(val_losses[-(args.early_stopping+1):-1])):
        #     print("Early stopping...")
        #     break

    torch.save(net.state_dict(), f"{motif_gcn_temp_dir}/motif_GCN_model.pt")

    net.eval()
    out = net((feature, support))
    if torch.cuda.is_available():
        out = out.cpu().detach().numpy()
    else:
        out = out.detach().numpy()
    out = softmax(out, axis=1)
    pred = np.argmax(out, axis = 1)

    mode = "testing"

    if num_classes == 18:
        pred_to_label = {0: 'Amarillovirales', 1: 'Articulavirales', 2: 'Bunyavirales', 3: 'Cryppavirales', 
                         4: 'Durnavirales', 5: 'Ghabrivirales', 6: 'Hepelivirales', 7: 'Martellivirales', 
                         8: 'Mononegavirales', 9: 'Nidovirales', 10: 'Nodamuvirales', 11: 'Patatavirales', 
                         12: 'Picornavirales', 13: 'Reovirales', 14: 'Sobelivirales', 15: 'Stellavirales', 
                         16: 'Tolivirales', 17: 'Tymovirales'}
    
    elif num_classes == 3:
        pred_to_label = {0: 'Bunyavirales', 1: 'Mononegavirales', 2: 'Picornavirales'}
    
    elif num_classes == 9:
        pred_to_label = {0: 'Cryppavirales', 1: 'Durnavirales', 2: 'Ghabrivirales', 3: 'Hepelivirales', 
                         4: 'Nidovirales', 5: 'Nodamuvirales', 6: 'Sobelivirales', 7: 'Stellavirales', 
                         8: 'Tolivirales'}
    
    # pred_out_path = os.path.join(work_dir, 'log/motif_GCN/prediction.csv')
    # os.makedirs(os.path.join(work_dir, 'log/motif_GCN'))
    with open(pred_out_path, 'w') as f_out:
        # _ = f_out.write("contig_names,prediction\n")
        for key in test_to_id.keys():
            _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]]) + "," + str(np.max(out[test_to_id[key]])) + "\n")
            # if labels[test_to_id[key]] == -1:
            #     _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]]) + "\n")
            # else:
                # _ = f_out.write(str(key) + "," + str(pred_to_label[labels[test_to_id[key]]]) + "\n")


def ana_motif_gcn_out(pred_rst, truth_csv, fam_path, out_dir, prc_pred_rst, final_out_dir, num_class=18):
    """Analyze the result of motif GCN.
    """
    if num_class == 18:
        pred_to_label = {0: 'Amarillovirales', 1: 'Articulavirales', 2: 'Bunyavirales', 3: 'Cryppavirales', 
                         4: 'Durnavirales', 5: 'Ghabrivirales', 6: 'Hepelivirales', 7: 'Martellivirales', 
                         8: 'Mononegavirales', 9: 'Nidovirales', 10: 'Nodamuvirales', 11: 'Patatavirales', 
                         12: 'Picornavirales', 13: 'Reovirales', 14: 'Sobelivirales', 15: 'Stellavirales', 
                         16: 'Tolivirales', 17: 'Tymovirales'}
    
    elif num_class == 3:
        pred_to_label = {0: 'Bunyavirales', 1: 'Mononegavirales', 2: 'Picornavirales'}
    
    elif num_class == 9:
        pred_to_label = {0: 'Cryppavirales', 1: 'Durnavirales', 2: 'Ghabrivirales', 3: 'Hepelivirales', 
                         4: 'Nidovirales', 5: 'Nodamuvirales', 6: 'Sobelivirales', 7: 'Stellavirales', 
                         8: 'Tolivirales'}

    label_to_pred = {j: i for i, j in pred_to_label.items()}

    # load the family dictionary
    pfam_dict = {}
    pfam_df = pd.read_csv(fam_path, sep=',', header=None)
    for order, order_label in zip(pfam_df[0], pfam_df[1]):
        pfam_dict[order] = order_label
    pfam_dict['other'] = num_class
    rev_pfam_dict = {v: k for k, v in pfam_dict.items()}

    ture_label = pd.read_csv(truth_csv, sep=',', header=None)[0]

    pred_dict_temp = {}
    pred_rst_df = pd.read_csv(pred_rst, sep=',', header=None)
    for s_id, order, soft_value in zip(pred_rst_df[0], pred_rst_df[1], pred_rst_df[2]):
        # print(s_id, order)
        # pred_dict[int(s_id.split('test_')[1])] = pfam_dict[order]
        test_index = int(s_id.split('_')[1])
        if test_index in pred_dict_temp:
            if soft_value < pred_dict_temp[test_index][1]:
                pred_dict_temp[test_index] = [order, soft_value]
        else:
            pred_dict_temp[test_index] = [order, soft_value]

    pred_dict = {k: pfam_dict[v[0]] for k, v in pred_dict_temp.items()}
    with open(f"{out_dir}/prediction.motif_gcn.csv", 'w') as mp:
        for k, v in pred_dict.items():
            mp.write(f"test_{k},{pred_to_label[v]}\n")

    pred_label = []
    # output the result of misclassified reads
    with open(f"{out_dir}/mis_pred.csv", 'w') as mp:
        for index in range(len(ture_label)):
            if index in pred_dict:
                pred_label.append(pred_dict[index])
                if pred_dict[index] != ture_label[index]:
                    mp.write(f"test_{index}\t{rev_pfam_dict[ture_label[index]]}\t{rev_pfam_dict[pred_dict[index]]}\n")
            else:
                pred_label.append(num_class)
                mp.write(f"test_{index}\t{rev_pfam_dict[ture_label[index]]}\tother\n")
    # print(set(pred_label[: 2977]))
     
    # plot the confusion matrix
    ana_utils.plot_cm(true_labels=ture_label, predicted_labels=pred_label, dict_file=fam_path, 
                      title="Confusion Matrix", save_dir=out_dir, out_name=out_dir.split('/')[-1])
    
    ana_utils.classification_report(true_labels=ture_label, predicted_labels=pred_label, 
                                    dict_file=fam_path, save_dir=out_dir, std_out=True)

    # merge the results of PRC and motif GCN
    final_pred_out_path = os.path.join(final_out_dir, 'prediction.csv')
    # os.makedirs(os.path.join(final_out_dir))
    prc_df = pd.read_csv(prc_pred_rst, sep=',', header=None)
    motif_GCN_df = pd.read_csv(f"{out_dir}/prediction.motif_gcn.csv", sep=',', header=None)
    with open(final_pred_out_path, 'w') as final_rst:
        for index, pred in zip(prc_df[0], prc_df[1]):
            final_rst.write(f"{index},{pred}\n")
        for index, pred in zip(motif_GCN_df[0], motif_GCN_df[1]):
            final_rst.write(f"{index},{pred}\n")

    ana_utils.analyze_predict_result(pred_rst=final_pred_out_path,
                                     truth_csv=truth_csv,
                                     fam_path=fam_path,
                                     out_dir=final_out_dir,
                                     num_class=num_class)


def extract_meme(orig_dir, target_dir):
    """Copy the meme files from original directory to the target directory.
    """

    os.makedirs(f"{target_dir}/motif")
    bmp_list = ['Bunyavirales', 'Mononegavirales', 'Picornavirales']
    file_list = ['meme.html', 'meme.txt', 'meme.xml']

    for order in bmp_list:
        os.makedirs(f"{target_dir}/motif/{order}")
        for f in file_list:
            shutil.copyfile(f"{orig_dir}/motif/{order}/{f}", f"{target_dir}/motif/{order}/{f}")


def choose_cand_hmmer(prot_file, prc_rst, hmmer_db="hmmer_database/RdRp.hmm"):
    """Use hmmer to choose the candidate reads.
    """
    prot_dir = os.path.dirname(prot_file)
    prc_rst_seq = list(pd.read_csv(prc_rst, header=None)[0])
    print(prc_rst_seq[: 10])
    with open(f"{prot_file}.rest", 'w') as pf:
        for s in SeqIO.parse(prot_file, 'fasta'):
            if s.id[: -2] not in prc_rst_seq:
                pf.write(f">{s.id}\n{s.seq}\n")

    os.system(f"hmmscan --tblout {prot_dir}/hmmer.tblout --cpu 6 -o {prot_dir}/hmmer.out {hmmer_db} {prot_file}.rest")


def parse_hmmer_tblout(hmmer_tblout, prot_path, hmmer_rst_out, un_hmmer_out):
    """Parse the hmmer result and extract the sequences.
    """
    pfam_name_order_dict = {"PF00946": 'Mononegavirales', 'PF06317': 'Bunyavirales', 
                            'PF04196': 'Bunyavirales', }
    
    prot_index = SeqIO.index(prot_path, 'fasta')
    with open(hmmer_rst_out, 'w') as hro:
        with open(un_hmmer_out, 'w') as uho:
            with open(hmmer_tblout) as ht:
                for l in ht:
                    if l[0] != '#':
                        l = l.split()
                        pfam_id = l[1].split('.')[0]
                        s_id = l[2]
                        if pfam_id in pfam_name_order_dict:
                            # hro.write(f"{s_id[: -2]},{pfam_name_order_dict[pfam_id]}\n")
                            hro.write(f"{s_id},{pfam_name_order_dict[pfam_id]}\n")
                        else:
                            uho.write(f">{prot_index[s_id].id}\n{prot_index[s_id].seq}\n")

    # # prot_index = SeqIO.index(prot_path, 'fasta')
    # uho_list = []
    # with open(hmmer_rst_out, 'w') as hro:
    #     with open(hmmer_tblout) as ht:
    #         for l in ht:
    #             if l[0] != '#':
    #                 l = l.split()
    #                 pfam_id = l[1].split('.')[0]
    #                 s_id = l[2]
    #                 if pfam_id in pfam_name_order_dict:
    #                     hro.write(f"{s_id[: -2]},{pfam_name_order_dict[pfam_id]}\n")
    #                 else:
    #                     uho_list.append(s_id)

    # with open(un_hmmer_out, 'w') as uho:
    #     for s in SeqIO.parse(prot_path, 'fasta'):
    #         if s.id in uho_list:
    #             uho.write(f">{s.id}\n{s.seq}\n")


def run_on_real(sim_reads_path, database_name, input_format, num_class):
    """Run the scripts on the real data.
    """
    work_dir = os.path.dirname(sim_reads_path)

    # # convert the fasta file into csv file
    # fasta2csv_pre(fasta_file=sim_reads_path,
    #               in_format='fasta', label=0)

    # # run blastx
    # os.makedirs(f"{work_dir}/log/blastx_mv")
    # run_blastx(database_path=f"{database_name}/data/diamond_temp/train", 
    #            train_csv=f"{database_name}/data/train.csv", 
    #            test_csv=f"{work_dir}/test_rdrp_sim.csv",
    #            fam_dict_path=f"{database_name}/data/fam_label.csv", 
    #            query_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta", 
    #            blast_out=f"{work_dir}/output.blastx", 
    #            pred_out_dir=f"{work_dir}/log/blastx_mv")

    # # get the protein of blastx result
    # get_blastx_right_frame(reads_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta", 
    #                        blastx_out=f"{work_dir}/output.blastx", 
    #                        blastx_prot_out=f"{work_dir}/output.blastx.prot",
    #                        blastx_nucl_out=f"{work_dir}/output.blastx.nucl")

    # # run sga to build the edge between testing edges
    # run_sga(reads_file=f"{work_dir}/test_rdrp_sim.csv.index.fasta",
    #         tar_f=f"{work_dir}/output.blastx.nucl")

    # # run PRC to predict the labels
    # os.makedirs(f"{work_dir}/log/prc")
    # run_prc(train_csv=f"{database_name}/data/train.csv", 
    #         test_csv=f"{work_dir}/test_rdrp_sim.csv", 
    #         blastx_out=f"{work_dir}/output.blastx", 
    #         sga_out=f"{work_dir}/test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg", 
    #         num_class=num_class,
    #         pred_dir=f"{work_dir}/log/prc", 
    #         fam_path=f"{database_name}/data/fam_label.csv", 
    #         num_iter=500)
    
    # # get the protein of the PRC result
    # get_prc_prot(reads_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta", 
    #              blast_prot_path=f"{work_dir}/output.blastx.prot",
    #              prc_rst=f"{work_dir}/log/prc/prediction.csv",
    #              sga_rst=f"{work_dir}/test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg")

    # # translate the reads into protein
    # translate2protein(input_file_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta", 
    #                   filter=True)
    translate2protein_transeq(input_file_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta")

    # # run hmmer to filter out the suspected non-RdRp reads
    # choose_cand_hmmer(prot_file=f"{work_dir}/test_rdrp_sim.csv.index.fasta.protein", 
    #                   prc_rst=f"{work_dir}/log/prc/prediction.csv",
    #                   hmmer_db="hmmer_database/RdRp.hmm")

    # # save the hmmer result
    # os.makedirs(f"{work_dir}/log/hmmer")
    # parse_hmmer_tblout(hmmer_tblout=f"{work_dir}/hmmer.tblout", 
    #                    prot_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta.protein", 
    #                    hmmer_rst_out=f"{work_dir}/log/hmmer/prediction.csv", 
    #                    un_hmmer_out=f"{work_dir}/rest_reads_prot.fasta")
    
    # # process the file before running fimo
    # process_before_fimo(all_trans_path=f"{work_dir}/rest_reads_prot.fasta", 
    #                     prc_prot_path=f"{work_dir}/prc_temp/prc_seq.prc.prot.fasta",
    #                     blastx_prot_path=f"{work_dir}/output.blastx.prot")

    # # run fimo to get the relation between motifs and reads
    # os.makedirs(f"{work_dir}/temp_fimo_out")
    # run_fimo(motif_dir=f"{database_name}/motif",
    #          fasta_file_path=f"{work_dir}/rest_reads_prot.fasta.full",
    #          out_dir=f"{work_dir}/temp_fimo_out", num_class=num_class)

    # # convert the fasta file to csv file to run the deep learning model
    # fasta2csv(fasta_file=f"{work_dir}/rest_reads_prot.fasta.full", 
    #           in_format='fasta', label=0)

    # # run CNN to get the embedding vector of the testing reads
    # run_CNN_pred(database_name=database_name, 
    #              test_csv=f"{work_dir}/rest_reads_prot.fasta.full.csv",
    #              num_class=num_class)

    # # run fimo and extract the fimo output
    # run_merge_fimo_out(train_csv=f"{database_name}/data/train.csv", 
    #                    fimo_out_path=f"{work_dir}/fimo.out", 
    #                    train_meme_dir=f"{database_name}/motif", 
    #                    test_meme_dir=f"{work_dir}/temp_fimo_out", 
    #                    thres=1e-7, num_class=num_class)

    # # build the graph based on the all the output
    # os.makedirs(f"{work_dir}/motif_gcn_temp")
    # build_graph(train_embed=f"{database_name}/data/train.FC1.npy", 
    #             valid_embed=f"{database_name}/data/validation.FC1.npy", 
    #             test_embed=f"{work_dir}/rest_reads_prot.fasta.full.csv.FC1.npy", 
    #             train_blastp_out=f"{database_name}/data/diamond_temp/train.diamond.tab",
    #             train_csv=f"{database_name}/data/train.csv", 
    #             valid_csv=f"{database_name}/data/validation.csv", 
    #             test_fasta=f"{work_dir}/rest_reads_prot.fasta.full", 
    #             fimo_out=f"{work_dir}/fimo.out", 
    #             blastx_out=f"{work_dir}/output.blastx",
    #             sga_out=f"{work_dir}/test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg",
    #             motif_GCN_temp_dir_path=f"{work_dir}/motif_gcn_temp", 
    #             pseudo_pred_path=f"{work_dir}/log/prc/prediction.csv",
    #             e_thres=1, num_class=num_class)

    # # train the motif_GCN model
    # os.makedirs(f"{work_dir}/log/motif_gcn")
    # train_motif_GCN(motif_gcn_temp_dir=f"{work_dir}/motif_gcn_temp", 
    #                 train_csv_path=f"{database_name}/data/train.csv", 
    #                 pred_out_path=f"{work_dir}/log/motif_gcn/prediction.csv")

    # analyze the result of the final results
    os.makedirs(f"{work_dir}/log/final")
    ana_motif_gcn_out(pred_rst=f"{work_dir}/log/motif_gcn/prediction.csv", 
                      truth_csv=f"{work_dir}/test_rdrp_sim.csv", 
                      fam_path=f"{database_name}/data/fam_label.csv", 
                      out_dir=f"{work_dir}/log/motif_gcn", 
                      prc_pred_rst=f"{work_dir}/log/prc/prediction.csv",
                      final_out_dir=f"{work_dir}/log/final",
                      num_class=num_class)


def merge_rst(log_dir):
    """Merge the results from multiple result.
    合并多个stage的结果。
    """
    hmmer_rst = pd.read_csv(f"{log_dir}/hmmer/prediction.csv", sep=',', header=None)
    prc_rst = pd.read_csv(f"{log_dir}/prc/prediction.csv", sep=',', header=None)
    motif_gcn_rst = pd.read_csv(f"{log_dir}/motif_gcn/prediction.csv", sep=',', header=None)

    out_file = open(f"{log_dir}/final/prediction.csv", 'w')
    rst_seq_list = []

    for index, pred in zip(hmmer_rst[0], hmmer_rst[1]):
        out_file.write(f"{index},{pred}\n")
        rst_seq_list.append(index)
    
    for index, pred in zip(prc_rst[0], prc_rst[1]):
        if index not in rst_seq_list:
            out_file.write(f"{index},{pred}\n")
            rst_seq_list.append(index)

    for index, pred in zip(motif_gcn_rst[0], motif_gcn_rst[1]):
        if index not in rst_seq_list:
            out_file.write(f"{index},{pred}\n")
            rst_seq_list.append(index)
    
    out_file.close()


if __name__ == "__main__":

    rdrpbin_args = rdrpbin_cmd()

    run_on_real(sim_reads_path=rdrpbin_args.input_file,
                database_name=rdrpbin_args.database, 
                input_format = rdrpbin_args.format, 
                num_class=3)
    
    # merge_rst(log_dir="/home/tangxubo/42/DeepRdRp/GCN/Jiang_virus/log")

    # parse_hmmer_tblout(hmmer_tblout=f"/home/tangxubo/42/DeepRdRp/GCN/psnv_rdrp/SRR9216246.psnv_rdrp.fasta.protein.hmmer", 
    #                    prot_path=f"/home/tangxubo/42/DeepRdRp/GCN/psnv_rdrp/SRR9216246.psnv_rdrp.fasta.protein", 
    #                    hmmer_rst_out=f"/home/tangxubo/42/DeepRdRp/GCN/psnv_rdrp/prediction.csv", 
    #                    un_hmmer_out=f"/home/tangxubo/42/DeepRdRp/GCN/psnv_rdrp/rest_reads_prot.fasta")

