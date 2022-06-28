import align_utils
from Bio import SeqIO
import argparse
import cnn_utils
import os
import pandas as pd
import graph_utils
import motif_utils
import motif_gcn_utils
import seq_utils

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

    # parser.add_argument(
    #     '-d', "--database",
    #     type=str,
    #     default="RdRpBin_db",
    #     help="Database."
    #     )

    parser.add_argument(
        '-f', "--format",
        default="fasta",
        type=str,
        help="Format of input file (fasta (default), fastq)")

    parser.add_argument(
        '-t', "--thread",
        default=1,
        type=int,
        help="The number of threads")

    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.01)
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50)
    
    parser.add_argument(
        '--hidden', 
        type=int, 
        default=64)
        
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=5e-4)
    
    parser.add_argument(
        '--no_gcn', 
        type=str, 
        default='no',
        help="Run RdRpBin without running GCN (yes/ no)")
    
    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='RdRpBin_1.1'
    )

    rdrpbin_args = parser.parse_args()

    assert (rdrpbin_args.format in ['fasta', 'fastq'])

    return rdrpbin_args


def merge_rst(log_dir):
    """Merge the results from multiple result.
    合并多个stage的结果。
    """
    os.makedirs(f"{log_dir}/final")
    out_file = open(f"{log_dir}/final/prediction.csv", 'w')
    rst_seq_list = []
    
    try:
        hmmer_rst = pd.read_csv(f"{log_dir}/hmmer/prediction.csv", sep=',', header=None)
        for index, pred in zip(hmmer_rst[0], hmmer_rst[1]):
            out_file.write(f"{index},{pred}\n")
            rst_seq_list.append(index)
    except pd.errors.EmptyDataError:
        pass
    
    try:
        prc_rst = pd.read_csv(f"{log_dir}/prc/prediction.csv", sep=',', header=None)
        for index, pred in zip(prc_rst[0], prc_rst[1]):
            if index not in rst_seq_list:
                out_file.write(f"{index},{pred}\n")
                rst_seq_list.append(index)
    except pd.errors.EmptyDataError:
        pass
    
    try:
        motif_gcn_rst = pd.read_csv(f"{log_dir}/motif_gcn/prediction.csv", sep=',', header=None)
        for index, pred in zip(motif_gcn_rst[0], motif_gcn_rst[1]):
            if index not in rst_seq_list:
                out_file.write(f"{index},{pred}\n")
                rst_seq_list.append(index)
    except pd.errors.EmptyDataError:
        pass
    
    out_file.close()


def merge_rst_s12(log_dir):
    """Merge the results from multiple result.
    合并多个stage的结果。
    """
    os.makedirs(f"{log_dir}/final")
    out_file = open(f"{log_dir}/final/prediction.csv", 'w')
    rst_seq_list = []
        
    try:
        prc_rst = pd.read_csv(f"{log_dir}/prc/prediction.csv", sep=',', header=None)
        for index, pred in zip(prc_rst[0], prc_rst[1]):
            if index not in rst_seq_list:
                out_file.write(f"{index},{pred}\n")
                rst_seq_list.append(index)
    except pd.errors.EmptyDataError:
        pass
        
    out_file.close()


def output_reads(seq_path, format, pred_path):
    """Output the identified reads.
    """
    work_dir = os.path.dirname(os.path.abspath(seq_path))
    os.makedirs(f"{work_dir}/RdRp_reads")
    
    pred_dict = {}
    with open(pred_path) as pp:
        for l in pp:
            l = l.strip().split(',')
            pred_dict[l[0].split('_test')[0]] = l[1]

    for s in SeqIO.parse(seq_path, format):
        if s.id in pred_dict:
            with open(f"{work_dir}/RdRp_reads/{pred_dict[s.id]}.fasta", 'a') as pr:
                pr.write(f">{s.id}\n{s.seq}\n")


def run_on_real(args, sim_reads_path, database_name, input_format, num_class, num_thread=1):
    """Run the scripts on the real data.
    """
    work_dir = os.path.dirname(os.path.abspath(sim_reads_path))

    # convert the fasta file into csv file
    seq_utils.fasta2csv_pre(fasta_file=sim_reads_path,
                  in_format=input_format, label=0)

    # run blastx
    os.makedirs(f"{work_dir}/log/blastx_mv")
    align_utils.run_blastx(database_path=f"{database_name}/data/diamond_temp/train", 
               train_csv=f"{database_name}/data/train.csv", 
               fam_dict_path=f"{database_name}/data/fam_label.csv", 
               query_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta", 
               blast_out=f"{work_dir}/output.blastx", 
               pred_out_dir=f"{work_dir}/log/blastx_mv",
               threads=num_thread)

    # get the protein of blastx result
    align_utils.get_blastx_right_frame(reads_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta", 
                           blastx_out=f"{work_dir}/output.blastx", 
                           blastx_prot_out=f"{work_dir}/output.blastx.prot",
                           blastx_nucl_out=f"{work_dir}/output.blastx.nucl")

    # run sga to build the edge between testing edges
    # align_utils.run_sga(reads_file=f"{work_dir}/test_rdrp_sim.csv.index.fasta",
    #         tar_f=f"{work_dir}/output.blastx.nucl", threads=num_thread)
    align_utils.run_sga(reads_dir=work_dir, threads=num_thread)

    # run PRC to predict the labels
    os.makedirs(f"{work_dir}/log/prc")
    graph_utils.run_prc(train_csv=f"{database_name}/data/train.csv", 
            test_csv=f"{work_dir}/test_rdrp_sim.csv", 
            blastx_out=f"{work_dir}/output.blastx", 
            sga_out=f"{work_dir}/test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg", 
            num_class=num_class,
            pred_dir=f"{work_dir}/log/prc", 
            num_iter=500)

    if args.no_gcn.lower() in ('yes', 'true', 't', 'y', '1'):
        merge_rst_s12(log_dir=f"{work_dir}/log")
        
        output_reads(seq_path=sim_reads_path, 
                    format=input_format, 
                    pred_path=f"{work_dir}/log/final/prediction.csv")
        return
    
    # get the protein of the PRC result
    graph_utils.get_prc_prot(reads_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta", 
                 blast_prot_path=f"{work_dir}/output.blastx.prot",
                 prc_rst=f"{work_dir}/log/prc/prediction.csv",
                 sga_rst=f"{work_dir}/test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg")

    # translate the reads into protein
    seq_utils.translate2protein_transeq(input_file_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta",
                              threads=num_thread)

    # run hmmer to filter out the suspected non-RdRp reads
    motif_utils.choose_cand_hmmer(prot_file=f"{work_dir}/test_rdrp_sim.csv.index.fasta.protein", 
                      prc_rst=f"{work_dir}/log/prc/prediction.csv",
                      hmmer_db="hmmer_database/RdRp.hmm")

    # save the hmmer result
    os.makedirs(f"{work_dir}/log/hmmer")
    motif_utils.parse_hmmer_tblout(hmmer_tblout=f"{work_dir}/hmmer.tblout", 
                       prot_path=f"{work_dir}/test_rdrp_sim.csv.index.fasta.protein", 
                       hmmer_rst_out=f"{work_dir}/log/hmmer/prediction.csv", 
                       un_hmmer_out=f"{work_dir}/rest_reads_prot.fasta")
    
    # process the file before running fimo
    motif_utils.process_before_fimo(all_trans_path=f"{work_dir}/rest_reads_prot.fasta", 
                        prc_prot_path=f"{work_dir}/prc_temp/prc_seq.prc.prot.fasta",
                        blastx_prot_path=f"{work_dir}/output.blastx.prot")

    # run fimo to get the relation between motifs and reads
    os.makedirs(f"{work_dir}/temp_fimo_out")
    motif_utils.run_fimo(motif_dir=f"{database_name}/motif",
             fasta_file_path=f"{work_dir}/rest_reads_prot.fasta.full",
             out_dir=f"{work_dir}/temp_fimo_out", num_class=num_class,
             threads=num_thread)

    # convert the fasta file to csv file to run the deep learning model
    seq_utils.fasta2csv(fasta_file=f"{work_dir}/rest_reads_prot.fasta.full", 
              in_format='fasta', label=0)

    # run CNN to get the embedding vector of the testing reads
    cnn_utils.run_CNN_pred(database_name=database_name, 
                 test_csv=f"{work_dir}/rest_reads_prot.fasta.full.csv",
                 num_class=num_class, threads=num_thread)

    # run fimo and extract the fimo output
    motif_utils.run_merge_fimo_out(train_csv=f"{database_name}/data/train.csv", 
                       fimo_out_path=f"{work_dir}/fimo.out", 
                       train_meme_dir=f"{database_name}/motif", 
                       test_meme_dir=f"{work_dir}/temp_fimo_out", 
                       thres=1e-7, num_class=num_class)

    # build the graph based on the all the output
    os.makedirs(f"{work_dir}/motif_gcn_temp")
    graph_utils.build_graph(train_embed=f"{database_name}/data/train.FC1.npy", 
                valid_embed=f"{database_name}/data/validation.FC1.npy", 
                test_embed=f"{work_dir}/rest_reads_prot.fasta.full.csv.FC1.npy", 
                train_blastp_out=f"{database_name}/data/diamond_temp/train.diamond.tab",
                train_csv=f"{database_name}/data/train.csv", 
                valid_csv=f"{database_name}/data/validation.csv", 
                test_fasta=f"{work_dir}/rest_reads_prot.fasta.full", 
                fimo_out=f"{work_dir}/fimo.out", 
                blastx_out=f"{work_dir}/output.blastx",
                sga_out=f"{work_dir}/test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg",
                motif_GCN_temp_dir_path=f"{work_dir}/motif_gcn_temp", 
                pseudo_pred_path=f"{work_dir}/log/prc/prediction.csv",
                e_thres=1, num_class=num_class)

    # train the motif_GCN model
    os.makedirs(f"{work_dir}/log/motif_gcn")
    motif_gcn_utils.train_motif_GCN(args=args,
                    motif_gcn_temp_dir=f"{work_dir}/motif_gcn_temp", 
                    train_csv_path=f"{database_name}/data/train.csv", 
                    pred_out_path=f"{work_dir}/log/motif_gcn/prediction.csv")

    merge_rst(log_dir=f"{work_dir}/log")

    output_reads(seq_path=sim_reads_path, 
                 format=input_format, 
                 pred_path=f"{work_dir}/log/final/prediction.csv")


if __name__ == "__main__":

    rdrpbin_args = rdrpbin_cmd()
    print(rdrpbin_args)

    # n_class = 0
    # if rdrpbin_args.database == 'RdRpBin_db':
    #     n_class = 18

    n_class = 18

    run_on_real(args = rdrpbin_args,
                sim_reads_path=rdrpbin_args.input_file,
                database_name=rdrpbin_args.database, 
                input_format = rdrpbin_args.format, 
                num_class=n_class, num_thread=rdrpbin_args.thread)
    
