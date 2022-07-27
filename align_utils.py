from Bio import SeqIO
from collections import Counter
import os
import pandas as pd
import subprocess


def run_blastx(database_path, train_csv, fam_dict_path, 
               query_path, blast_out, pred_out_dir, threads):
    """Run blastx and analyze the result.
    """
    subprocess.run(f"diamond blastx -d {database_path} -q {query_path} -o {blast_out} -p {threads} -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qframe --very-sensitive ", shell=True)
    
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


def trans_spec_frame(dna_seq, spec_frame):
    """Translate all DNA sequences in fasta file
    """
    frame_dict = {1: 0, -1: 1, 2: 2, -2: 3, 3: 4, -3: 5}

    dna_seq = dna_seq.seq
    # use both fwd and rev sequences
    dna_seqs = [dna_seq, dna_seq.reverse_complement()]

    # generate all translation frames
    aa_seqs = [s[i:].translate(stop_symbol="X") for i in range(3) for s in dna_seqs]
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


# def run_sga(reads_file, tar_f, threads):
#     """Run SGA on all the reads to build the edges.
#     """
#     reads_dir = os.path.dirname(reads_file)
#     os.system(f"cd {reads_dir}\nsga preprocess {reads_file} > {reads_file}.sga.prep")
#     os.system(f"cd {reads_dir}\nsga index -t {threads} -a ropebwt {reads_file}.sga.prep")
#     os.system(f"cd {reads_dir}\nsga preprocess {tar_f} > {tar_f}.sga.prep")
#     os.system(f"cd {reads_dir}\nsga index -t {threads} -a ropebwt {tar_f}.sga.prep")
#     os.system(f"cd {reads_dir}\nsga overlap -t {threads} -m 80 -e 0.01 -d 2 -f {tar_f}.sga.prep --exhaustive {reads_file}.sga.prep")
#     os.system(f"cd {reads_dir}\ngunzip -f test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg.gz")
#     # remove the index files
#     # os.system(f"cd {reads_dir}\nrm {reads_file}.sga.prep")


def run_sga(reads_dir, threads):
    """Run SGA on all the reads to build the edges.
    """
    reads_file = "test_rdrp_sim.csv.index.fasta"
    tar_f = "output.blastx.nucl"
    if not os.path.exists(os.path.join(reads_dir, tar_f)):
        print(f"No RdRps detected")
        exit(0)
    subprocess.run(f"cd {reads_dir}\nsga preprocess {reads_file} > {reads_file}.sga.prep", shell=True)
    subprocess.run(f"cd {reads_dir}\nsga index -t {threads} -a ropebwt {reads_file}.sga.prep", shell=True)
    subprocess.run(f"cd {reads_dir}\nsga preprocess {tar_f} > {tar_f}.sga.prep", shell=True)
    subprocess.run(f"cd {reads_dir}\nsga index -t {threads} -a ropebwt {tar_f}.sga.prep", shell=True)
    subprocess.run(f"cd {reads_dir}\nsga overlap -t {threads} -m 80 -e 0.01 -d 2 -f {tar_f}.sga.prep --exhaustive {reads_file}.sga.prep", shell=True)
    subprocess.run(f"cd {reads_dir}\ngunzip -f test_rdrp_sim.csv.index.fasta.sga.output.blastx.nucl.sga.asqg.gz", shell=True)
    # remove the index files
    # os.system(f"cd {reads_dir}\nrm {reads_file}.sga.prep")


def make_diamond_db(fn_in, fn_out, cpu: int):
    """Build Diamond blastp database.
    """
    subprocess.run(f"diamond makedb --threads {cpu} --in {fn_in} -d {fn_out}", shell=True)


def run_diamond(aa_fp, db_fp, cpu: int, diamond_out_fn, e_value):
    """Run Diamond to blastp
    """
    subprocess.run(f"diamond blastp --threads {cpu} --sensitive -d {db_fp} -q {aa_fp} -o {diamond_out_fn} -e {e_value}", shell=True)


