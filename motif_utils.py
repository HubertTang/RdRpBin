from Bio import SeqIO
from multiprocessing import Pool
import os
import pandas as pd


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


def run_fimo(motif_dir, fasta_file_path, out_dir, num_class=18, threads=1):
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
    
    # num_procr = multiprocessing.cpu_count()
    pool = Pool(processes=threads)
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
    """Parse the hmmer result and extract the potential RdRp reads.
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


