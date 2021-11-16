from Bio import SeqIO
import multiprocessing
from multiprocessing import Pool
import os
import shutil
import subprocess
from tqdm import tqdm


def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = iterator.__next__()
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch


def split_fasta(fasta_file, num_split=10):
    """Split original fasta file into several fasta files.
    """
    num_reads = int(subprocess.check_output("grep -c '^>' {}".format(fasta_file), shell=True).split()[0])
    num_per_split = num_reads // num_split
    
    record_iter = SeqIO.parse(fasta_file, "fasta")
    for i, batch in enumerate(batch_iterator(record_iter, num_per_split)):
        filename = f"{fasta_file}.{i}"
        with open(filename, "w") as handle:
            count = SeqIO.write(batch, handle, "fasta")
        print("Wrote %i records to %s" % (count, filename))

    if os.path.exists(f"{fasta_file}.{num_split}"):
        with open(f"{fasta_file}.{num_split+1}", 'w') as outfile:
            for fname in [f"{fasta_file}.{num_split-1}", f"{fasta_file}.{num_split}"]:
                with open(fname) as infile:
                    outfile.write(infile.read())

        os.remove(f"{fasta_file}.{num_split-1}")
        os.remove(f"{fasta_file}.{num_split}")
        os.rename(f"{fasta_file}.{num_split+1}", f"{fasta_file}.{num_split-1}")


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


def trans_6_frame_all(dna_fasta, filter_stop_codon=True):
    """Translate all DNA sequences in fasta file
    """
    print(f"Processing {dna_fasta.split('/').pop()} ... ...")

    num_reads = int(subprocess.check_output("grep -c '^>' {}".format(dna_fasta), shell=True).split()[0])
    
    protein_file = open(f"{dna_fasta}.protein",'w')
    for s in tqdm(SeqIO.parse(dna_fasta, 'fasta'), total=num_reads):
        dna_seq = s.seq
        # use both fwd and rev sequences
        dna_seqs = [dna_seq, dna_seq.reverse_complement()]
        # dna_seqs = [dna_seq, dna_seq[::-1]]

        # generate all translation frames
        aa_seqs = (s[i:].translate(stop_symbol="@") for i in range(3) for s in dna_seqs)

        if filter_stop_codon:
            for p_index, aa in enumerate(aa_seqs):
                # only keep the frame without stop codon
                if '@' not in aa:
                    protein_file.write(f">{s.id}_{p_index}\n{aa}\n")
                    
        else:
            for p_index, aa in enumerate(aa_seqs):
                temp_id = ""
                temp_seq = ""
                temp_num_stop = 100
                temp_has_out = False
                # only keep the frame without stop codon
                if '@' not in aa:
                    temp_has_out = True
                    protein_file.write(f">{s.id}_{p_index}\n{aa}\n")
                if str(aa).count('@') < temp_num_stop:
                    temp_seq = aa
                    temp_id = f"{s.id}_{p_index}"
                    temp_num_stop = str(aa).count('@')
                if not temp_has_out:
                    protein_file.write(f">{temp_id}\n{temp_seq}\n")

    protein_file.close()
    protein_list = [s for s in SeqIO.parse(f"{dna_fasta}.protein", 'fasta')]
    SeqIO.write(protein_list, f"{dna_fasta}.protein", 'fasta')


def translate2protein(input_file_path, filter=True):
    """Translate the DNA to protein.
    """
    work_dir = os.path.dirname(input_file_path)
    input_file_name = input_file_path.split('/')[-1]
    
    # translate and filter out the reads with stop codon
    num_procr = multiprocessing.cpu_count()
    split_fasta(fasta_file=input_file_path, num_split=num_procr)
    
    pool = Pool(processes=num_procr)
    for temp_id in range(num_procr):
        pool.apply_async(trans_6_frame_all, [f"{input_file_path}.{temp_id}", filter])
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


def transeq(dna_path):
    """Parse the transeq result, discard the protein with stop codons
    """
    subprocess.run(f"transeq {dna_path} {dna_path}.pep -frame=6 -trim -sformat pearson", shell=True)

    with open(f"{dna_path}.protein", 'w') as protein_file:
        for s in SeqIO.parse(f"{dna_path}.pep", 'fasta'):
            if "*" not in s:
                protein_file.write(f">{s.id}\n{s.seq}\n")


def translate2protein_transeq(input_file_path, threads):
    """Translate the DNA to protein.
    """
    work_dir = os.path.dirname(input_file_path)
    input_file_name = input_file_path.split('/')[-1]
    
    # translate and filter out the reads with stop codon
    # num_procr = multiprocessing.cpu_count()
    split_fasta(fasta_file=input_file_path, num_split=threads)

    pool = Pool(processes=threads)
    for temp_id in range(threads):
        pool.apply_async(transeq, [f"{input_file_path}.{temp_id}"])
    pool.close()
    pool.join()
    
    # move all the file into a temp directory
    os.makedirs(f"{work_dir}/temp")
    for i in range(threads):
        os.remove(f"{input_file_path}.{i}")
        os.remove(f"{input_file_path}.{i}.pep")
        shutil.move(f"{input_file_path}.{i}.protein", 
                    f"{work_dir}/temp/{input_file_name}.{i}.protein")
    merge_text(input_dir=f"{work_dir}/temp",
               out_path=f"{work_dir}/{input_file_name}.protein")
    
    shutil.rmtree(f"{work_dir}/temp")


# def fasta2csv(fasta_file, in_format, output_len, label):
def fasta2csv(fasta_file, in_format, label, out_len=66):
    """Convert fasta file into csv file.
    """
    with open(f"{fasta_file}.csv", 'w') as csv:
        for s in SeqIO.parse(fasta_file, in_format):
            # csv.write(f"{label},{str(s.seq)[: output_len].ljust(output_len, 'X')}\n")
            csv.write(f"{label},{str(s.seq)[: out_len]}\n")


