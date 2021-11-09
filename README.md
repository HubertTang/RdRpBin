# RdRpBin

RdRpBin can identify and classify the RNA virus reads in metagenomic data. It uses RNA-dependent RNA polymerase gene as marker gene, and combines alignment-based strategies and graph-based learning models to do viral composition analysis and novel RNA virus discovery in metagenomic data.

The input file can be `fasta` or `fastq` format which contains the sequencing reads. Please note that since the software generates temporary data in running, please make sure there is enough free space on your hard disk, about 5 times the size of the input file.



# Required Dependencies

Recommend using Anaconda to install the following packages:

1. Python 3.x
2. Pytorch
3. DIAMOND
4. SGA
5. networkx



# How to run RdRpBin?

1. Create a directory `<input_dir>` containing your sequencing reads file.

2. Run the `main.py` script

   `python mian.py <input file path>`

   Optional arguments:

   -d, --database: the path of the database directory. Default: `RdRpBin_db`

   -f, --format: the format of the input file. Default: `fasta`

   -t, --thread: the number of threads

3. The identified RdRp reads will save in `<input_dir>/log/final`.