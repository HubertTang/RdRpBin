# RdRpBin

RdRpBin can identify and classify the RNA virus reads in metagenomic data. It uses RNA-dependent RNA polymerase gene as marker gene, and combines alignment-based strategies and graph-based learning models to do viral composition analysis and novel RNA virus discovery in metagenomic data.

The input file can be `fasta` or `fastq` format which contains the sequencing reads. Please note that since the software generates temporary data in running, please make sure there is enough free space on your hard disk, about 2 times the size of the input file.



# Required Dependencies

Recommend using Anaconda to install the following packages:

1. Python 3.x
2. Pytorch
3. diamond
4. sga
5. networkx
6. emboss
7. hmmer
8. biopython
9. tqdm

The Anaconda environment has been save in `environment.yml`, you can use following command to install the environment:

```bash
conda env create -f environment.yml
conda activate RdRpBin
```



# How to run RdRpBin?

1. Download the reference dataset and taxonomy files from [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/xubotang2-c_my_cityu_edu_hk/EjViwW1ComFDjo7TeRCN9-4Bqv3wpRAN9oXyWYWNJ1L9gw?e=ClO00i)  (or  [百度网盘/Baidu Netdisk](https://pan.baidu.com/s/1NeOjjicVL5KChp4T5ArlyQ)  code: 5gv5) and uncompress them in the same directory with `main.py`.

2. Create an empty directory `<input_dir>` and put the sequencing reads file `<input_reads>` into this directory.

3. Run the `main.py` script

   `python mian.py <path of input_reads>`

   Optional arguments:

   -f, --format: the format of the input file. Default: `fasta`.

   -t, --thread: the number of threads. Default: 1.

   --learning_rate: the learning rate of GCN. Default: 0.01.

   --epochs: the number of GCN training epochs. Default: 50.

   --hidden: the size of the hidden vector. Default: 64.

   --weight_decay: the weight decay parameter. Default: 5e-4.

   --no_gcn: run RdRpBin without running GCN. (This argument can reduce the running time with a little decrease in recall). Default: no.

4. The identified RdRp reads will save in `<input_dir>/RdRp_reads`.

# Citation
Tang, X., Shang, J., & Sun, Y. (2022). RdRp-based sensitive taxonomic classification of RNA viruses for metagenomic data. Briefings in bioinformatics, 23(2), bbac011.
