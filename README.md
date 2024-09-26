![SHAPEwarp logo](http://www.incarnatolab.com/images/software/SHAPEwarp.png)
<br />
## Introduction

The model-guided search for structurally-homologous RNAs is a non-trivial task, as it largely depends on the quality of the inferred structure model. When it comes to inferring RNA structures from chemical probing data, the challenges are numerous. Use of different chemical probes, or of different approaches for incorporating experimental reactivities as pseudo-free energy contributions can significantly affect the reliability of the inferred RNA structure model.

__SHAPEwarp__ is a sequence-agnostic method for the identification of structurally-similar RNA elements in a database of chemical probing-derived reactivity profiles. The approach used by SHAPEwarp is inspired by the BLAST algorithm and builds on top of two widely used methods for similarity search in time series data: Mueen's Algorithm for Similarity Search ([MASS](https://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html)) and dynamic time warping (DTW). 

For support requests, please post your questions to: <https://github.com/dincarnato/SHAPEwarp/issues>

For a complete documentation, please refer to: <https://shapewarp-docs.readthedocs.io/en/latest/>


## Author(s)

Edoardo Morandi (emorandi[at]rnaframework.com)<br/>
Danny Incarnato (dincarnato[at]rnaframework.com)<br/>


## References

Morandi *et al*., 2022. SHAPE-guided RNA structure homology search and motif discovery. Nature Communications (PMID: [35361788](https://pubmed.ncbi.nlm.nih.gov/35361788/))

Scholten *et al*., 2024. SHAPEwarp-web: sequence-agnostic search for structurally homologous RNA regions across databases of chemical probing data. Nucleic Acids Research (PMID: [38709889](https://pubmed.ncbi.nlm.nih.gov/38709889/))


## License

This program is free software, and can be redistribute and/or modified under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

Please see <http://www.gnu.org/licenses/> for more information.


## Prerequisites

- Linux system
- Rust and Cargo (Installation instructions: <https://doc.rust-lang.org/cargo/getting-started/installation.html>)


## Installation

```bash
$ git clone https://github.com/dincarnato/SHAPEwarp
$ cd SHAPEwarp

# Add to PKG_CONFIG_PATH the path to the directory containing RNAlib2.pc from the ViennaRNA package
$ export PKG_CONFIG_PATH=/path/to/dir/containing/RNAlib2.pc

$ export RUSTFLAGS=-Ctarget-cpu=native
$ cargo build --release
```

The SHAPEwarp executable will be located under ``target/release/``.<br/>


### Note for Mac OS X users:
To compile SHAPEwarp on Mac OS X, after having installed the ViennaRNA package, open the RNAlib2.pc file in a text editor and replace the ``-lstdc++`` flag with ``-lc++``.</br>


## Testing the SHAPEwarp installation

To test SHAPEwarp on a small test dataset, issue the following command from within the SHAPEwarp install directory:

```bash
target/release/SHAPEwarp --query test_data/query.txt --database test_data/test.db --output test_out --ow
```
The search will take less than 10 seconds, and the expected output should look like the following:

```bash
 query    db_entry       query_start  query_end  db_start  db_end  query_seed  db_seed  score    pvalue    evalue    status
 16S_750  16S_Bsubtilis  0            99         758       857     15-79       773-837  109.103  5.665e-8  1.003e-5  !
```
