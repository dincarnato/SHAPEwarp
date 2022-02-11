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


## Reference

Morandi *et al*., submitted. SHAPE-guided RNA structure homology search and motif discovery.


## License

This program is free software, and can be redistribute and/or modified under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

Please see <http://www.gnu.org/licenses/> for more information.


## Prerequisites

- Linux system
- Rust and Cargo (Installation instructions: <https://doc.rust-lang.org/cargo/getting-started/installation.html>)
- RNA Framework v2.7.2 or higher (<https://github.com/dincarnato/RNAFramework/>)
- FFI::Platypus v1.56 or higher (<https://metacpan.org/pod/FFI::Platypus>)
- FFI::Platypus::Lang::Rust v0.09 or higher (<https://metacpan.org/pod/FFI::Platypus::Lang::Rust>)


## Installation

Clone the SHAPEwarp git repository:

```bash
git clone https://github.com/dincarnato/SHAPEwarp
```
This will create a "SHAPEwarp" folder.<br/>
To compile the modules needed for kmer lookup issue:

```bash
cd SHAPEwarp
perl Makefile.PL
make
make test
```
If the installation went fine, the expected output of the ``make test`` command should look like the following:

```bash
"/usr/bin/perl" -MFFI::Build::MM=cmd -e fbx_build
"/usr/bin/perl" -MFFI::Build::MM=cmd -e fbx_test
PERL_DL_NONLAZY=1 "/usr/bin/perl" "-MExtUtils::Command::MM" "-MTest::Harness" "-e" "undef *Test::Harness::Switches; test_harness(0, 'blib/lib', 'blib/arch')" t/*.t
t/basic.t .. ok
All tests successful.
Files=1, Tests=1,  1 wallclock secs ( 0.03 usr  0.00 sys +  0.54 cusr  0.04 csys =  0.61 CPU)
Result: PASS
```
SHAPEwarp builds on top of the [RNA Framework](https://github.com/dincarnato/RNAFramework/). To use SHAPEwarp, the ``lib/`` folder of the RNA Framework must be added to the ``PERL5LIB`` environment variable:

```bash
export PERL5LIB=$PERL5LIB:/path/to/RNAFramework/lib
```


## Testing the SHAPEwarp installation

To test SHAPEwarp on a small test dataset, issue the following command from within the SHAPEwarp install directory:

```bash
./SHAPEwarp -q t/query.txt -d t/ -o test_out -ow
```
The search will take &lt;1 minute and the expected output should look like the following:

```bash
Query   DB             Qstart  Qend  Dstart  Dend   Qseed    Dseed      Score    P-value    E-value

test    16S_Bsubtilis  7       170   916     1079   16-100   925-1009   173.76   4.83e-08   7.63e-06   !

test    16S_Bsubtilis  1       170   128     297    79-100   206-227    86.50    5.01e-04   0.08       ?
```