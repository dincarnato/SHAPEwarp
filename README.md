![DRACO logo](http://www.incarnatolab.com/images/software/SHAPEwarp.png)
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

Morandi *et al*., submitted. SHAPE-guided RNA structure similarity search and motif discovery.

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
To compile the modules needed for kmer lookup:

```bash
cd SHAPEwarp
perl Makefile.PL
make
make test
```
SHAPEwarp builds on top of RNA Framework. To use SHAPEwarp, you must add the RNA Framework's ``lib/`` folder to your ``PERL5LIB`` environment variable:

```bash
export PERL5LIB=$PERL5LIB:/path/to/RNAFramework/lib
```