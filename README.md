![SHAPEwarp logo](http://www.incarnatolab.com/images/software/SHAPEwarp.png)
<br />
## Author(s)

Edoardo Morandi (emorandi[at]rnaframework.com)<br/>
Danny Incarnato (dincarnato[at]rnaframework.com)<br/>


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


## Testing your SHAPEwarp installation

To test if SHAPEwarp is properly working, from within the SHAPEwarp install directory issue the following command:

```bash
./SHAPEwarp -q t/query.txt -d t/ -o test_out -ow
```
The expected output should look like the following:

```bash
Query   DB             Qstart  Qend  Dstart  Dend   Qseed    Dseed      Score    P-value    E-value

test    16S_Bsubtilis  7       170   916     1079   16-100   925-1009   173.76   4.83e-08   7.63e-06   !

test    16S_Bsubtilis  1       170   128     297    79-100   206-227    86.50    5.01e-04   0.08       ?
```