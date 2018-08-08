# The Heeger & Bergen pyramid based texture synthesis algorithm

This code is distributed under the terms of the GPLv3 license.

Copyright (C) 2012, Thibaud Briand, ENS Cachan <thibaud.briand@ens-cachan.fr>
Copyright (C) 2012, Jonathan Vacher, ENS Cachan <jvacher@ens-cachan.fr>

README file for hb.c v1.1: Heeger & Bergen texture synthesis


Content of this README file:
1. Authors
2. URLs
3. Requirements
4. Folders
5. Compilation
6. Usage
7. Test
8. Releases
9. Generate HTML Documentation

1. Authors

Thibaud BRIAND   ; <thibaud.briand@ens-cachan.fr>
Jonathan VACHER  ; <jvacher@ens-cachan.fr>

2. URLs

This source code is an ANSI C implementation of
the Heeger & Bergen texture synthesis algorithm described in the IPOL webpage
http://www.ipol.im/pub/art/2014/79/


The last release should be available at: https://github.com/tbriand/hb_texture_synthesis


This code is provided with an HTML documentation produced by doxygen:


3. Requirements:
 - ANSI C compiler
 - getopt
 - libpng
 - libfftw3
 - (optional) doxygen (to reproduce the HTML documentation)

4. Folders

The archive heegerbergen_1.00.zip has tree subfolders:
 - data/ contains an example texture
 - src/ contains the ANSI C source code
 - doc/ contains the necessary files to reproduce the HTML documentation
 with doxygen (see paragraph 9)

5. Compilation:

Execute the provided Makefile from the main folder (function make).

6. Usage: (displayed in executing ./hb -s scales -k orientations -i iterations
-n noise.png -g seed -x row_ratio -y colum_ratio
-e edge_handling -r smooth -p crop input.png output.png)

Required parameters:
 input.png   :   name of the input PNG image
 output.png  :   name of the output PNG image

Optionnal parameters:
scales          :   int to specify the number of pyramid scales (by default 4)
orientations    :   int>=2 to specify the number of orientations (by default 4)
iterations      :   int>=1 to specify the number of iterations (by default 5)
noise           :   name of the noise PNG image
seed            :   unsigned int to specify the seed for the random number
                    generator (seed = time(NULL) by default)
row_ratio       :   int>=1 to specify the row ratio of extension (by default 1)
column_ratio    :   int>=1 to specify the column ratio of
		    extension (by default 1)
edge_handling   :   int to specify the type of edge handling :
		    periodic component (0) and mirror symmetrization (1)
smooth          :   int=1 to add the smooth component (by default 0)
crop 		:   int=1 to write the cropped input
		    (PNG image called input_cropped.png)

7. Test:
 To test the module run:
    ./hb -g 0 data/sample.png output.png
 output.png should be the same as the image sample_hb.png
 provided with the source code.

8. Releases:

 - version 1.1:
		second submitted version

9. Generate HTML Documentation

This code is provided with an HTML documentation produced by doxygen:

If doxygen is installed on your system, you can generate
the HTML documentation by running
doxygen hb_doxygen.conf
from the subdirectory doc/. The main page of the documentation is the file
doc/dochmtl/index.html
