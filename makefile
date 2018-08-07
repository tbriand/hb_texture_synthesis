# Copyright 2009, 2010 IPOL Image Processing On Line http://www.ipol.im/
# Author: Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
# Author: Thibaud Briand, Jonathan Vacher
#
# Copying and distribution of this file, with or without
# modification, are permitted in any medium without royalty provided
# the copyright notice and this notice are preserved.  This file is
# offered as-is, without any warranty.

CSRC	= src/io_png.c src/hb_lib.c src/hb.c src/eig3.c src/filters.c src/periodic_component.c src/matching_hist.c src/mt19937ar.c src/bilinear_zoom.c 

SRC	= $(CSRC)
OBJ	= $(CSRC:.c=.o)
BIN	= hb

COPT	= -std=c99 -O3 -funroll-loops -fomit-frame-pointer -fopenmp



#-g
#

CFLAGS	= -ansi -Wextra -Werror -Wall  $(COPT)

default: $(BIN) 

%.o	: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(BIN)	: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) -lpng -lfftw3f -lfftw3f_threads -lm


.PHONY	: clean distclean
clean	:
	$(RM) $(OBJ)
distclean	: clean
	$(RM) $(BIN)
