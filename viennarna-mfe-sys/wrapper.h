#include <constraints/SHAPE.h>
#include <constraints/soft.h>
#include <fold_compound.h>
#include <mfe.h>
#include <mfe_window.h>

struct vrna_hc_depot_s {
  unsigned int strands;
  size_t *up_size;
  struct hc_nuc **up;
  size_t *bp_size;
  struct hc_basepair **bp;
};

struct hc_nuc {
  int direction;
  unsigned char context;
  unsigned char nonspec;
};

struct hc_basepair {
  size_t list_size;
  size_t list_mem;
  unsigned int *j;
  unsigned int *strand_j;
  unsigned char *context;
};
