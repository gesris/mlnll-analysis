#!/bin/bash

source /work/gristo/mlnll-analysis/utils/setup_lcg.sh

### Drawing data to .csv

# 4 bins
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/4_bins_mig01x100_nosysimpl_nosys_shapes/ nosysimpl_nosys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/4_bins_mig01x100_nosysimpl_shapes/ nosysimpl_sys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/4_bins_mig01x100_wsysimpl_nosys_shapes/ wsysimpl_nosys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/4_bins_mig01x100_wsysimpl_shapes/ wsysimpl_sys


# 8 bins
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/8_bins_mig01x100_nosysimpl_nosys_shapes/ nosysimpl_nosys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/8_bins_mig01x100_nosysimpl_shapes/ nosysimpl_sys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/8_bins_mig01x100_wsysimpl_nosys_shapes/ wsysimpl_nosys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/8_bins_mig01x100_wsysimpl_shapes/ wsysimpl_sys

# 16 bins
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/16_bins_mig01x100_nosysimpl_nosys_shapes/ nosysimpl_nosys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/16_bins_mig01x100_nosysimpl_shapes/ nosysimpl_sys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/16_bins_mig01x100_wsysimpl_nosys_shapes/ wsysimpl_nosys
#python scan_custom_write_data.py /work/gristo/mlnll-analysis/output/16_bins_mig01x100_wsysimpl_shapes/ wsysimpl_sys


### Creating plots from data

# 4 bins
python scan_custom.py /work/gristo/mlnll-analysis/output/4_bins_mig01x100_nosysimpl_shapes/ \
/work/gristo/mlnll-analysis/output/4_bins_mig01x100_nosysimpl_nosys_shapes/ \
/work/gristo/mlnll-analysis/output/4_bins_mig01x100_wsysimpl_shapes/ \
/work/gristo/mlnll-analysis/output/4_bins_mig01x100_wsysimpl_nosys_shapes/ \
4_bins \
mig01x100

# 8 bins
python scan_custom.py /work/gristo/mlnll-analysis/output/8_bins_mig01x100_nosysimpl_shapes/ \
/work/gristo/mlnll-analysis/output/8_bins_mig01x100_nosysimpl_nosys_shapes/ \
/work/gristo/mlnll-analysis/output/8_bins_mig01x100_wsysimpl_shapes/ \
/work/gristo/mlnll-analysis/output/8_bins_mig01x100_wsysimpl_nosys_shapes/ \
8_bins \
mig01x100

# 16 bins
python scan_custom.py /work/gristo/mlnll-analysis/output/16_bins_mig01x100_nosysimpl_shapes/ \
/work/gristo/mlnll-analysis/output/16_bins_mig01x100_nosysimpl_nosys_shapes/ \
/work/gristo/mlnll-analysis/output/16_bins_mig01x100_wsysimpl_shapes/ \
/work/gristo/mlnll-analysis/output/16_bins_mig01x100_wsysimpl_nosys_shapes/ \
16_bins \
mig01x100





