# Peregrine Config

## SPEC2017 Setup

### Install SPEC2017
To use the SPEC2017 benchmarks, you must setup SPEC2017 externally. Follow the [installaton instructions](https://www.spec.org/cpu2017/Docs/install-guide-unix.htm) on the SPEC website to install SPEC2017. Make sure to note the directory you install SPEC2017 to, which will be used for the `--specdir` command line argument for the `peregrine.py` config.

### Add the SPEC Peregrine Config
Add the `peregrine.cfg` (found in the [peregrine](https://github.com/chriswaligorski/peregrine/) repo) to build the benchmarks. Place it in the `config/` directory.

### Generate the benchmark run directories

```bash
cd SPEC2017-1-1-9
source shrc
runcpu --config=peregrine --action=runsetup --size=test 505.mcf_r 520.omnetpp_r 523.xalancbmk_r 541.leela_r 548.exchange2_r 531.deepsjeng_r 557.xz_r 500.perlbench_r 525.x264_r 502.gcc_r
```
