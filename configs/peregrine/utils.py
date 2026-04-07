peregrine_benchmarks = [
    "branch_storm",
    "collatz",
    "dhrystone",
    "linpack",
    "sieve",
    "sparse",
    "towers",
    "whetstone",
]

spec_benchmarks = [
    "505.mcf_r",
    "520.omnetpp_r",
    "523.xalancbmk_r",
    "541.leela_r",
    "548.exchange2_r",
    "531.deepsjeng_r",
    "557.xz_r",
    # "500.perlbench_r",
    "525.x264_r",
    "502.gcc_r",
]


spec_benchmark_args = {
    "505.mcf_r": {
        "binary": "mcf_r_base.peregrine-m64",
        "arguments": ["inp.in"],
    },
    "520.omnetpp_r": {
        "binary": "omnetpp_r_base.peregrine-m64",
        "arguments": ["-f", "omnetpp.ini", "-c", "General", "-r", "0"],
    },
    "523.xalancbmk_r": {
        "binary": "cpuxalan_r_base.peregrine-m64",
        "arguments": ["-v", "test.xml", "xalanc.xsl"],
    },
    "541.leela_r": {
        "binary": "leela_r_base.peregrine-m64",
        "arguments": ["test.sgf"],
    },
    "548.exchange2_r": {
        "binary": "exchange2_r_base.peregrine-m64",
        "arguments": ["0"],
    },
    "531.deepsjeng_r": {
        "binary": "deepsjeng_r_base.peregrine-m64",
        "arguments": ["test.txt"],
    },
    "557.xz_r": {
        "binary": "xz_r_base.peregrine-m64",
        "arguments": [
            "cpu2006docs.tar.xz",
            "4",
            "055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9"
            "713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae",
            "1548636",
            "1555348",
            "0",
        ],
    },
    # "500.perlbench_r": {
    #     "binary": "perlbench_r_base.peregrine-m64",
    #     "arguments": ["test.pl"],
    # },
    "525.x264_r": {
        "binary": "x264_r_base.peregrine-m64",
        "arguments": [
            "--dumpyuv",
            "50",
            "--frames",
            "156",
            "-o",
            "BuckBunny_New.264",
            "BuckBunny.yuv",
            "1280x720",
        ],
    },
    "502.gcc_r": {
        "binary": "cpugcc_r_base.peregrine-m64",
        "arguments": [
            "t1.c",
            "-O3",
            "-finline-limit=50000",
            "-o",
            "t1.opts-O3_-finline-limit_50000.s",
        ],
    },
}
