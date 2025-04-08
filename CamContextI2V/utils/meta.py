import textwrap

DEFAULT_MACHINE = "marvin"
DEFAULT_CONFIG_OUTPUT = 'configs/generated'
DEFAULT_DEPLOY = False
DEFAULT_META_FILE = ".exp_meta.yaml"
DEFAULT_PARTITION = { # Only required for slurm runs
    "example_host": "example_partition",
}

HOSTNAMES = {
    "example_host": "user@example-domain.de",
}

SOURCE_PATH = {
    "example_host": "<source directory>",
}
EXPERIMENT_DIRECTORIES = {
    "example_host": "<experiment directory>",

}
PARTITION_SETUPS = {
    "example_host": {
        "example_partition": {
            "gpu_per_node": 8,
            "num_gpu": 8,
            "batch_size": 3,
            "accumulate_grad_batches": 3,
            "run_time": "71:59:59"
        }
    },
}
DATA_DIRECTORIES = {

    "example_host": {
        "val": {
            "data_dir":  "<path to the root directory holding the videos>",
            "meta_path": "<path containing the meta information of each video>",
            "meta_list": "<txt file containing all valid video names>",
            "caption_file": "<json file containing all captions>"
        },
        "train": {
            "data_dir":  "<path to the root directory holding the videos>",
            "meta_path": "<path containing the meta information of each video>",
            "meta_list": "<txt file containing all valid video names>",
            "caption_file": "<json file containing all captions>"
        },
    },
}
ENVIRONMENT_SETUP = {

    "example_host": textwrap.dedent("""\
        <bash commands to load the python environment>
    """)
}
