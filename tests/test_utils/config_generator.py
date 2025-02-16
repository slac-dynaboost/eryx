def generate_base_config():
    """
    Creates a minimal working configuration for the OnePhonon model.
    """
    base_config = {
        "setup": {
            "pdb_path": "tests/pdbs/5zck.pdb",
            "root_dir": "test_output",
            "hsampling": [-4, 4, 1],
            "ksampling": [-17, 17, 1],
            "lsampling": [-29, 29, 1],
            "res_limit": 0,
            "batch_size": 10000,
            "n_processes": 8,
        },
        "OnePhonon": {
            "gnm_cutoff": 4.0,
            "gamma_intra": 1.0,
            "gamma_inter": 1.0,
            "expand_p1": True
        }
    }
    return base_config

def generate_edge_case_configs():
    """
    Creates a list of variant configuration dictionaries.
    Edge cases include parameter boundaries and error conditions.
    """
    configs = []

    # Variant 1: Low resolution limit
    variant1 = generate_base_config()
    variant1["setup"]["res_limit"] = 5.0
    configs.append(variant1)

    # Variant 2: High batch size and number of processes
    variant2 = generate_base_config()
    variant2["setup"]["batch_size"] = 20000
    variant2["setup"]["n_processes"] = 16
    configs.append(variant2)

    # Variant 3: Invalid config (simulate error condition)
    variant3 = generate_base_config()
    variant3["OnePhonon"]["gamma_inter"] = -1.0
    configs.append(variant3)

    return configs
