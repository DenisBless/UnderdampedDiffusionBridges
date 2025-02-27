def get_init_fn(alg_name):
    if alg_name == 'ula_ud':
        from algorithms.underdamped.ula_ud import init_ula_ud
        return init_ula_ud

    if alg_name == 'ula':
        from algorithms.overdamped.ula import init_ula
        return init_ula

    elif alg_name == 'mcd_ud':
        from algorithms.underdamped.mcd_ud import init_mcd_ud
        return init_mcd_ud

    elif alg_name == 'mcd':
        from algorithms.overdamped.mcd import init_mcd
        return init_mcd

    elif alg_name == 'cmcd_ud':
        from algorithms.underdamped.cmcd_ud import init_cmcd_ud
        return init_cmcd_ud

    elif alg_name == 'cmcd':
        from algorithms.overdamped.cmcd import init_cmcd
        return init_cmcd

    elif alg_name == 'dbs_ud':
        from algorithms.underdamped.dbs_ud import init_dbs_ud
        return init_dbs_ud

    elif alg_name == 'dbs':
        from algorithms.overdamped.dbs import init_dbs
        return init_dbs

    elif alg_name == 'dis_ud':
        from algorithms.underdamped.dis_ud import init_dis_ud
        return init_dis_ud

    elif alg_name == 'dis':
        from algorithms.overdamped.dis import init_dis
        return init_dis

    else:
        raise ValueError(f'No algorithm named {alg_name}.')
