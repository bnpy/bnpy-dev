from bnpy.alloc_models import hierarchical_model as hmod

def create_and_initialize_hierarchical_model_for_dataset(
        mod_list,
        dataset,
        **input_kwargs):
    hyper_kwargs = hmod.make_hyper_kwargs(mod_list, **input_kwargs)
    init_kwargs = hmod.make_init_kwargs(mod_list, **input_kwargs)
    local_step_kwargs = hmod.make_local_step_kwargs(mod_list, **input_kwargs)

    # Now create the hyper parameters
    hyper_dict = hmod.init_hyper_params(
        mod_list, dataset, **hyper_kwargs)

    # Now create the global parameters
    param_dict, info_dict = hmod.init_global_params(
        mod_list, dataset, hyper_dict, **init_kwargs)
    return param_dict, hyper_dict, info_dict, local_step_kwargs