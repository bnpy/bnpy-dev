def create_and_initialize_hierarchical_model_for_dataset(
        mod_list,
        dataset,
        **input_kwargs):
    hyper_kwargs = make_hyper_kwargs(mod_list, **input_kwargs)
    init_kwargs = make_init_kwargs(mod_list, **input_kwargs)
    local_step_kwargs = make_local_step_kwargs(mod_list, **input_kwargs)

    # Now create the hyper parameters
    hyper_dict = init_hyper_params(
        mod_list, dataset, **hyper_kwargs)

    # Now create the global parameters
    param_dict, info_dict = init_global_params(
        mod_list, dataset, hyper_dict, **init_kwargs)

    extra_kwargs = dict()
    for key in input_kwargs:
        if key in hyper_kwargs or key in init_kwargs:
            continue
        if key in local_step_kwargs:
            continue
        extra_kwargs[key] = input_kwargs[key]
    return param_dict, hyper_dict, info_dict, local_step_kwargs, extra_kwargs

def make_hyper_kwargs(mod_list, **input_kwargs):
    kwargs = dict()
    for mod in mod_list:
        kwargs.update(mod.default_hyper_kwargs)
    for key in kwargs:
        if key in input_kwargs:
            kwargs[key] = input_kwargs[key]
    return kwargs

def make_init_kwargs(mod_list, **input_kwargs):
    kwargs = dict()
    for mod in mod_list:
        kwargs.update(mod.default_init_kwargs)
    for key in kwargs:
        if key in input_kwargs:
            kwargs[key] = input_kwargs[key]
    return kwargs

def make_local_step_kwargs(mod_list, **input_kwargs):
    kwargs = dict()
    for mod in mod_list:
        kwargs.update(mod.default_local_step_kwargs)
    for key in kwargs:
        if key in input_kwargs:
            kwargs[key] = input_kwargs[key]
    return kwargs

def init_hyper_params(mod_list, data, **hyper_kwargs):
    hyper_dict = dict()
    for mod in mod_list:
        hyper_dict.update(
            mod.init_hyper_params(data, **hyper_kwargs))
    return hyper_dict

def init_global_params(mod_list, data, hyper_dict, **init_kwargs):
    param_dict = dict()
    info_dict = dict()
    for mod in mod_list:
        cur_param_dict, cur_info_dict = \
            mod.init_global_params(data, hyper_dict, **init_kwargs)
        param_dict.update(cur_param_dict)
        info_dict.update(cur_info_dict)
    return param_dict, info_dict

def calc_local_params(
        mod_list, data, param_dict,
        hyper_dict=None, LP=None, **local_step_kwargs):
    if LP is None:
        LP = dict()
    for mod in mod_list:
        if hasattr(mod, 'calc_log_proba'):
            LP['log_proba_NK'] = mod.calc_log_proba(
                data, param_dict, **local_step_kwargs)
            # TODO sum across multiple obsmodels??
        else:
            LP = mod.calc_local_params(
                data, param_dict,
                hyper_dict=hyper_dict,
                LP=LP,
                **local_step_kwargs)
    return LP

def summarize_local_params_for_update(mod_list, data, LP, **kwargs):
    SS = dict()
    for mod in mod_list:
        cur_SS = \
            mod.summarize_local_params_for_update(
                data, LP, **kwargs)
        SS.update(cur_SS)
    return SS

def summarize_local_params_for_loss(mod_list, data, LP, **kwargs):
    SS_for_loss = dict()
    for mod in mod_list:
        if hasattr(mod, 'summarize_local_params_for_loss'):
            cur_SS_for_loss = \
                mod.summarize_local_params_for_loss(
                    data, LP, **kwargs)
            SS_for_loss.update(cur_SS_for_loss)
    return SS_for_loss

def calc_loss_from_summaries(
        mod_list, param_dict, hyper_dict, SS, SS_for_loss=None):
    loss = 0
    for mod in mod_list:
        loss += mod.calc_loss_from_summaries(
            param_dict, hyper_dict, SS, SS_for_loss)
    return loss

def update_global_params_from_summaries(
        mod_list, SS, param_dict, hyper_dict, **kwargs):
    if param_dict is None:
        param_dict = dict()
    for mod in mod_list:
        cur_param_dict = mod.update_global_params_from_summaries(
            SS, param_dict, hyper_dict, **kwargs)
        param_dict.update(cur_param_dict)
    return param_dict


## Proposals (merge/delete/birth/etc)
def make_plans_for_proposals(
        mod_list, data, LP, SSU, SSL, param_dict, hyper_dict, **kwargs):
    plan_dict_list = list()
    for mod in mod_list:
        if hasattr(mod, 'make_plans_for_proposals'):
            plan_dict_list = mod.make_plans_for_proposals(
                data, LP, SSU, SSL, param_dict, hyper_dict, **kwargs)
    return plan_dict_list

def make_summary_seeds_for_proposals(
        mod_list, data, LP, SSU, SSL, param_dict, hyper_dict, **kwargs):
    seed_list = list()
    for mod in mod_list:
        if hasattr(mod, 'make_summary_seeds_for_proposals'):
            seed_list = mod.make_summary_seeds_for_proposals(
                data, LP, SSU, SSL, param_dict, hyper_dict, **kwargs)
    return seed_list

def evaluate_proposals(
        mod_list, GP, SSU, SSL, HP,
        cur_uids_K=None,
        cur_loss=None,
        **kwargs):
    for mod in mod_list:
        if hasattr(mod, 'evaluate_proposals'):
            evaluate_proposals = mod.evaluate_proposals
    def calc_loss(GP, SSU, SSL):
        return calc_loss_from_summaries(
            mod_list, GP, HP, SSU, SSL)
    def update_global_params(GP, SSU, SSL):
        return update_global_params_from_summaries(
            mod_list, SSU, GP, HP)
    GP, SSU, SSL, cur_loss, cur_uids_K = evaluate_proposals(
        GP, SSU, SSL,
        calc_loss=calc_loss,
        update_global_params=update_global_params,
        cur_loss=cur_loss,
        cur_uids_K=cur_uids_K,
        **kwargs)
    return GP, SSU, SSL, cur_loss, cur_uids_K