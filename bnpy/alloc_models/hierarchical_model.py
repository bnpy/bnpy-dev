def create_and_initialize_hierarchical_model_for_dataset(
        mod_list,
        dataset,
        **input_kwargs):
    hyper_kwargs = make_hyper_kwargs(mod_list, **input_kwargs)
    init_kwargs = make_init_kwargs(mod_list, **input_kwargs)
    local_step_kwargs = make_local_step_kwargs(mod_list, **input_kwargs)
    move_plan_kwargs = make_move_plan_kwargs(mod_list, **input_kwargs)

    # Now create the hyper parameters
    hyper_dict = init_hyper_params(
        mod_list, dataset, **hyper_kwargs)

    # Now create the global parameters
    param_dict, init_info_dict = init_global_params(
        mod_list, dataset, hyper_dict, **init_kwargs)

    extra_kwargs = dict()
    for key in input_kwargs:
        if key in hyper_kwargs or key in init_kwargs:
            continue
        if key in local_step_kwargs:
            continue
        extra_kwargs[key] = input_kwargs[key]

    kwargs_by_use = dict(
        init_kwargs=init_kwargs,
        local_step_kwargs=local_step_kwargs,
        global_step_kwargs=dict(),
        move_plan_kwargs=move_plan_kwargs,
        )
    return (param_dict, hyper_dict, init_info_dict,
        kwargs_by_use, extra_kwargs)

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

def make_move_plan_kwargs(mod_list, **input_kwargs):
    kwargs = dict()
    for mod in mod_list:
        if hasattr(mod, 'default_merge_kwargs'):
            kwargs.update(mod.default_merge_kwargs)
        if hasattr(mod, 'default_split_kwargs'):
            kwargs.update(mod.default_split_kwargs)
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

def to_common_params(mod_list, GP, **kwargs):
    ''' Convert provided global parameters into "common" point estimates
    '''
    common_GP = dict()
    for mod in mod_list:
        common_GP.update(mod.to_common_params(GP, **kwargs))
    return common_GP

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
    fdict = make_function_dict(mod_list)
    cur_kwargs = kwargs.copy()
    cur_kwargs.update(fdict)
    for mod in mod_list:
        if hasattr(mod, 'make_summary_seeds_for_proposals'):
            seed_list = mod.make_summary_seeds_for_proposals(
                data, LP, SSU, SSL, param_dict, hyper_dict, **cur_kwargs)
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

def reorder_summaries_and_update_params(
        mod_list, GP, SSU, SSL, HP,
        cur_uids_K=None,
        cur_loss=None,
        **kwargs):
    did_change = False
    for mod in mod_list:
        if hasattr(mod, 'reorder_summaries_and_update_params'):
            SSU, SSL, GP, new_uids_K, did_change = \
                mod.reorder_summaries_and_update_params(
                    SSU, SSL, GP, HP, cur_uids_K)
    if did_change:
        new_loss = calc_loss_from_summaries(mod_list, GP, HP, SSU, SSL)
    else:
        new_loss = cur_loss
    return GP, SSU, SSL, new_loss, cur_uids_K

def make_model(mod_list, GP, HP):
    ''' Create a model-like object

    Returns
    -------
    model : instance of anonymous class
    '''
    fdict = make_function_dict(mod_list)
    class Model(object):
        def __init__(self, GP, HP):
            self.GP = GP
            self.HP = HP
        def calc_local_params(self, data, **kwargs):
            return fdict['calc_local_params'](data, self.GP)
    return Model(GP, HP)

def make_function_dict(mod_list):
    ''' Create encapsulated functions that do not need mod_list arg

    Returns
    -------
    fdict : dict of function handles
    '''
    def _to_common_params(GP, **kwargs):
        return to_common_params(mod_list, GP, **kwargs)
    def _calc_local_params(data, GP, LP=None, **kwargs):
        return calc_local_params(mod_list, data, GP, LP, **kwargs)
    def _summarize_local_params_for_update(data, LP, **kwargs):
        return summarize_local_params_for_update(mod_list, data, LP, **kwargs)
    def _summarize_local_params_for_loss(data, LP, **kwargs):
        return summarize_local_params_for_loss(mod_list, data, LP, **kwargs)
    def _update_global_params_from_summaries(SSU, GP, HP, **kwargs):
        return update_global_params_from_summaries(
            mod_list, SSU, GP, HP, **kwargs)
    def _init_global_params(data, HP, **kwargs):
        return init_global_params(mod_list, data, HP, **kwargs)
    return dict(
        to_common_params=_to_common_params,
        calc_local_params=_calc_local_params,
        summarize_local_params_for_update=_summarize_local_params_for_update,
        summarize_local_params_for_loss=_summarize_local_params_for_loss,
        update_global_params_from_summaries=\
            _update_global_params_from_summaries,
        init_global_params=_init_global_params)