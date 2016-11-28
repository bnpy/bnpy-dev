import bnpy.data

from bnpy.alloc_models import hierarchical_model as hmod
from bnpy.utils_io import parse_user_input_into_kwarg_dict

def train_model(
        dataset=None,
        alloc_model_name=None,
        obs_model_name=None,
        alg_name=None,
        **kwargs):
    # Pack into kwarg dict, then parse
    if alloc_model_name is not None:
        kwargs['alloc_model_name'] = alloc_model_name
    if obs_model_name is not None:
        kwargs['obs_model_name'] = obs_model_name
    if alg_name is not None:
        kwargs['alg_name'] = alg_name
    dataset, mod_list, alg, kwargs = parse_user_input_into_kwarg_dict(
        dataset_or_path=dataset, **kwargs)
    print 'User input'
    for key in sorted(kwargs.keys()):
        print '--', key, kwargs[key]

    print 'Dataset'
    if not isinstance(dataset, bnpy.data.XData):
        if isinstance(dataset, str):
            dataset = bnpy.data.XData.read_file(dataset)
        else:
            dataset = bnpy.data.XData(dataset)
    print dataset.get_stats_summary()

    print 'Initialization'
    # print 'K = ' % kwargs['init_K']
    # print 'init_procedure = ' % kwargs['init_procedure']
    GP, HP, init_info, kwargs_by_use, extra_kwargs = \
        hmod.create_and_initialize_hierarchical_model_for_dataset(
            mod_list, dataset, **kwargs)

    alg_kwargs = dict(**alg.default_alg_kwargs)
    for key in alg_kwargs:
        if key in extra_kwargs:
            alg_kwargs[key] = extra_kwargs[key]
            del extra_kwargs[key]
    for key in sorted(extra_kwargs.keys()):
        print 'Unrecognized kwarg %s %s' % (key, extra_kwargs[key])

    alg_kwargs.update(kwargs_by_use)
    print 'Training...'
    return alg.fit(mod_list, dataset, GP, HP, **alg_kwargs)

if __name__ == '__main__':
    train_model()