import argparse
import numpy as np

import bnpy

def parse_user_input_into_kwarg_dict(
        dataset_or_path=None,
        alloc_name_map=None,
        obs_name_map=None,
        alg_name_map=None,
        **kwargs):
    '''

    Examples
    --------
    >>> dpath, mod_list, alg, kwargs = parse_user_input_into_kwarg_dict(
    ...     dataset=None,
    ...     alloc_model_name='dp_mix_vb',
    ...     obs_model_name='gauss_diag_covar_vb',
    ...     K=5,
    ...     init_procedure='LP_from_rand_examples')
    >>> print mod_list[0].__name__
    bnpy.obs_models.gauss_diag_covar_vb_posterior_estimator
    >>> print alg.__name__
    bnpy.training_algs.coord_descent
    >>> print kwargs
    {'K': 5, 'init_procedure': 'LP_from_rand_examples'}
    '''
    # Load the required maps here and avoid circular imports...
    if obs_name_map is None:
        obs_name_map = bnpy.obs_models.obs_name_map
    if alloc_name_map is None:
        alloc_name_map = bnpy.alloc_models.alloc_name_map
    if alg_name_map is None:
        alg_name_map = bnpy.training_algs.alg_name_map

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default=None)
    parser.add_argument(
        '--alloc_model_name',
        default='dp_mix_vb',
        type=str,
        choices=alloc_name_map.keys())
    parser.add_argument(
        '--obs_model_name',
        default='gauss_diag_covar_vb',
        type=str,
        choices=obs_name_map.keys())
    parser.add_argument(
        '--alg_name',
        default='coord_descent',
        type=str)

    if len(kwargs.keys()) > 0:
        arg_list, extra_dict = kwargs_to_arglist(**kwargs)
        args, unk_list = parser.parse_known_args(arg_list)
    else:
        args, unk_list = parser.parse_known_args()

    unk_kwargs = arglist_to_kwargs(unk_list)

    if dataset_or_path is None:
        dataset_or_path = args.dataset
    mod_list = []
    for mod_name in args.obs_model_name.split(','):
        if mod_name in obs_name_map:
            mod_list.append(obs_name_map[mod_name])
        else:
            raise ValueError(
                "Unrecognized obs_model_name " + args.obs_model_name)

    if args.alloc_model_name not in alloc_name_map:
        raise ValueError(
            "Unrecognized alloc_model_name " + args.alloc_model_name)
    mod_list.append(alloc_name_map[args.alloc_model_name])

    alg_module = alg_name_map[args.alg_name]
    return dataset_or_path, mod_list, alg_module, unk_kwargs

def kwargs_to_arglist(**kwargs):
    ''' Transform dict key/value pairs into an interleaved list.

    Returns
    -------
    arglist : list of str or dict or ndarray
    SafeDictForComplexTypes : dict

    Examples
    ------
    >>> kwargs = dict(a=5, b=7.7, c='stegosaurus')
    >>> kwlist, _ = kwargs_to_arglist(**kwargs)
    >>> kwlist[0:2]
    ['--a', '5']
    '''
    keys = kwargs.keys()
    keys.sort(key=len)  # sorty by length, smallest to largest
    arglist = list()
    SafeDict = dict()
    for key in keys:
        val = kwargs[key]
        if isinstance(val, dict) or isinstance(val, np.ndarray):
            SafeDict[key] = val
        else:
            arglist.append('--' + key)
            arglist.append(str(val))
    return arglist, SafeDict

def arglist_to_kwargs(alist, doConvertFromStr=True):
    ''' Transform list into key/val pair dictionary

    Neighboring entries in list are interpreted as key/value pairs.

    Returns
    -------
    kwargs : dict
        Each value is cast to appropriate primitive type
        (float/int/str) if possible. Complicated types are left alone.

    Examples
    ---------
    >>> arglist_to_kwargs(['--a', '1', '--b', 'stegosaurus'])
    {'a': 1, 'b': 'stegosaurus'}
    >>> arglist_to_kwargs(['requiredarg', 1])
    {}
    '''
    kwargs = dict()
    a = 0
    while a + 1 < len(alist):
        curarg = alist[a]
        if curarg.startswith('--'):
            argname = curarg[2:]
            argval = alist[a + 1]
            if isinstance(argval, str) and doConvertFromStr:
                curType = _getTypeFromString(argval)
                kwargs[argname] = curType(argval)
            else:
                kwargs[argname] = argval
            a += 1
        a += 1
    return kwargs


def _getTypeFromString(defVal):
    ''' Determine Python type from the provided default value.

    Returns
    ---------
    t : type
        type object. one of {int, float, str} if defVal is a string
        otherwise, evaluates and returns type(defVal)

    Examples
    ------
    >>> _getTypeFromString('deinonychus')
    <type 'str'>
    >>> _getTypeFromString('3.14')
    <type 'float'>
    >>> _getTypeFromString('555')
    <type 'int'>
    >>> _getTypeFromString('555.0')
    <type 'float'>
    >>> _getTypeFromString([1,2,3])
    <type 'list'>
    '''
    if not isinstance(defVal, str):
        return type(defVal)
    try:
        int(defVal)
        return int
    except Exception as e:
        pass
    try:
        float(defVal)
        return float
    except Exception as e:
        pass
    return str