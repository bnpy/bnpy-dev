import numpy as np
import multiprocessing

from bnpy.alloc_models.mixtures \
    import dp_mix_vb_posterior_estimator as amod
from bnpy.obs_models \
    import gauss_diag_covar_vb_posterior_estimator as omod
from bnpy.alloc_models \
    import hierarchical_model as hmod

from bnpy.data import \
    convert_dataset_to_shared_mem_dict, \
    make_list_of_slices_for_dataset
from bnpy.utils_parallel import \
    SharedMemWorker, \
    make_shared_mem_dict_from_numpy_dict, \
    convert_shared_mem_dict_to_numpy_dict

from incremental_coord_descent import increment_in_place

default_alg_kwargs = dict(
    n_laps=1,
    n_parallel_workers=1,
    do_loss_after_local=True,
    do_loss_after_global=True,
    )

def fit(
        mod_list, data, GP, HP,
        local_step_kwargs=None,
        global_step_kwargs=None,
        n_laps=default_alg_kwargs['n_laps'],
        n_parallel_workers=default_alg_kwargs['n_parallel_workers'],
        do_loss_after_local=default_alg_kwargs['do_loss_after_local'],
        do_loss_after_global=default_alg_kwargs['do_loss_after_global'],
        **alg_kwargs):
    ''' Apply coord descent algorithm for specific dataset and initialization

    Returns
    -------
    GP : dict, updated global params
    '''
    # Parse input
    if local_step_kwargs is None:
        local_step_kwargs = dict()
    if global_step_kwargs is None:
        global_step_kwargs = dict()
    loss_history = list()

    shared_data_dict, make_dataset_from_shared_data_dict = \
        convert_dataset_to_shared_mem_dict(data)
    shared_HP = make_shared_mem_dict_from_numpy_dict(HP)
    shared_GP = make_shared_mem_dict_from_numpy_dict(GP)

    # Create a JobQ (to hold tasks to be done)
    # and a ResultsQ (to hold results of completed tasks)
    manager = multiprocessing.Manager()
    JobQ = manager.Queue()
    ResultQ = manager.Queue()

    # Create multiple workers
    for uid in range(n_parallel_workers):
        worker = SharedMemWorker(
            uid, JobQ, ResultQ,
            hmod=hmod,
            mod_list=mod_list,
            make_dataset_from_shared_data_dict=\
                make_dataset_from_shared_data_dict,
            shared_data_dict=shared_data_dict,
            shared_hyper_dict=shared_HP,
            shared_param_dict=shared_GP,
            verbose=1)
        worker.start()


    for lap_id in range(n_laps):
        # MAP!
        # Add each slice to the pending job queue
        # To be completed by one of the parallel workers
        for cur_slice in make_list_of_slices_for_dataset(
                data, n_parallel_workers):
            JobQ.put(dict(
                slice_interval=cur_slice,
                ))
        # Pause at this line until all jobs are marked complete.
        JobQ.join()

        # REDUCE!
        # Aggregate results across across all workers
        SS_update, SS_loss = ResultQ.get()
        while not ResultQ.empty():
            slice_SS_update, slice_SS_loss = ResultQ.get()
            SS_update = increment_in_place(SS_update, slice_SS_update)
            SS_loss = increment_in_place(SS_loss, slice_SS_loss)

        # Update global params
        GP = convert_shared_mem_dict_to_numpy_dict(shared_GP)
        GP = hmod.update_global_params_from_summaries(
            mod_list, SS_update, GP, HP,
            fill_existing_arrays=True)
        if do_loss_after_global:
            loss = hmod.calc_loss_from_summaries(
                mod_list, GP, HP, SS_update, SS_loss)
            loss_history.append(loss)
            print "% .5e" % loss

    # Finished! Send shutdown signal to all workers
    # Passing None to JobQ is shutdown signal
    for worker_id in range(n_parallel_workers):
        JobQ.put(None)

    return GP, dict(
        SS_update=SS_update,
        SS_loss=SS_loss,
        loss_history=loss_history)

if __name__ == '__main__':
    prng = np.random.RandomState(0)
    X_a_NaD = 0.1 * prng.randn(500, 3) - 5
    X_b_NbD = 0.1 * prng.randn(500, 3) + 0
    X_c_NcD = 0.1 * prng.randn(5, 3) + 5
    X_ND = np.vstack([X_a_NaD, X_b_NbD, X_c_NcD])

    mod_list = [omod, amod]
    GP, HP, _, local_kwargs = \
        hmod.create_and_initialize_hierarchical_model_for_dataset(
            mod_list, X_ND, K=3)
    fit(mod_list, X_ND, GP, HP,
        local_kwargs=local_kwargs,
        n_laps=10,
        n_parallel_workers=4)