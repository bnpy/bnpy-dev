"""
Shared memory parallel implementation of local step

Classes
--------
SharedMemWorker : subclass of Process
    Defines work to be done by a single "worker" process
    which is created with references to shared read-only data
    We assign this process "jobs" via a queue,
    and read its results from a separate results queue.
"""

import sys
import os
import multiprocessing
from multiprocessing import sharedctypes
import itertools
import numpy as np
import ctypes
import time

from shared_mem_util import \
    convert_shared_mem_dict_to_numpy_dict

class SharedMemWorker(multiprocessing.Process):

    ''' Single "worker" process that processes tasks delivered via queues
    '''

    def __init__(self, uid, JobQueue, ResultQueue,
                 hmod,
                 mod_list,
                 shared_data_dict,
                 shared_param_dict,
                 shared_hyper_dict,
                 make_dataset_from_shared_data_dict=None,
                 local_step_kwargs=None,
                 verbose=0):
        super(SharedMemWorker, self).__init__()
        self.uid = uid
        self.JobQueue = JobQueue
        self.ResultQueue = ResultQueue

        self.hmod = hmod
        self.mod_list = mod_list
        self.shared_data_dict = shared_data_dict
        self.shared_param_dict = shared_param_dict
        self.shared_hyper_dict = shared_hyper_dict

        self.make_dataset_from_shared_data_dict = \
            make_dataset_from_shared_data_dict

        if local_step_kwargs is None:
            local_step_kwargs = dict()
        self.local_step_kwargs = local_step_kwargs
        self.verbose = verbose

    def print_msg(self, msg):
        if self.verbose:
            for line in msg.split("\n"):
                print "#%d: %s" % (self.uid, line)

    def run(self):
        self.print_msg("process start! pid=%d" % (os.getpid()))

        # Construct iterator with sentinel value of None (for termination)
        job_iterator = iter(self.JobQueue.get, None)

        hmod = self.hmod
        mod_list = self.mod_list
        local_step_kwargs = self.local_step_kwargs

        for slice_dict in job_iterator:
            # Grab slice of data to work on
            if 'slice_interval' in slice_dict:
                # Load data from shared memory
                slice_dataset = self.make_dataset_from_shared_data_dict(
                    self.shared_data_dict,
                    **slice_dict)
            else:
                # Load data slice directly from file
                raise NotImplementedError("TODO")
            #self.print_msg("slice " + str(slice_dict['slice_interval']))
            
            # Get latest global parameters as dict of numpy arrays
            GP = convert_shared_mem_dict_to_numpy_dict(
                self.shared_param_dict)

            # Do local step on current slice
            slice_LP = hmod.calc_local_params(
                mod_list,
                slice_dataset,
                GP,
                **local_step_kwargs)
            slice_SS_update = hmod.summarize_local_params_for_update(
                mod_list, slice_dataset, slice_LP)
            slice_SS_loss = hmod.summarize_local_params_for_loss(
                mod_list, slice_dataset, slice_LP)

            # Add to result queue to wrap up this task!
            self.ResultQueue.put(
                (slice_SS_update, slice_SS_loss))
            self.JobQueue.task_done()

        # Clean up
        self.print_msg("process stop! pid=%d" % (os.getpid()))
