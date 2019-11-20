#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import gc
import sys
import time
import contextlib
import random
import argparse
try:
    from guppy import hpy
    guppy_available = True
except ImportError:
    guppy_available = False
try:
    from pympler.muppy import muppy
    from pympler.muppy import summary
    from pympler.muppy import tracker
    from pympler.asizeof import *
    pympler_available = True
except ImportError:
    pympler_available = False
except AttributeError:
    pympler_available = False

from pyutilib.pyro import shutdown_pyro_components
from pyutilib.misc import import_file

from pyomo.common import pyomo_command
from pyomo.common.plugin import ExtensionPoint
from pyomo.core.base import maximize, minimize, Var, Suffix
from pyomo.opt.base import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.opt import undefined
from pyomo.pysp.phextension import IPHExtension
from pyomo.pysp.ef_writer_script import ExtensiveFormAlgorithm
from pyomo.pysp.ph import ProgressiveHedging
from pyomo.pysp.phutils import (reset_nonconverged_variables,
                                reset_stage_cost_variables,
                                _OLD_OUTPUT)
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.solutionwriter import ISolutionWriterExtension
from pyomo.pysp.util.misc import (launch_command,
                                  load_extensions)
import pyomo.pysp.phsolverserverutils

from pyomo.pysp.benderinit import benderinit

#
# utility method to construct an option parser for ph arguments,
# to be supplied as an argument to the runph method.
#

# #todo currently a copy
def GenerateScenarioTreeForBenders(options,
                              scenario_instance_factory,
                              include_scenarios=None):

    scenario_tree = scenario_instance_factory.generate_scenario_tree(
        include_scenarios=include_scenarios,
        downsample_fraction=options.scenario_tree_downsample_fraction,
        bundles=options.scenario_bundle_specification,
        random_bundles=options.create_random_bundles,
        random_seed=options.scenario_tree_random_seed,
        verbose=options.verbose)

    #
    # print the input tree for validation/information purposes.
    #
    if options.verbose:
        scenario_tree.pprint()

    #
    # validate the tree prior to doing anything serious
    #
    scenario_tree.validate()
    if options.verbose:
        print("Scenario tree is valid!")

    if options.solver_manager_type != "phpyro":

        start_time = time.time()

        if not _OLD_OUTPUT:
            print("Constructing scenario tree instances")
        instance_dictionary = \
            scenario_instance_factory.construct_instances_for_scenario_tree(
                scenario_tree,
                output_instance_construction_time=options.output_instance_construction_time,
                compile_scenario_instances=options.compile_scenario_instances,
                verbose=options.verbose)

        if options.verbose or options.output_times:
            print("Time to construct scenario instances=%.2f seconds"
                  % (time.time() - start_time))

        if not _OLD_OUTPUT:
            print("Linking instances into scenario tree")
        start_time = time.time()

        # with the scenario instances now available, link the
        # referenced objects directly into the scenario tree.
        scenario_tree.linkInInstances(instance_dictionary,
                                      objective_sense=options.objective_sense,
                                      create_variable_ids=True)

        if options.verbose or options.output_times:
            print("Time link scenario tree with instances=%.2f seconds"
                  % (time.time() - start_time))

    return scenario_tree




def BenderAlgorithmBuilder(options, scenario_tree):

    import pyomo.environ
    import pyomo.solvers.plugins.smanager.phpyro
    import pyomo.solvers.plugins.smanager.pyro

    solution_writer_plugins = ExtensionPoint(ISolutionWriterExtension)
    for plugin in solution_writer_plugins:
        plugin.disable()

    solution_plugins = []
    if len(options.solution_writer) > 0:
        for this_extension in options.solution_writer:
            if this_extension in sys.modules:
                print("User-defined PH solution writer module="
                      +this_extension+" already imported - skipping")
            else:
                print("Trying to import user-defined PH "
                      "solution writer module="+this_extension)
                # make sure "." is in the PATH.
                original_path = list(sys.path)
                sys.path.insert(0,'.')
                import_file(this_extension)
                print("Module successfully loaded")
                sys.path[:] = original_path # restore to what it was

            # now that we're sure the module is loaded, re-enable this
            # specific plugin.  recall that all plugins are disabled
            # by default in phinit.py, for various reasons. if we want
            # them to be picked up, we need to enable them explicitly.
            import inspect
            module_to_find = this_extension
            if module_to_find.rfind(".py"):
                module_to_find = module_to_find.rstrip(".py")
            if module_to_find.find("/") != -1:
                module_to_find = module_to_find.split("/")[-1]

            for name, obj in inspect.getmembers(sys.modules[module_to_find],
                                                inspect.isclass):
                import pyomo.common
                # the second condition gets around goofyness related
                # to issubclass returning True when the obj is the
                # same as the test class.
                if issubclass(obj, pyomo.common.plugin.SingletonPlugin) and name != "SingletonPlugin":
                    for plugin in solution_writer_plugins(all=True):
                        if isinstance(plugin, obj):
                            plugin.enable()
                            solution_plugins.append(plugin)

    #
    # if any of the ww extension configuration options are specified
    # without the ww extension itself being enabled, halt and warn the
    # user - this has led to confusion in the past, and will save user
    # support time.
    #
    if (len(options.ww_extension_cfgfile) > 0) and \
       (options.enable_ww_extensions is False):
        raise ValueError("A configuration file was specified "
                         "for the WW extension module, but the WW extensions "
                         "are not enabled!")

    if (len(options.ww_extension_suffixfile) > 0) and \
       (options.enable_ww_extensions is False):
        raise ValueError("A suffix file was specified for the WW "
                         "extension module, but the WW extensions are not "
                         "enabled!")

    if (len(options.ww_extension_annotationfile) > 0) and \
       (options.enable_ww_extensions is False):
        raise ValueError("A annotation file was specified for the "
                         "WW extension module, but the WW extensions are not "
                         "enabled!")

    #
    # disable all plugins up-front. then, enable them on an as-needed
    # basis later in this function. the reason that plugins should be
    # disabled is that they may have been programmatically enabled in
    # a previous run of PH, and we want to start from a clean slate.
    #
    ph_extension_point = ExtensionPoint(IPHExtension)

    for plugin in ph_extension_point:
        plugin.disable()

    ph_plugins = []
    #
    # deal with any plugins. ww extension comes first currently,
    # followed by an option user-defined plugin.  order only matters
    # if both are specified.
    #
    if options.enable_ww_extensions:

        import pyomo.pysp.plugins.wwphextension

        # explicitly enable the WW extension plugin - it may have been
        # previously loaded and/or enabled.
        ph_extension_point = ExtensionPoint(IPHExtension)

        for plugin in ph_extension_point(all=True):
           if isinstance(plugin,
                         pyomo.pysp.plugins.wwphextension.wwphextension):

              plugin.enable()
              ph_plugins.append(plugin)

              # there is no reset-style method for plugins in general,
              # or the ww ph extension in plugin in particular. if no
              # configuration or suffix filename is specified, set to
              # None so that remnants from the previous use of the
              # plugin aren't picked up.
              if len(options.ww_extension_cfgfile) > 0:
                 plugin._configuration_filename = options.ww_extension_cfgfile
              else:
                 plugin._configuration_filename = None
              if len(options.ww_extension_suffixfile) > 0:
                 plugin._suffix_filename = options.ww_extension_suffixfile
              else:
                 plugin._suffix_filename = None
              if len(options.ww_extension_annotationfile) > 0:
                 plugin._annotation_filename = options.ww_extension_annotationfile
              else:
                 plugin._annotation_filename = None

    if len(options.user_defined_extensions) > 0:
        for this_extension in options.user_defined_extensions:
            if this_extension in sys.modules:
                print("User-defined PH extension module="
                      +this_extension+" already imported - skipping")
            else:
                print("Trying to import user-defined PH extension module="
                      +this_extension)
                # make sure "." is in the PATH.
                original_path = list(sys.path)
                sys.path.insert(0,'.')
                import_file(this_extension)
                print("Module successfully loaded")
                # restore to what it was
                sys.path[:] = original_path

            # now that we're sure the module is loaded, re-enable this
            # specific plugin.  recall that all plugins are disabled
            # by default in phinit.py, for various reasons. if we want
            # them to be picked up, we need to enable them explicitly.
            import inspect
            module_to_find = this_extension
            if module_to_find.rfind(".py"):
                module_to_find = module_to_find.rstrip(".py")
            if module_to_find.find("/") != -1:
                module_to_find = module_to_find.split("/")[-1]

            for name, obj in inspect.getmembers(sys.modules[module_to_find],
                                                inspect.isclass):
                import pyomo.common
                # the second condition gets around goofyness related
                # to issubclass returning True when the obj is the
                # same as the test class.
                if issubclass(obj, pyomo.common.plugin.SingletonPlugin) and name != "SingletonPlugin":
                    ph_extension_point = ExtensionPoint(IPHExtension)
                    for plugin in ph_extension_point(all=True):
                        if isinstance(plugin, obj):
                            plugin.enable()
                            ph_plugins.append(plugin)

    ph = None
    solver_manager = None
    try:

        # construct the solver manager.
        if options.verbose:
            print("Constructing solver manager of type="
                  +options.solver_manager_type)
        solver_manager = SolverManagerFactory(
            options.solver_manager_type,
            host=options.pyro_host,
            port=options.pyro_port)

        if solver_manager is None:
            raise ValueError("Failed to create solver manager of "
                             "type="+options.solver_manager_type+
                         " specified in call to PH constructor")

        ph = ProgressiveHedging(options)

        if isinstance(solver_manager,
                      pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):

            if scenario_tree.contains_bundles():
                num_jobs = len(scenario_tree._scenario_bundles)
                if not _OLD_OUTPUT:
                    print("Bundle solver jobs available: "+str(num_jobs))
            else:
                num_jobs = len(scenario_tree._scenarios)
                if not _OLD_OUTPUT:
                    print("Scenario solver jobs available: "+str(num_jobs))

            servers_expected = options.phpyro_required_workers
            if (servers_expected is None):
                servers_expected = num_jobs

            timeout = options.phpyro_workers_timeout if \
                      (options.phpyro_required_workers is None) else \
                      None

            solver_manager.acquire_servers(servers_expected,
                                           timeout)

        ph.initialize(scenario_tree=scenario_tree,
                      solver_manager=solver_manager,
                      ph_plugins=ph_plugins,
                      solution_plugins=solution_plugins)

    except:

        if ph is not None:

            ph.release_components()

        if solver_manager is not None:

            if isinstance(solver_manager,
                          pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):
                solver_manager.release_servers(shutdown=ph._shutdown_pyro_workers)
            elif isinstance(solver_manager,
                          pyomo.solvers.plugins.smanager.pyro.SolverManager_Pyro):
                if ph._shutdown_pyro_workers:
                    solver_manager.shutdown_workers()

        print("Failed to initialize progressive hedging algorithm")
        raise

    return ph

def BenderFromScratch(options):
    start_time = time.time()
    if options.verbose:
        print("Importing model and scenario tree files")

    scenario_instance_factory = \
        ScenarioTreeInstanceFactory(options.model_directory,
                                    options.instance_directory)

    if options.verbose or options.output_times:
        print("Time to import model and scenario tree "
              "structure files=%.2f seconds"
              %(time.time() - start_time))

    try:

        scenario_tree = \
            GenerateScenarioTreeForBenders(options,
                                      scenario_instance_factory)

    except:
        print("Failed to initialize model and/or scenario tree data")
        scenario_instance_factory.close()
        raise

    ph = None
    try:
        ph = BenderAlgorithmBuilder(options, scenario_tree)
    except:

        print("A failure occurred in PHAlgorithmBuilder. Cleaning up...")
        if ph is not None:
            ph.release_components()
        scenario_instance_factory.close()
        raise

    return ph

#