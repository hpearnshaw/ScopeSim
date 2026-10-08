# -*- coding: utf-8 -*-
"""Contains the UserCommands class and some helper functions."""

from warnings import warn
from copy import deepcopy
from pathlib import Path
from collections.abc import Iterable, Collection, Mapping, MutableMapping
from typing import Any

import yaml
import httpx
from packaging.version import parse

from astar_utils import NestedMapping, RecursiveNestedMapping, NestedChainMap
from astar_utils.nested_mapping import recursive_update, is_bangkey

from .. import rc, __version__
from ..utils import find_file, top_level_catch, get_logger


logger = get_logger(__name__)

__all__ = ["UserCommands"]


class UserCommands(NestedChainMap):
    """
    Contains all the setting a user may wish to alter for an optical train.

    Most of the important settings are kept in the internal nested dictionary.
    Setting can be accessed by using the alias names. Currently these are:

    - ATMO: atmospheric and observatory location settings
    - TEL: telescope related settings
    - RO: relay optics settings, i.e. between telescope and instrument
    - INST: instrument optics settings
    - DET: detector settings
    - OBS: observation settings, and
    - SIM: simulation settings

    All of the settings are contained in a special ``SystemDict`` dictionary
    that allows the user to access all the settings via a bang-string (!). E.g::

        cmds = UserCommands()
        cmds["!SIM.file.local_packages_path]

    .. note::

       To use this format for accessing hierarchically-stored values, the bang
       string must always begin with a "!"

    Alternatively the same value can be accessed via the normal dictionary
    format. E.g::

        cmds["SIM"]["file"]["local_packages_path"]


    Parameters
    ----------
    use_instrument : str, optional
        The name of the main instrument to use

    yamls : list, optional
        list of yaml filenames that are needed for the combined optical system
        E.g. ["MICADO_Standalone_RO.yaml", "MICADO_H4RG.yaml", "MICADO_.yaml"]

    mode_yamls : list of yamls, optional
        list of yaml docs ("OBS" docs) that are applicable only to specific
        operational modes of the instrument.
        Further yaml files can be specified in the recursive doc entry: "yamls"

    set_modes : list of strings, optional
        A list of default mode yamls to load. E.g. ["SCAO", "IMG_4mas"]

    properties : dict, optional
        Any extra "OBS" properties that should be added

    ignore_effects : list
        Not yet implemented

    add_effects : list
        Not yet implemented

    override_effect_values : dict
        Not yet implemented


    Attributes
    ----------
    cmds : RecursiveNestedMapping
        Built from the ``properties`` dictionary of a yaml dictionary. All
        values here are accessible globally by all ``Effects`` objects in an
        ``OpticalTrain`` once the ``UserCommands`` has been passed to the
        ``OpticalTrain``.

    Notes
    -----
    .. attention:: We track your IP address when ``ScopeSim`` checks for updates

        When initialising a UserCommands object via ``use_instrument=``,
        ``ScopeSim`` checks on the database whether there are updates to the
        instrument package. Our server records the IP address of each query for
        out own statistics only.

        WE DO NOT STORE OR TRACK PERSONAL DATA. THESE STATISTICS ARE NEEDED FOR
        GETTING MORE FUNDING TO CONTINUE DEVELOPING THIS PROJECT.

        We are doing this solely as a way of showing the austrian funding agency
        that people are indeed using this software (or not). Your participation
        in this effort greatly helps our chances of securing the next grant.

        However, if you would still like to avoid your IP address being stored,
        you can run ``scopesim`` 100% anonymously by setting::

        >>> scopsim.rc.__config__["!SIM.reports.ip_tracking"] = True

        at the beginning of each session. Alternatively you can also pass the
        same bang keyword when generating a ``UserCommand`` object::

        >>> from scopesim import UserCommands
        >>> UserCommands(use_instrument="MICADO",
        ...              properties={"!SIM.reports.ip_tracking": False})

        If you use a custom ``yaml`` configuration file, you can also add this
        keyword to the ``properties`` section of the ``yaml`` file.

    .. versionchanged:: v0.8.0

       This now inherits from (a subclass of) `collections.ChainMap`.

    """

    @top_level_catch
    def __init__(self, *maps, **kwargs):
        if not maps:
            maps = [rc.__config__]
        if not any(getattr(_map, "title", "") == "CurrSys" for _map in maps):
            # Don't add another layer when .new_child() is called.
            maps = [RecursiveNestedMapping(title="CurrSys"), *maps]

        super().__init__(*maps)

        self.yaml_dicts = []
        # HACK: the deepcopy is necessary because otherwise some subdicts
        #       e.g. properties gets emptied, not sure why
        self._kwargs = deepcopy(kwargs)
        self.ignore_effects = []
        self.package_name = "UVEX"
        self.package_status = ""
        self.default_yamls = []
        self.modes_dict = {}

        self.update(**kwargs)

    def _load_yaml_dict(self, yaml_dict):
        logger.debug("    called load dict yaml")

        # FIXME: See if this occurs outside the test_package. If not, remove
        #        the if statement and the logging call and just put the assert
        #        back in, which is more efficient.
        # assert "alias" in yaml_dict, f"no alias found in {yaml_dict}"
        if "alias" not in yaml_dict:
            logger.error(
                "No 'alias' found in %s. This shouldn't happen outside testing"
                "and mocking.", yaml_dict)

        self.update_alias(self.maps[0], yaml_dict)
        self.yaml_dicts.append(yaml_dict)

        if "packages" in yaml_dict:
            logger.debug("        found packages")
            self.update(packages=yaml_dict["packages"])

        # recursive
        sub_yamls = yaml_dict.get("yamls", [])
        logger.debug("      found %d sub-yamls", len(sub_yamls))
        self._load_yamls(sub_yamls)

        if "mode_yamls" in yaml_dict:
            logger.debug("        found mode_yamls")
            self.update(mode_yamls=yaml_dict["mode_yamls"])

        if yaml_dict.get("object", "") == "configuration":
            # we're in default.yaml
            self.package_status = yaml_dict.get("status", "unknown")
            if self.package_status == "deprecated":
                warn("The selected instrument package is deprecated!",
                     DeprecationWarning, stacklevel=7)
            if self.package_status == "concept":
                raise NotImplementedError(
                    "The selected instrument package is not yet supported."
                )
            if self.package_status == "experimental":
                # or rather warning?
                logger.info(
                    "The selected instrument package is still in experimental "
                    "stage, results may not be representative of physical "
                    "instrument, use with care."
                )

        logger.debug("      dict yaml done")

    def _load_yamls(self, yamls: Collection) -> None:
        logger.debug("called load yaml with %d yamls", len(yamls))
        for yaml_input in yamls:
            if isinstance(yaml_input, str):
                logger.debug("  found str yaml: %s", yaml_input)
                if (yaml_file := find_file(yaml_input)) is None:
                    logger.error("%s could not be found.", yaml_input)
                    continue

                yaml_dicts = load_yaml_dicts(yaml_file)
                logger.debug("  loaded %d yamls from %s", len(yaml_dicts), yaml_input)
                # recursive
                for yaml_dict in yaml_dicts:
                    self._load_yaml_dict(yaml_dict)
                if yaml_input == "default.yaml":
                    logger.debug("    setting default yaml")
                    self.default_yamls = yaml_dicts
                logger.debug("  str yaml done")

            elif isinstance(yaml_input, Mapping):
                self._load_yaml_dict(yaml_input)
            else:
                raise ValueError("yaml_dicts must be a filename or a "
                                 f"mapping (dict): {yaml_input}")

    def update(self, other=None, /, **kwargs):
        """
        Update the current parameters with a yaml dictionary.

        See the ``UserCommands`` main docstring for acceptable kwargs
        """
        if other is not None:
            self.update(**other)

        if "use_instrument" in kwargs:
            self.package_name = kwargs["use_instrument"]
            self.update(packages=[kwargs["use_instrument"]],
                        yamls=["default.yaml"])
            
            # check_for_updates(self.package_name)

        if "packages" in kwargs:
            add_packages_to_rc_search(self["!SIM.file.local_packages_path"],
                                      kwargs["packages"])

        self._load_yamls(kwargs.get("yamls", []))

        if mode_yamls := kwargs.get("mode_yamls"):
            # Convert the yaml list of modes to a dict object
            # TODO: Why isn't this a dict with name as key to begin with???
            #       But that's an IRDB thing...
            #       Also, is the "name" needed in the dict later? If yes, put
            #       this back to where it was:
            # Update on this: so the original was needed because during the
            #       set_modes call, this gets called again, and now thows an
            #       error because the name key is no longer there.....
            self.modes_dict = {mode["name"]: mode for mode in mode_yamls}
            # self.modes_dict = {mode.pop("name"): mode for mode in mode_yamls}

            if "modes" in self["!OBS"]:
                # This shouldn't be necessary, i.e. we might want to see that
                # error if it occurs...
                # if not isinstance(self["!OBS.modes"], list):
                #     self["!OBS.modes"] = [self["!OBS.modes"]]
                for mode_name in self["!OBS.modes"]:
                    mode_yaml = self.modes_dict[mode_name]
                    self._load_yaml_dict(mode_yaml)

        if modes := list(kwargs.get("set_modes", [])):
            self.set_modes(*modes)

        # Calling underlying NestedMapping's update method to avoid recursion
        self.maps[0].update(kwargs.get("properties", {}))

        self.ignore_effects = kwargs.get("ignore_effects", [])

        if "add_effects" in kwargs:
            raise NotImplementedError(
                "The 'add_effects' keyword is not yet supported.")

        if "override_effect_values" in kwargs:
            raise NotImplementedError(
                "The 'override_effect_values' keyword is not yet supported.")

    @staticmethod
    def update_alias(mapping: MutableMapping, new_dict: Mapping) -> None:
        """Update a dict-like according to the alias-properties syntax.

        This used to be part of `astar_utils.NestedMapping`, but is specific to
        ScopeSim and thus belongs somewhere here. It should only be used in the
        context of YAML-dicts loaded by UserCommands, hence it was put here.
        """
        if isinstance(new_dict, NestedMapping):
            new_dict = new_dict.dic  # Avoid updating with another one

        if alias := new_dict.get("alias"):
            logger.debug("updating alias %s", alias)
            propdict = new_dict.get("properties", {})
            if alias in mapping:
                mapping[alias] = recursive_update(mapping[alias], propdict)
            else:
                mapping[alias] = propdict
        else:
            # Catch any bang-string properties keys
            to_pop = []
            for key in new_dict:
                if is_bangkey(key):
                    logger.debug(
                        "Bang-string key %s was seen in .update. This should "
                        "not occur outside mocking in testing!", key)
                    mapping[key] = new_dict[key]
                    to_pop.append(key)
            for key in to_pop:
                new_dict.pop(key)

            if len(new_dict) > 0:
                mapping = recursive_update(mapping, new_dict)

    def set_modes(self, *modes) -> None:
        """Reload with the specified `modes`.

        .. versionchanged:: v0.8.0

        This used to take a single list-like argument, now used a "*args"
        approach to deal with multiple modes.
        """
        # TODO: Remove this as soon as we can be sure enough it won't break
        #       stuff or annoy anyone too badly.
        if (len(modes) == 1 and isinstance(modes, Iterable)
                and not isinstance(modes[0], str)):
            warn(
                "Passing a list to set_modes is deprecated and will no longer "
                "work in future versions. Please just pass all modes as "
                "arguments instead.", DeprecationWarning, stacklevel=2)
            modes = modes[0]

        for defyam in self.default_yamls:
            if "properties" not in defyam:
                continue
            if "modes" not in defyam["properties"]:
                continue

            defyam["properties"]["modes"].clear()
            for mode in modes:
                if mode not in self.modes_dict:
                    raise ValueError(f"mode '{mode}' was not recognised")

                defyam["properties"]["modes"].append(mode)
                if ((msg := self.modes_dict[mode].get("deprecate", "")) or
                        self.modes_dict[mode].get("status") == "deprecated"):
                    # Fallback if status: deprecated but not deprecate key
                    msg = msg or f"Instrument mode '{mode}' is deprecated."
                    warn(msg, DeprecationWarning, stacklevel=2)

                if self.modes_dict[mode].get("status") == "experimental":
                    # or rather warning?
                    logger.info(
                        "Mode '%s' is still in experimental stage, results "
                        "may not be representative of physical instrument.",
                        mode
                    )

                if self.modes_dict[mode].get("status") == "concept":
                    raise NotImplementedError(
                        f"Instrument mode '{mode}' is not yet supported."
                    )

        # Note: This used to completely reset the instance via the line below.
        #       Calling init like this is bad design, so I replaced is with a
        #       more manual reset.
        #       TLDR: If weird things start happening, look here...
        # self.__init__(yamls=self.default_yamls)
        self.yaml_dicts.clear()
        self._load_yamls(self.default_yamls)

    def list_modes(self) -> Iterable[tuple[str, ...]]:
        """Yield tuples of length >= 2 with mode names and descriptions.

        .. versionchanged:: v0.8.0

        This used to return the formatted string. For a broader range of use
        cases, it now returns a generator of tuples of strings.
        """
        for mode, subdict in self.modes_dict.items():
            # TODO: maybe find a prettier way to print the status...
            # TODO: maybe also print mode type (e.g. MICADO SCAO)
            desc = (
                subdict.get("description", "<None>") +
                f":status={subdict.get('status')}" * ("status" in subdict) +
                ":DEPRECATED" * (
                    "deprecate" in subdict and "status" not in subdict
                )
            )
            yield mode, *(s.strip() for s in desc.split(":"))

    @property
    def modes(self) -> None:
        """Print all modes, if any."""
        modes = "\n".join(f"{mode}: {', '.join(desc)}"
                          for mode, *desc in self.list_modes())
        if modes:
            print(modes)
        else:
            print("<No modes found>")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(**{self._kwargs!r})"

    def _repr_pretty_(self, printer, cycle):  # inheritance dosen't work here??
        """For ipython."""
        if cycle:
            printer.text("UserCommands(...)")
        else:
            printer.text(str(self))


def patch_fake_symlinks(path: Path):
    """Fix broken symlinks in path.

    The irdb has some symlinks in it, which work fine under linux, but not
    always under windows, see https://stackoverflow.com/a/11664406 .

    "This makes symlinks created and committed e.g. under Linux appear as
    plain text files that contain the link text under Windows"

    It is therefore necessary to assume that these can be regular files.

    E.g. when Path.cwd() is
    WindowsPath('C:/Users/hugo/hugo/repos/irdb/MICADO/docs/example_notebooks')
    and path is WindowsPath('inst_pkgs/MICADO')
    then this function should return
    WindowsPath('C:/Users/hugo/hugo/repos/irdb/MICADO')
    """
    path = path.resolve()
    if path.exists() and path.is_dir():
        # A normal directory.
        return path
    if path.exists() and path.is_file():
        # Could be a regular file, or a broken symlink.
        size = path.stat().st_size
        if size > 250 or size == 0:
            # A symlink is probably not longer than 250 characters.
            return path
        line = open(path).readline()
        if len(line) != size:
            # There is more content in the file, so probably not a link.
            return path
        pline = Path(line)
        if pline.exists():
            # The file contains exactly a path that exists. So it is
            # probably a link.
            return pline.resolve()
    if path.exists():
        # The path exists, but is not a file or directory. Just return it.
        return path
    # The path does not exist.
    parent = path.parent
    pathup = patch_fake_symlinks(parent)

    if pathup == parent:
        raise FileNotFoundError(f"Cannot find path '{path.name}'")

    return patch_fake_symlinks(pathup / path.name)


def load_yaml_dicts(filename: str) -> list[dict[str, Any]]:
    """
    Load one or more dicts stored in a YAML file under `filename`.

    Parameters
    ----------
    filename : str
        Path to the YAML file

    Returns
    -------
    yaml_dicts : list
        A list of dicts

    """
    with open(filename, encoding="utf-8") as file:
        return list(yaml.full_load_all(file))


def add_packages_to_rc_search(local_path, package_list):
    """
    Add the paths of a list of locally saved packages to the search path list.

    Parameters
    ----------
    local_path : str
        Where the pacakges are located. Generally given by the value in
        scopesim.rc.__config__["!SIM.file.local_packages_path"]

    package_list : list
        A list of the package names to add

    """
    try:
        plocal_path = patch_fake_symlinks(Path(local_path))
    except FileNotFoundError:
        # retry with mocks
        plocal_path = rc.__pkg_dir__ / "tests/mocks"

    for pkg in package_list:
        pkg_dir = plocal_path / pkg
        if not pkg_dir.exists():
            # retry with mocks
            pkg_dir = rc.__pkg_dir__ / "tests/mocks" / pkg
            if not pkg_dir.exists():  # still no
                # For sub-packages (e.g. ELT, Armazones) it might be ok to not
                # find them, but if we're at the top level, this should not go
                # silent.
                if len(package_list) == 1:
                    raise ValueError(f"Package could not be found: {pkg_dir}")
                else:
                    logger.warning("Package could not be found: %s", pkg_dir)

        rc.__search_path__.append_first(pkg_dir)
