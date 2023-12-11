# GMFU Experiment Running Manager
# Reconstructor: Star <Star@seas.hahaha.edu>
# For license information, see LICENSE.TXT
"""Experiment Running Manager"""
import os
from omegaconf import OmegaConf
from dataclasses import dataclass


@dataclass
class ExperimentRunManager:
    def __init__(
        self,
        exp_version_manager: str = ".pt_tmp/exp_version_manager.yml",
    ) -> None:
        """The manages the versions of the experiments based on the name. It is a simple dictionary(yaml) based lookup.
        Primary purpose is to avoid overwriting of saved models while runing the training without changing the experiment name.
        Args:
            exp_version_manager (str, optional): The path of the yml file which acts as version control.
                Defaults to ".pt_tmp/exp_version_manager.yml".
        """
        super().__init__()
        self._exp_version_manager = exp_version_manager
        if os.path.exists(exp_version_manager):
            self.exp_version_manager = OmegaConf.load(exp_version_manager)
        else:
            self.exp_version_manager = OmegaConf.create({})
            os.makedirs(os.path.split(exp_version_manager)[0], exist_ok=True)
            with open(self._exp_version_manager, "w") as file:
                OmegaConf.save(config=self.exp_version_manager, f=file)
    def update_versions(self, name):
        if name in self.exp_version_manager.keys():
            uid = self.exp_version_manager[name] + 1
        else:
            uid = 1
        self.exp_version_manager[name] = uid
        with open(self._exp_version_manager, "w") as file:
            OmegaConf.save(config=self.exp_version_manager, f=file)
        return uid