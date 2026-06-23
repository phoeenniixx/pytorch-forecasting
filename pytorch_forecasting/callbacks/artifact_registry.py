from pathlib import Path
from typing import Any, Optional

from lightning.pytorch.callbacks import Callback


class _ArtifactRegistry:
    """Reads/writes artifacts.yaml."""

    @staticmethod
    def write(
        registry_path: Path, artifacts: dict[str, Any], overwrite: bool = False
    ) -> None:
        """Initial write.

        It is used by _save() to save scalers and other "static" artifacts
        before training starts. The static artifacts are those artifacts which never
        change during the training process - like the configs, datamodule.metadata,
        scalers etc. The non-static artifacts changes as the training/prediction
        process moves forward. Eg of this would be the model checkpoints - if we plan
        to save "best" model (any model that has better performance based on some
        metric)

        Parameters
        ----------
        registry_path : Path
           the path of the yaml file where we have to write the artifacts.
        artifacts : dict[str, Any]
           A dictionary of the artifacts we want to write.
           The Keys of the dictionary would be the "type" of artifact - like scaler,
           configs etc. And the value would be the actual object path to be written.
        overwrite: bool, default=False
            Whether to overwrite the artifact.yaml (if present) or not.
        """
        # Steps:
        # 1. First see if there is already a `artifacts.yaml` file.
        #   1.1 If yes,
        #       1.1.1 If overwrite is True, We just overwrite the artifacts.yaml - With
        #           a warning
        #       1.1.2 If overwrite is False, throw an error
        #   1.2 If no, create a new artifacts.yaml file
        # 2. write the artifacts dict to the artifacts.yaml file.
        pass

    @staticmethod
    def update(registry_path: Path) -> None:
        """Merge-update specific keys.
        It is used by ArtifactRegistryCallback during
        training, and by _save() for the manual_checkpoint key. It will never overwrite
        the artifacts.yaml file, but just update an existing artifact entry.

        Parameters
        ----------
        registry_path : Path
             the path of the yaml file where we have to write the artifacts.
        """
        pass

    @staticmethod
    def get(registry_path: Path, key: str) -> dict[str, Any] | None:
        """Read a single key.

        Parameters
        -----------
        registry_path : Path
            the path of the yaml file from where we have to read the artifacts.
        key : str
            the key we want to know.

        Returns
        -------
        dict
            The dictionary if with the key as the key passed and value as the value
            read, or None with a warning if missing/file doesn't exist yet.
        """
        pass


class ArtifactRegistryCallback(Callback):
    """Called every time Lightning's ModelCheckpoint actually persists a file.
    Re-reads ckpt_cb.best_model_path / .last_model_path (always current,
    never stale -- Lightning deletes superseded files itself) and writes
    that into artifacts.yaml as the live truth, not a historical log.
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        # TODO: find the ModelCheckpoint instance among trainer.callbacks
        # TODO: _ArtifactRegistry.update(
        #          ...
        #       )
        pass
