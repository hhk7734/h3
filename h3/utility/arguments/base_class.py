import logging
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator, List, Optional

from ..logging import set_root_logger as _set_root_logger
from .action import PathAction


class ArgumentsBaseClass(Namespace, Mapping):
    @abstractmethod
    def __init__(self) -> None:
        # pylint: disable=super-init-not-called
        self._parser = ArgumentParser()

        # Must add arguments to `self._keys` except `self.log_level` and
        # `self.log_file`
        self._keys: List[str] = []

        self._log_parser = ArgumentParser()
        self.log_level: int
        self.log_file: Optional[Path]

    @abstractmethod
    def add_arguments(self) -> None:
        self._log_parser.add_argument(
            "--verbose",
            action="store_const",
            const=logging.INFO,
            default=logging.WARNING,
            dest="log_level",
        )

        self._log_parser.add_argument(
            "--debug",
            action="store_const",
            const=logging.DEBUG,
            dest="log_level",
        )

        self._log_parser.add_argument(
            "--log_file",
            action=PathAction,
            default=None,
        )

    @abstractmethod
    def validate(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        string = ""
        if self.log_level == logging.INFO:
            string += " --verbose"
        elif self.log_level == logging.DEBUG:
            string += " --debug"
        if self.log_file is not None:
            string += f" --log_file {self.log_file}"
        return string

    def __getitem__(self, key: str) -> Any:
        if key in self._keys:
            return getattr(self, key)

        raise KeyError(key)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def parse(
        self,
        args: Optional[List[str]] = None,
        validate: bool = True,
        set_root_logger: bool = True,
    ) -> List[str]:
        """
        Args:
            args (Optional[List[str]]): If None, args = sys.argv[1:]
            validate (bool): If True, call validate()
            set_root_logger (bool): If True, call set_root_logger()

        Returns:
            List[str]: Rest of the arguments after parsing
        """
        self.add_arguments()
        _, rest = self._log_parser.parse_known_args(args=args, namespace=self)
        if set_root_logger:
            _set_root_logger(log_level=self.log_level, log_file=self.log_file)

        # TODO: `exit_on_error` option is added in python 3.9. After python 3.9,
        # add error logging using try catch.
        _, rest = self._parser.parse_known_args(args=rest, namespace=self)
        if validate:
            self.validate()
        return rest
