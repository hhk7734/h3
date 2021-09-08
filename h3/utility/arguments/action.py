from argparse import Action, ArgumentError, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, Sequence, Union


class PathAction(Action):
    """
    Args:
        exist_check (bool): Check for file existence. Defaults to False.
        file_extension (Union[str, Sequence[str], None]): If not None, check
            for file extension. Defaults to None.

    Examples:
        parser.add_argument("--output", action=PathAction)
        parser.add_argument("--input", exist_check=True, action=PathAction)
        parser.add_argument(
            "--input", file_extension="pdbqt", action=PathAction
        )
    """

    def __init__(
        self,
        *args,
        exist_check: bool = False,
        file_extension: Union[str, Sequence[str], None] = None,
        type: Any = None,
        **kwargs,
    ) -> None:
        # pylint: disable=redefined-builtin
        super().__init__(*args, type=Path, **kwargs)
        self._exist_check = exist_check
        if isinstance(file_extension, str):
            self._file_extension = [file_extension]
        else:
            self._file_extension = file_extension

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str,
        option_strings: Optional[str] = None,
    ) -> None:
        """
        When user set an argument that is applied this action, `__call__` is
        called. If not, when `values` is None, `__call__` will be not called.

        Args:
            parser (ArgumentParser): .
            namespace (Namespace): .
            values (str): .
            option_strings (Optional[str]): .

        Raises:
            ArgumentError: .
        """
        path = Path(values).resolve()

        if (
            self._file_extension is not None
            and path.suffix[1:] not in self._file_extension
        ):
            raise ArgumentError(
                self, f"file extension must be {self._file_extension}"
            )

        if self._exist_check and not path.exists():
            raise ArgumentError(self, f"`{path}` does not exist")

        setattr(namespace, self.dest, path)


_YES = "yes"
_NO = "no"
_FORCE = "force"
_OPTIONS = (_YES, _NO, _FORCE)


class ExecuteAction(Action):
    """
    Examples:
        parser.add_argument("--run", action=ExecuteAction)
        parser.add_argument(
            "--run", action=ExecuteAction, default=ExecuteAction.YES
        )

        --run [yes]
        --run no
        --run force
    """

    YES = _YES
    NO = _NO
    FORCE = _FORCE
    OPTIONS = _OPTIONS

    def __init__(
        self,
        *args,
        choices: Sequence[str] = _OPTIONS,
        const: Any = None,
        default: str = _NO,
        nargs: Any = None,
        **kwargs,
    ):
        if not set(choices).issubset(set(self.OPTIONS)):
            raise ValueError(
                f"`choices` is {choices}. Do not add any choices except"
                f" {self.OPTIONS}"
            )

        if default not in choices:
            raise ValueError(
                f'`default` is "{default}". It must be one of {choices}.'
            )

        super().__init__(
            *args,
            choices=choices,
            const=self.YES,
            default=default,
            nargs="?",
            **kwargs,
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str,
        option_strings: Optional[str] = None,
    ) -> None:
        setattr(namespace, self.dest, values)
