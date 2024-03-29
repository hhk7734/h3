from pathlib import Path

from ...utility.arguments import ArgumentsBaseClass, ExecuteAction, PathAction


class Arguments(ArgumentsBaseClass):
    def __init__(self) -> None:
        super().__init__()

        self.dataset_dir: Path
        self._keys.append("dataset_dir")
        self.batch_size: int
        self._keys.append("batch_size")
        self.gpu: str
        self._keys.append("gpu")

    def add_arguments(self) -> None:
        super().add_arguments()

        self._parser.add_argument(
            "--dataset_dir",
            action=PathAction,
            exist_check=True,
            required=True,
        )

        self._parser.add_argument(
            "--batch_size",
            default=8,
            type=int,
        )

        self._parser.add_argument(
            "--gpu",
            action=ExecuteAction,
            default=ExecuteAction.YES,
        )

    def validate(self) -> None:
        # pylint: disable=useless-super-delegation
        super().validate()

    def __str__(self) -> str:
        string = super().__str__()

        string += f" --dataset_dir {self.dataset_dir}"
        string += f" --batch_size {self.batch_size}"
        string += f" --gpu {self.gpu}"

        return string
