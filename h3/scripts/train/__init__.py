from pathlib import Path

from ...utility.arguments import ArgumentsBaseClass, PathAction


class Arguments(ArgumentsBaseClass):
    def __init__(self) -> None:
        super().__init__()

        self.dataset_dir: Path
        self._keys.append("dataset_dir")

    def add_arguments(self) -> None:
        super().add_arguments()

        self._parser.add_argument(
            "--dataset_dir",
            action=PathAction,
            exist_check=True,
            required=True,
        )

    def validate(self) -> None:
        super().validate()

    def __str__(self) -> str:
        string = super().__str__()

        string += f" --dataset_dir {self.dataset_dir}"

        return string
