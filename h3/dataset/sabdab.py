import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from Bio.PDB import PDBIO, PDBParser, Select
from bs4 import BeautifulSoup

_BASE_URL = "http://opig.stats.ox.ac.uk"

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummaryData:
    id: str
    hchain: str
    lchain: str


Summary = Dict[str, List[SummaryData]]


class _HLSelect(Select):
    def accept_residue(self, residue):
        _, _, chain, (hetero_flag, res_num, _) = residue.full_id
        if hetero_flag != " ":
            return 0
        if chain == "H" and res_num <= 112:
            return 1
        elif chain == "L" and res_num <= 109:
            return 1
        else:
            return 0


class SAbDab:
    @staticmethod
    def _download(url: str, output_file: Path) -> None:
        with output_file.open("w") as fd:
            fd.write(requests.get(url).content.decode("utf-8"))

    @classmethod
    def download_non_redundant_summary(
        cls,
        output_file: Path,
        max_sequence_identity: int = 99,
        paired_vh_vl_only: bool = True,
        in_complex: str = "All",
        resolution_cutoff: float = 3.0,
        r_factor_cutoff: Optional[float] = None,
    ):
        """
        http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search

        Search for a non-redundant set of antibodies
        """
        if in_complex not in ("All", "Bound only", "Unbound only"):
            raise ValueError(
                "`in_complex` must be one of `All`, `Bound only`, "
                "`Unbound only`"
            )
        search_url = f"{_BASE_URL}/webapps/newsabdab/sabdab/search"
        params = {
            "seqid": max_sequence_identity,
            "paired": paired_vh_vl_only,
            "nr_complex": in_complex,
            "nr_res": resolution_cutoff,
            "nr_rfactor": r_factor_cutoff
            if r_factor_cutoff is not None
            else "",
        }
        query = requests.get(url=search_url, params=params)
        LOG.debug("search: %s", query.url)
        html = BeautifulSoup(query.content, "html.parser")

        try:
            href_url = html.find(id="downloads").find("a").get("href")
        except:
            LOG.error("Failed to find download link in %s", query.url)
            raise

        summary_file_url = f"{_BASE_URL}{href_url}"

        LOG.info(
            "downloading SAbDab summary file %s from %s",
            output_file,
            summary_file_url,
        )

        cls._download(url=summary_file_url, output_file=output_file)

    @staticmethod
    def read_summary(summary_file: Path) -> Summary:
        summary_data = {}
        summary_df = pd.read_csv(summary_file, sep="\t")
        for _, row in summary_df.iterrows():
            if row.pdb not in summary_data:
                summary_data[row.pdb] = []
            summary_data[row.pdb].append(
                SummaryData(id=row.pdb, hchain=row.Hchain, lchain=row.Lchain)
            )
        return summary_data

    @classmethod
    def download_pdbs(
        cls,
        summary: Summary,
        pdb_dir: Path,
        structure_type: str = "chothia",
        max_workers: int = 16,
    ):
        pdb_dir.mkdir(exist_ok=True, parents=True)
        for pdb_file in pdb_dir.glob("*.pdb"):
            pdb_file.unlink()
        pdb_url = f"{_BASE_URL}/webapps/newsabdab/sabdab/pdb"
        pdb_list = {
            pdb_id: {
                "path": pdb_dir.joinpath(f"{pdb_id}.pdb"),
                "url": f"{pdb_url}/{pdb_id}/?scheme={structure_type}",
            }
            for pdb_id in summary
        }
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = [
                executor.submit(
                    lambda v: cls._download(
                        url=v["url"], output_file=v["path"]
                    ),
                    value,
                )
                for value in pdb_list.values()
            ]

            total = len(pdb_list)
            count = 0
            interval = total // 10
            for _ in as_completed(results):
                count += 1
                if count % interval == 0:
                    LOG.info(
                        "%d %% PDB files have been downloaded",
                        round(count / total * 100),
                    )

    @staticmethod
    def truncate_pdbs(summary: Summary, pdb_dir: Path):
        truncated_dir = pdb_dir.joinpath("truncated")
        truncated_dir.mkdir(exist_ok=True, parents=True)
        for pdb_file in truncated_dir.glob("*.pdb"):
            pdb_file.unlink()

        untruncated_dir = pdb_dir.joinpath("untruncated_dir")
        untruncated_dir.mkdir(exist_ok=True, parents=True)
        for pdb_file in untruncated_dir.glob("*.pdb"):
            pdb_file.unlink()

        total = len(list(pdb_dir.glob("*.pdb")))
        count = 0
        interval = total // 10

        for pdb_file in pdb_dir.glob("*.pdb"):
            count += 1
            if count % interval == 0:
                LOG.info(
                    "%d %% PDB files have been truncated",
                    round(count / total * 100),
                )

            pdb_id = pdb_file.stem
            hchain = summary[pdb_id][0].hchain
            lchain = summary[pdb_id][0].lchain
            if hchain == lchain:
                shutil.move(
                    str(pdb_file), untruncated_dir.joinpath(pdb_file.name)
                )
                continue

            parser = PDBParser()
            try:
                structure = parser.get_structure("antibody", pdb_file)
            except ValueError:
                LOG.error("failed to read %s", pdb_file)
                shutil.move(
                    str(pdb_file), untruncated_dir.joinpath(pdb_file.name)
                )
                continue

            if len(list(structure.get_models())) > 1:
                shutil.move(
                    str(pdb_file), untruncated_dir.joinpath(pdb_file.name)
                )
                continue

            for chain in list(structure[0].get_chains()):
                if chain.id not in (hchain, lchain):
                    structure[0].detach_child(chain.id)

            if hchain == "L" or lchain == "H":
                for chain in structure.get_chains():
                    if chain.id == hchain:
                        chain.id = "A" if lchain != "A" else "C"
                        hchain = chain.id
                    elif chain.id == lchain:
                        chain.id = "B" if hchain != "B" else "D"
                        lchain = chain.id

            for chain in structure.get_chains():
                if chain.id == hchain:
                    chain.id = "H"
                elif chain.id == lchain:
                    chain.id = "L"

            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            pdb_io.save(str(truncated_dir.joinpath(pdb_file.name)), _HLSelect())
