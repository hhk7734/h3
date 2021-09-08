import logging
import sys

from ...dataset.sabdab import SAbDab
from . import Arguments

args = Arguments()
rest_args = args.parse()

LOG = logging.getLogger(__name__)

LOG.debug(args)
if len(rest_args) != 0:
    LOG.error("Unparsed arguments exist. %s", rest_args)
    sys.exit(1)

try:
    summary_tsv = args.dataset_dir.joinpath("summary.tsv")
    if not summary_tsv.exists():
        SAbDab.download_non_redundant_summary(output_file=summary_tsv)

    pdb_dir = args.dataset_dir.joinpath("pdb")
    if not pdb_dir.exists():
        summary = SAbDab.read_summary(summary_file=summary_tsv)
        SAbDab.download_pdbs(summary=summary, pdb_dir=pdb_dir)

    truncated_pdb_dir = pdb_dir.joinpath("truncated")
    if not truncated_pdb_dir.exists():
        summary = SAbDab.read_summary(summary_file=summary_tsv)
        SAbDab.truncate_pdbs(summary=summary, pdb_dir=pdb_dir)

except Exception as e:
    LOG.exception(e)
    sys.exit(1)
