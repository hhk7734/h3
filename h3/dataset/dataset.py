import json
import logging
import math
from pathlib import Path
from typing import Dict

import torch
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from torch.nn import functional as F
from torch.utils.data import Dataset

LOG = logging.getLogger(__name__)

AMINO_TO_NUM = {
    "A": 0,  # Alanine
    "C": 1,  # Cysteine
    "D": 2,  # Aspartic Acid
    "E": 3,  # Glutamic Acid
    "F": 4,  # Phenylalanine
    "G": 5,  # Glycine
    "H": 6,  # Histidine
    "I": 7,  # Isoleucine
    "K": 8,  # Lysine
    "L": 9,  # Leucine
    "M": 10,  # Methionine
    "N": 11,  # Asparagine
    "P": 12,  # Proline
    "Q": 13,  # Glutamine
    "R": 14,  # Arginine
    "S": 15,  # Serine
    "T": 16,  # Threonine
    "V": 17,  # Valine
    "W": 18,  # Tryptophan
    "Y": 19,  # Tyrosine
    "U": 20,  # Selenocysteine
}


class H3Dataset(Dataset):
    def __init__(self, dataset_dir: Path) -> None:
        super().__init__()

        self._pdb_file_list = list(
            dataset_dir.joinpath("pdb", "truncated").glob("*.pdb")
        )
        self._fasta_dir = dataset_dir.joinpath("fasta")
        self._fasta_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def convert_pdb_to_fasta(pdb: Path, fasta: Path) -> None:
        pdb_id = pdb.stem
        parser = PDBParser()
        structure = parser.get_structure(pdb_id, pdb)

        fasta_str = ""
        for chain in structure.get_chains():
            seq = seq1("".join([residue.resname for residue in chain]))
            description = {"id": pdb_id, "chain": chain.id, "length": len(seq)}
            fasta_str += f">{json.dumps(description)}\n"
            for i in range(0, len(seq), 80):
                fasta_str += f"{seq[i:i + 80]}\n"

        with fasta.open("w") as fd:
            fd.write(fasta_str)

    @staticmethod
    def get_sequences(fasta: Path) -> Dict[str, str]:
        sequences = {}
        for record in SeqIO.parse(fasta, "fasta"):
            # pylint: disable=bare-except
            try:
                description = json.loads(record.description)
                chain_id = description["chain"]
            except:
                LOG.warning("description is not json format in %s", fasta)
                chain_id = record.description[:2]

            sequences[chain_id] = record.seq.upper()
        return sequences

    @staticmethod
    def _generate_dist_matrix(
        coords: torch.Tensor,
        mask: torch.Tensor,
        mask_fill_value: float = -1,
    ):
        coords = coords.unsqueeze(0)
        dist_mat_shape = (coords.shape[1], coords.shape[1], coords.shape[2])
        row_expand = coords.transpose(0, 1).expand(dist_mat_shape)
        col_expand = coords.expand(dist_mat_shape)
        dist_mat = (row_expand - col_expand).norm(dim=2)

        n = len(mask)
        not_mask = torch.ones(n).type(dtype=mask.dtype) - mask
        not_mask = (
            not_mask.unsqueeze(0).transpose(0, 1).expand(n, n).add(not_mask)
        )
        dist_mat[not_mask > 0] = mask_fill_value

        return dist_mat

    @staticmethod
    def _generate_cb_cb_dihedral(
        ca_coords: torch.Tensor,
        cb_coords: torch.Tensor,
        mask: torch.Tensor,
        mask_fill_value: float = -1,
    ):
        mat_shape = (ca_coords.shape[0], ca_coords.shape[0], ca_coords.shape[1])

        b1 = (cb_coords - ca_coords).expand(mat_shape)
        b2 = cb_coords.expand(mat_shape)
        b2 = b2.transpose(0, 1) - b2
        b3 = -1 * b1.transpose(0, 1)

        n1 = torch.cross(b1, b2)
        n1 /= n1.norm(dim=2, keepdim=True)
        n2 = torch.cross(b2, b3)
        n2 /= n2.norm(dim=2, keepdim=True)
        m1 = torch.cross(b2 / b2.norm(dim=2, keepdim=True), n1)

        dihedral_mat = torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))
        dihedral_mat *= 180 / math.pi

        mask = mask.expand((len(mask), len(mask)))
        mask = mask & mask.transpose(0, 1)
        dihedral_mat[mask == 0] = mask_fill_value

        return dihedral_mat

    @staticmethod
    def _generate_ca_cb_dihedral(
        ca_coords: torch.Tensor,
        cb_coords: torch.Tensor,
        n_coords: torch.Tensor,
        mask: torch.Tensor,
        mask_fill_value: float = -1,
    ):
        mat_shape = (ca_coords.shape[0], ca_coords.shape[0], ca_coords.shape[1])

        b1 = (ca_coords - n_coords).expand(mat_shape)
        b2 = (cb_coords - ca_coords).expand(mat_shape)
        b3 = cb_coords.expand(mat_shape)
        b3 = b3.transpose(0, 1) - b3

        n1 = torch.cross(b1, b2)
        n1 /= n1.norm(dim=2, keepdim=True)
        n2 = torch.cross(b2, b3)
        n2 /= n2.norm(dim=2, keepdim=True)
        m1 = torch.cross(b2 / b2.norm(dim=2, keepdim=True), n1)

        dihedral_mat = torch.atan2(
            (m1 * n2).sum(-1), (n1 * n2).sum(-1)
        ).transpose(0, 1)
        dihedral_mat *= 180 / math.pi

        mask = mask.expand((len(mask), len(mask)))
        mask = mask & mask.transpose(0, 1)
        dihedral_mat[mask == 0] = mask_fill_value

        return dihedral_mat

    @staticmethod
    def _generate_ca_cb_cb_planar(
        ca_coords: torch.Tensor,
        cb_coords: torch.Tensor,
        mask: torch.Tensor,
        mask_fill_value: float = -1,
    ):
        mat_shape = (ca_coords.shape[0], ca_coords.shape[0], ca_coords.shape[1])

        v1 = (ca_coords - cb_coords).expand(mat_shape)
        v2 = cb_coords.expand(mat_shape)
        v2 = v2.transpose(0, 1) - v2

        planar_mat = (v1 * v2).sum(-1) / (v1.norm(dim=2) * v2.norm(dim=2))
        planar_mat = torch.acos(planar_mat).transpose(0, 1)
        planar_mat *= 180 / math.pi

        mask = mask.expand((len(mask), len(mask)))
        mask = mask & mask.transpose(0, 1)
        planar_mat[mask == 0] = mask_fill_value

        return planar_mat

    @classmethod
    def get_label_from_pdb(cls, pdb: Path):
        parser = PDBParser()
        structure = parser.get_structure(pdb.stem, pdb)
        residues = list(structure.get_residues())

        def get_cb_or_ca_coord(residue):
            if "CB" in residue:
                return residue["CB"].get_coord()

            if "CA" in residue:
                return residue["CA"].get_coord()

            return [0, 0, 0]

        def get_atom_coord(residue, atom_type):
            if atom_type in residue:
                return residue[atom_type].get_coord()
            return [0, 0, 0]

        cb_ca_coords = torch.tensor([get_cb_or_ca_coord(r) for r in residues])
        ca_coords = torch.tensor([get_atom_coord(r, "CA") for r in residues])
        cb_coords = torch.tensor([get_atom_coord(r, "CB") for r in residues])
        n_coords = torch.tensor([get_atom_coord(r, "N") for r in residues])

        cb_mask = torch.ByteTensor([1 if sum(_) != 0 else 0 for _ in cb_coords])
        mask = torch.ByteTensor([1] * len(cb_coords))

        output_matrix = torch.stack(
            [
                cls._generate_dist_matrix(cb_ca_coords, mask=mask),
                cls._generate_cb_cb_dihedral(
                    ca_coords, cb_coords, mask=(mask & cb_mask)
                ),
                cls._generate_ca_cb_dihedral(
                    ca_coords, cb_coords, n_coords, mask=(mask & cb_mask)
                ),
                cls._generate_ca_cb_cb_planar(
                    ca_coords, cb_coords, mask=(mask & cb_mask)
                ),
            ]
        ).type(torch.float)

        return output_matrix

    @staticmethod
    def get_h3_cdr_indices(pdb: Path):
        parser = PDBParser()
        structure = parser.get_structure(pdb.stem, pdb)
        chain = structure[0]["H"]
        min_i = 200
        max_i = 0
        for i, res in enumerate(chain):
            if 95 <= res.id[1] <= 102:
                if i < min_i:
                    min_i = i
                if i > max_i:
                    max_i = i

        return torch.Tensor([min_i, max_i]).type(torch.int)

    @staticmethod
    def get_dist_bins(num_bins):
        first_bin = 4
        bins = [
            (first_bin + 0.5 * i, first_bin + 0.5 + 0.5 * i)
            for i in range(num_bins - 2)
        ]
        bins.append((bins[-1][1], float("Inf")))
        bins.insert(0, (0, first_bin))
        return bins

    @staticmethod
    def get_omega_bins(num_bins):
        first_bin = -180
        bin_width = 2 * 180 / num_bins
        bins = [
            (first_bin + bin_width * i, first_bin + bin_width * (i + 1))
            for i in range(num_bins)
        ]
        return bins

    @staticmethod
    def get_theta_bins(num_bins):
        first_bin = -180
        bin_width = 2 * 180 / num_bins
        bins = [
            (first_bin + bin_width * i, first_bin + bin_width * (i + 1))
            for i in range(num_bins)
        ]
        return bins

    @staticmethod
    def get_phi_bins(num_bins):
        first_bin = 0
        bin_width = 180 / num_bins
        bins = [
            (first_bin + bin_width * i, first_bin + bin_width * (i + 1))
            for i in range(num_bins)
        ]
        return bins

    @staticmethod
    def get_bin_values(bins):
        bin_values = [t[0] for t in bins]
        bin_width = (bin_values[2] - bin_values[1]) / 2
        bin_values = [v + bin_width for v in bin_values]
        bin_values[0] = bin_values[1] - 2 * bin_width
        return bin_values

    @classmethod
    def bin_dist_angle_matrix(cls, dist_angle_mat, num_bins=26):
        dist_bins = cls.get_dist_bins(num_bins)
        omega_bins = cls.get_omega_bins(num_bins)
        theta_bins = cls.get_theta_bins(num_bins)
        phi_bins = cls.get_phi_bins(num_bins)

        binned_matrix = torch.zeros(dist_angle_mat.shape, dtype=torch.long)
        for i, (lower_bound, upper_bound) in enumerate(dist_bins):
            bin_mask = (dist_angle_mat[0] >= lower_bound).__and__(
                dist_angle_mat[0] < upper_bound
            )
            binned_matrix[0][bin_mask] = i
        for i, (lower_bound, upper_bound) in enumerate(omega_bins):
            bin_mask = (dist_angle_mat[1] >= lower_bound).__and__(
                dist_angle_mat[1] < upper_bound
            )
            binned_matrix[1][bin_mask] = i
        for i, (lower_bound, upper_bound) in enumerate(theta_bins):
            bin_mask = (dist_angle_mat[2] >= lower_bound).__and__(
                dist_angle_mat[2] < upper_bound
            )
            binned_matrix[2][bin_mask] = i
        for i, (lower_bound, upper_bound) in enumerate(phi_bins):
            bin_mask = (dist_angle_mat[3] >= lower_bound).__and__(
                dist_angle_mat[3] < upper_bound
            )
            binned_matrix[3][bin_mask] = i

        return binned_matrix

    def __getitem__(self, index):
        for i in range(5):
            pdb = self._pdb_file_list[(index + i) % len(self._pdb_file_list)]
            fasta = self._fasta_dir.joinpath(pdb.stem + ".fasta")
            if not fasta.exists():
                self.convert_pdb_to_fasta(pdb=pdb, fasta=fasta)

            seq_list = self.get_sequences(fasta=fasta)
            if len(seq_list) != 2:
                LOG.warning("skip %s, it has %d chains", fasta, len(seq_list))
                continue

            seq_num_list = []
            for seq in seq_list.values():
                seq_num = []
                for amino in seq:
                    seq_num.append(AMINO_TO_NUM[amino])
                seq_num_list.append(torch.Tensor(seq_num).type(torch.uint8))

            dist_angle = self.get_label_from_pdb(pdb=pdb)
            bin_mat = self.bin_dist_angle_matrix(dist_angle)

            h3 = self.get_h3_cdr_indices(pdb=pdb)

            return (
                pdb.stem,
                F.one_hot(seq_num_list[0].long()),
                F.one_hot(seq_num_list[1].long()),
                bin_mat,
                h3,
            )

    def __len__(self) -> int:
        return len(self._pdb_file_list)

    @staticmethod
    def merge_samples_to_minibatch(samples):
        samples.sort(key=lambda x: len(x[2]), reverse=True)
        return H3AntibodyBatch(zip(*samples)).data()


class H3AntibodyBatch:
    def __init__(self, batch_data):
        (
            self.id_,
            self.heavy_prim,
            self.light_prim,
            self.dist_angle_mat,
            self.h3,
        ) = batch_data

    @staticmethod
    def pad_data_to_same_shape(tensor_list, pad_value=0):
        shapes = torch.Tensor([_.shape for _ in tensor_list])
        target_shape = torch.max(shapes.transpose(0, 1), dim=1)[0].int()

        padded_dataset_shape = [len(tensor_list)] + list(target_shape)
        padded_dataset = torch.Tensor(*padded_dataset_shape)
        for i, data in enumerate(tensor_list):
            padding = reversed(
                target_shape - torch.Tensor(list(data.shape)).int()
            )

            padding = F.pad(padding.unsqueeze(0).t(), (1, 0, 0, 0)).view(-1, 1)
            padding = padding.view(1, -1)[0].tolist()

            padded_data = F.pad(data, padding, value=pad_value)
            padded_dataset[i] = padded_data

        return padded_dataset

    def data(self):
        return self.features(), self.labels()

    def features(self):
        """Gets the one-hot encoding of the sequences with a feature that
        delimits the chains"""
        X = [torch.cat(_, 0) for _ in zip(self.heavy_prim, self.light_prim)]
        X = self.pad_data_to_same_shape(X, pad_value=0)

        X = F.pad(X, (0, 1, 0, 0, 0, 0))
        for i, h_prim in enumerate(self.heavy_prim):
            X[i, len(h_prim) - 1, X.shape[2] - 1] = 1

        return X.transpose(1, 2).contiguous()

    def labels(self):
        """Gets the distance matrix data of the batch with -1 padding"""
        label_mat = (
            self.pad_data_to_same_shape(self.dist_angle_mat, pad_value=-1)
            .transpose(0, 1)
            .long()
        )

        return label_mat

    def batch_mask(self):
        """Gets the mask data of the batch with zero padding"""
        """Code to use when masks are added
        masks = self.mask
        masks = pad_data_to_same_shape(masks, pad_value=0)
        return masks
        """
        raise NotImplementedError("Masks have not been added to antibodies yet")
