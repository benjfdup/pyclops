from abc import ABCMeta
from typing import final
import math

from rdkit import Chem
from rdkit.Chem import AllChem

from ..loss_structure_modifier import LossStructureModifier
from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ....losses.carboxylic_carbo import (
    AspGlu,
    AspAsp,
    GluGlu,
    AspCTerm,
    GluCTerm,
)

class CarboxylicCarboxylicModifier(LossStructureModifier, metaclass=ABCMeta):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    
    @final
    def _inner_mod(self,
                initial_parsed_mol: Chem.Mol,
                o1_idx: int,
                o2_idx: int,
                ) -> Chem.Mol:
        """
        Inner functionality to modify the structure according to the corresponding ChemicalLoss.
        """
        
        emol = Chem.EditableMol(initial_parsed_mol)

        # Create nitrogen atoms to replace the oxygens
        n1 = Chem.Atom('N')
        n2 = Chem.Atom('N')
        
        # Replace the oxygen atoms with nitrogen atoms
        emol.ReplaceAtom(o1_idx, n1)
        emol.ReplaceAtom(o2_idx, n2)

        outer_1 = emol.AddAtom(Chem.Atom('C'))
        outer_2 = emol.AddAtom(Chem.Atom('C'))

        emol.AddBond(outer_1, o1_idx, Chem.BondType.SINGLE)
        # NOTE: NOT adding the bond between outer_2 and o2_idx here - saving for last

        bc1 = emol.AddAtom(Chem.Atom('C')) # bonded to outer_1
        bc2 = emol.AddAtom(Chem.Atom('C'))
        bc3 = emol.AddAtom(Chem.Atom('C'))
        bc4 = emol.AddAtom(Chem.Atom('C')) # bounded to outer_2
        bc5 = emol.AddAtom(Chem.Atom('C'))
        bc6 = emol.AddAtom(Chem.Atom('C'))

        emol.AddBond(bc1, bc2, Chem.BondType.AROMATIC)
        emol.AddBond(bc2, bc3, Chem.BondType.AROMATIC)
        emol.AddBond(bc3, bc4, Chem.BondType.AROMATIC)
        emol.AddBond(bc4, bc5, Chem.BondType.AROMATIC)
        emol.AddBond(bc5, bc6, Chem.BondType.AROMATIC)
        emol.AddBond(bc6, bc1, Chem.BondType.AROMATIC)

        emol.AddBond(bc1, outer_1, Chem.BondType.SINGLE)
        emol.AddBond(bc4, outer_2, Chem.BondType.SINGLE)  # Add this bond now
        
        # Get the intermediate molecule and sanitize it
        intermediate_mol = emol.GetMol()

        # normalized midpoint of n1 and n2 -- write code here... save it as a tuple (x, y, z)
        conf = intermediate_mol.GetConformer()
        n1_pos = conf.GetAtomPosition(o1_idx)  # n1 replaced o1_idx
        n2_pos = conf.GetAtomPosition(o2_idx)  # n2 replaced o2_idx
        
        # Calculate midpoint
        midpoint_x = (n1_pos.x + n2_pos.x) / 2.0
        midpoint_y = (n1_pos.y + n2_pos.y) / 2.0
        midpoint_z = (n1_pos.z + n2_pos.z) / 2.0
        
        # Normalize to unit length
        magnitude = (midpoint_x**2 + midpoint_y**2 + midpoint_z**2)**0.5
        normalized_midpoint = (midpoint_x / magnitude, midpoint_y / magnitude, midpoint_z / magnitude)
        base_offset_coeff = 0.0
        base_offset_const = 1.2 #1.5
        #base_offset = (normalized_midpoint[0] * base_offset_coeff, normalized_midpoint[1] * base_offset_coeff, normalized_midpoint[2] * base_offset_coeff)
        base_offset = (base_offset_coeff * midpoint_x + base_offset_const * normalized_midpoint[0], 
                       base_offset_coeff * midpoint_y + base_offset_const * normalized_midpoint[1], 
                       base_offset_coeff * midpoint_z + base_offset_const * normalized_midpoint[2])

        # Here, we add positions to the atoms
        conf.SetAtomPosition(outer_1, tuple(n1_pos[i] + base_offset[i] for i in range(3)))
        conf.SetAtomPosition(outer_2, tuple(n2_pos[i] + base_offset[i] for i in range(3)))

        ring_radius = 1.39
        ring_center = ((conf.GetAtomPosition(outer_1).x + conf.GetAtomPosition(outer_2).x) / 2,
                      (conf.GetAtomPosition(outer_1).y + conf.GetAtomPosition(outer_2).y) / 2,
                      (conf.GetAtomPosition(outer_1).z + conf.GetAtomPosition(outer_2).z) / 2)
        
        # place bc1 and bc4 along the outer_1 - outer_2 line, each with a distance of ring_radius from the center
        # with bc1 being closer to outer_1 and bc4 being closer to outer_2
        outer_1_pos = conf.GetAtomPosition(outer_1)
        outer_2_pos = conf.GetAtomPosition(outer_2)
        direction = (outer_2_pos.x - outer_1_pos.x, outer_2_pos.y - outer_1_pos.y, outer_2_pos.z - outer_1_pos.z)
        magnitude = (direction[0]**2 + direction[1]**2 + direction[2]**2)**0.5
        norm_dir = (direction[0]/magnitude, direction[1]/magnitude, direction[2]/magnitude)
        
        conf.SetAtomPosition(bc4, (ring_center[0] + norm_dir[0] * ring_radius, ring_center[1] + norm_dir[1] * ring_radius, ring_center[2] + norm_dir[2] * ring_radius))
        conf.SetAtomPosition(bc1, (ring_center[0] - norm_dir[0] * ring_radius, ring_center[1] - norm_dir[1] * ring_radius, ring_center[2] - norm_dir[2] * ring_radius))

        angles_in_degrees = [60, 120, 240, 300]
        points = [bc2, bc3, bc5, bc6]
        first_axis = (outer_1_pos.x - ring_center[0], outer_1_pos.y - ring_center[1], outer_1_pos.z - ring_center[2])
        origin_to_midpoint = (ring_center[0], ring_center[1], ring_center[2])
        second_axis = (first_axis[1] * origin_to_midpoint[2] - first_axis[2] * origin_to_midpoint[1], 
                      first_axis[2] * origin_to_midpoint[0] - first_axis[0] * origin_to_midpoint[2], 
                      first_axis[0] * origin_to_midpoint[1] - first_axis[1] * origin_to_midpoint[0])
        
        # Normalize the axes
        first_mag = (first_axis[0]**2 + first_axis[1]**2 + first_axis[2]**2)**0.5
        second_mag = (second_axis[0]**2 + second_axis[1]**2 + second_axis[2]**2)**0.5
        norm_first = (first_axis[0]/first_mag, first_axis[1]/first_mag, first_axis[2]/first_mag)
        norm_second = (second_axis[0]/second_mag, second_axis[1]/second_mag, second_axis[2]/second_mag)
        
        for i, angle in enumerate(angles_in_degrees):
            # remember, the circle will take the form of (cos(angle) * normalized_first_axis + sin(angle) * normalized_second_axis) * radius + ring_center
            angle_rad = math.radians(angle)
            cos_val = math.cos(angle_rad)
            sin_val = math.sin(angle_rad)
            
            x = ring_center[0] + ring_radius * (cos_val * norm_first[0] + sin_val * norm_second[0])
            y = ring_center[1] + ring_radius * (cos_val * norm_first[1] + sin_val * norm_second[1])
            z = ring_center[2] + ring_radius * (cos_val * norm_first[2] + sin_val * norm_second[2])
            
            conf.SetAtomPosition(points[i], (x, y, z))

        # distance needs to be 1.51 -- need to decrease outer bonds to inner ring by 0.684 EACH (in total 1.368)
        for outer, inner in [(outer_1, bc1), (outer_2, bc4)]:
            outer_pos, inner_pos = conf.GetAtomPosition(outer), conf.GetAtomPosition(inner)
            direction = tuple((inner_pos[i] - outer_pos[i]) for i in range(3))
            magnitude = sum(d**2 for d in direction)**0.5
            offset = tuple(d / magnitude * 0.684 for d in direction)
            conf.SetAtomPosition(outer, tuple(outer_pos[i] + offset[i] for i in range(3)))


        # IMPLEMENT HERE:
        # 1. GET THE DISPLACEMENT VECTOR BETWEEN THE BENZENE RING CENTER AND THE NITROGEN MIDPOINT.
        nitrogen_midpoint = (midpoint_x, midpoint_y, midpoint_z)
        displacement_vector = (ring_center[0] - nitrogen_midpoint[0], 
                              ring_center[1] - nitrogen_midpoint[1], 
                              ring_center[2] - nitrogen_midpoint[2])
        
        # 2. GET THE COMPONENT OF THAT VECTOR THAT IS ALONG THE LINE BETWEEN THE NITROGENS.
        n1_to_n2_vector = (n2_pos.x - n1_pos.x, n2_pos.y - n1_pos.y, n2_pos.z - n1_pos.z)
        n1_to_n2_magnitude = (n1_to_n2_vector[0]**2 + n1_to_n2_vector[1]**2 + n1_to_n2_vector[2]**2)**0.5
        n1_to_n2_unit = (n1_to_n2_vector[0]/n1_to_n2_magnitude, 
                         n1_to_n2_vector[1]/n1_to_n2_magnitude, 
                         n1_to_n2_vector[2]/n1_to_n2_magnitude)
        
        # Dot product to get the component along the nitrogen line
        dot_product = (displacement_vector[0] * n1_to_n2_unit[0] + 
                      displacement_vector[1] * n1_to_n2_unit[1] + 
                      displacement_vector[2] * n1_to_n2_unit[2])
        
        # Component vector along the nitrogen line
        component_along_nitrogen_line = (dot_product * n1_to_n2_unit[0], 
                                        dot_product * n1_to_n2_unit[1], 
                                        dot_product * n1_to_n2_unit[2])
        
        # 3. SUBTRACT THAT COMPONENT FROM THE OUTER CARBONS AND ALL CARBONS IN THE BENZENE RING (BC1-BC6)
        atoms_to_adjust = [outer_1, outer_2, bc1, bc2, bc3, bc4, bc5, bc6]
        
        for atom_idx in atoms_to_adjust:
            current_pos = conf.GetAtomPosition(atom_idx)
            new_pos = (current_pos.x - component_along_nitrogen_line[0],
                      current_pos.y - component_along_nitrogen_line[1], 
                      current_pos.z - component_along_nitrogen_line[2])
            conf.SetAtomPosition(atom_idx, new_pos)

        Chem.SanitizeMol(intermediate_mol)

        # Now add the final bond between outer_2 and o2_idx (which is now nitrogen)
        emol_final = Chem.EditableMol(intermediate_mol)
        emol_final.AddBond(outer_2, o2_idx, Chem.BondType.SINGLE)
        
        # Get the final molecule and sanitize it
        final_mol = emol_final.GetMol()
        Chem.SanitizeMol(final_mol)
        final_mol = Chem.RemoveHs(final_mol)

        final_mol = self._relax_atom_subset(final_mol, [bc1, bc2, bc3, bc4, bc5, bc6, outer_1, outer_2])
        
        return final_mol
    
    @final
    def _outer_mod(self,
                   chemical_loss: ChemicalLoss,
                   initial_parsed_mol: Chem.Mol,
                   mdtraj_atom_indexes_dict: AtomIndexDict,
                   rdkit_atom_indexes_dict: AtomIndexDict,
                   ) -> Chem.Mol:
        """
        To be used in the `_mod_struct` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It handles the logic for determining
        which oxygen to remove when forming a carboxylic acid, then calls `_inner_mod` to perform the actual
        structural modification.
        
        Args:
            chemical_loss: The chemical loss object containing atom indices.
            initial_parsed_mol: The initial parsed molecule.
            mdtraj_atom_indexes_dict: Dictionary mapping atom names to MDTraj atom indices.
            rdkit_atom_indexes_dict: Dictionary mapping atom names to RDKit atom indices.

        Returns:
            Chem.Mol, the modified RDKit molecule
        """
        inverse_mdtraj_atom_indexes_dict = self._invert_dict(mdtraj_atom_indexes_dict)
        o1_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['O1']]
        o2_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['O2']]

        o1_rdkit_idx = rdkit_atom_indexes_dict[o1_key]
        o2_rdkit_idx = rdkit_atom_indexes_dict[o2_key]
        
        return self._inner_mod(initial_parsed_mol, o1_rdkit_idx, o2_rdkit_idx)
    
    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        """
        return self._outer_mod(
            chemical_loss, 
            initial_parsed_mol, 
            mdtraj_atom_indexes_dict, 
            rdkit_atom_indexes_dict,
            )
    
class CarboxylicCarboSide2SideModifier(CarboxylicCarboxylicModifier):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries where a sidechain
    carboxyl group forms a bond with a sidechain carboxyl group.
    """
    pass

class AspGluModifier(CarboxylicCarboSide2SideModifier):
    """
    Aspartate's carboxyl group forms a bond with Glutamate's carboxyl group.
    """
    _method = AspGlu._method

class AspAspModifier(CarboxylicCarboSide2SideModifier):
    """
    Aspartate's carboxyl group forms a bond with itself.
    """
    _method = AspAsp._method

class GluGluModifier(CarboxylicCarboSide2SideModifier):
    """
    Glutamate's carboxyl group forms a bond with itself.
    """
    _method = GluGlu._method

class CarboxylicCarbo2CTermModifier(CarboxylicCarboxylicModifier):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries where a sidechain
    carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    pass

class AspCTermModifier(CarboxylicCarbo2CTermModifier):
    """
    Aspartate's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = AspCTerm._method

class GluCTermModifier(CarboxylicCarbo2CTermModifier):
    """
    Glutamate's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = GluCTerm._method