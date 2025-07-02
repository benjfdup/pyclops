import torch
from pyclops import ChemicalLossHandler
from pyclops.utils.constants import KB

chig_pdb = "/Users/bendupontjr/mphil_files/all_mphil_code/pyclops/examples/pdbs/chignolin.pdb"
n_atoms = 166
n_batch = 10

chem_loss_handler = ChemicalLossHandler.from_pdb_file(chig_pdb, units_factor = 1.0)

positions = torch.randn(n_batch,n_atoms, 3)

opt_vertices = chem_loss_handler._all_vertices

true_vertices = []
for loss in chem_loss_handler.chemical_losses:
    true_vertices.append(loss.vertex_indices)
true_vertices = torch.stack(true_vertices)

opt_distances = chem_loss_handler._compute_distances(positions, opt_vertices)
true_distances = []
for loss in chem_loss_handler.chemical_losses:
    true_distances.append(loss._compute_distances(positions, loss.vertex_indices))
true_distances = torch.stack(true_distances, dim=1) #[n_batch, n_losses, 6]  # SAME UP TO HERE.

opt_raw_losses = torch.zeros(n_batch, chem_loss_handler.n_losses, dtype=torch.float, device=chem_loss_handler._device) # [n_batch, n_losses]
for kde_idx, kde in enumerate(chem_loss_handler._kde_list):
    original_indices, _ = chem_loss_handler._kde_groups[kde] # [n_loss_subset, ]; the original vertex indices for this kde group

    start_idx = chem_loss_handler._kde_start_indices[kde_idx]
    length = chem_loss_handler._kde_lengths[kde_idx]
    end_idx = start_idx + length

    opt_distances_subset = opt_distances[:, start_idx:end_idx, :]
    n_loss_subset = opt_distances_subset.shape[1]
    opt_distances_subset_flat = opt_distances_subset.reshape(-1, 6)
    
    kde_values_flat = kde.score_samples(opt_distances_subset_flat) # shape [n_batch * n_loss_subset, ]
    kde_values = kde_values_flat.reshape(n_batch, n_loss_subset) # shape [n_batch, n_loss_subset]

    opt_raw_losses[:, original_indices] = kde_values # shape [n_batch, n_loss_subset] -> [n_batch, n_losses] at indices

true_raw_losses = torch.zeros(n_batch, chem_loss_handler.n_losses, dtype=torch.float, device=chem_loss_handler._device) # [n_batch, n_losses]
for indx in range(chem_loss_handler.n_losses):
    loss = chem_loss_handler.chemical_losses[indx]
    true_raw_losses[:, indx] = loss.kde_pdf.score_samples(true_distances[:, indx, :])

# SAME UP TO HERE. IGNORE WEIGHTS AND OFFSETS FOR NOW (THEY SHOULD BE HAVING NO EFFECT RIGHT NOW, W=1, O=0)

opt_raw_resonance_losses = torch.zeros(n_batch, chem_loss_handler.n_resonance_groups, dtype=torch.float, device=chem_loss_handler._device) # [n_batch, n_resonance_groups]
for group_idx in range(chem_loss_handler.n_resonance_groups):
    group_mask = (chem_loss_handler._resonance_groups == group_idx) # shape [n_losses, ]
    group_losses = opt_raw_losses[:, group_mask] # shape [n_batch, n_group_members]
    opt_raw_resonance_losses[:, group_idx] = torch.min(group_losses, dim=1)[0] # shape [n_batch, ]

# vv the below is never truly used but is here as an experiment vv
true_raw_resonance_losses = torch.zeros(n_batch, chem_loss_handler.n_resonance_groups, dtype=torch.float, device=chem_loss_handler._device) # [n_batch, n_resonance_groups]
for group_idx in range(chem_loss_handler.n_resonance_groups):
    # Find all losses belonging to this resonance group
    group_mask = (chem_loss_handler._resonance_groups == group_idx) # shape [n_losses, ]
    group_losses = true_raw_losses[:, group_mask] # shape [n_batch, n_group_members]
            
    # Take minimum over the group
    true_raw_resonance_losses[:, group_idx] = torch.min(group_losses, dim=1)[0] # shape [n_batch, ]

# SAME UP TO HERE.

opt_tempered_resonance_losses = -KB * chem_loss_handler._temp * opt_raw_resonance_losses

true_tempered_raw_losses = -KB * chem_loss_handler._temp * true_raw_losses
true_tempered_resonance_losses = torch.zeros(n_batch, chem_loss_handler.n_resonance_groups, dtype=torch.float, device=chem_loss_handler._device) # [n_batch, n_resonance_groups]
for group_idx in range(chem_loss_handler.n_resonance_groups):
    group_mask = (chem_loss_handler._resonance_groups == group_idx) # shape [n_losses, ]
    group_losses = true_tempered_raw_losses[:, group_mask] # shape [n_batch, n_group_members]
    true_tempered_resonance_losses[:, group_idx] = torch.min(group_losses, dim=1)[0] # shape [n_batch, ]

print("finished!")