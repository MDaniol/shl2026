#!/usr/bin/env bash
# Fire-and-forget E-FMDIV round: submit all extraction + consumption jobs with SLURM dependencies,
# so the whole pipeline runs unattended. Run from the repo root AFTER `git pull`:
#
#     cd ~/shl2026 && git pull && bash notebooks/mdaniol/hpc/run_fmdiv_round.sh
#
# DAG (afterok — downstream runs only if its extraction succeeds):
#   [GPU] imagebind extract --.--> [CPU] bake-off (clean deterministic table + imagebind_V0)
#                              \--> [CPU] vote {utica_V2, mantisv2_V1, imagebind_V0}
#   [GPU] dinov2 extract ------> [GPU] cross-channel head (dinov2_V2_pc)
#   [GPU] ast extract ---------> [GPU] cross-channel head (ast_V2_pc)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
H=notebooks/mdaniol/hpc
sub() { sbatch --parsable "$@"; }   # echo job id

# clean the mixed bake-off table so the deterministic re-run starts fresh (also closes #18)
[ -f notebooks/mdaniol/BAKEOFF_SPLIT.md ] && \
  mv notebooks/mdaniol/BAKEOFF_SPLIT.md "notebooks/mdaniol/BAKEOFF_SPLIT.premix.$(date +%Y%m%d_%H%M).md"

# --- GPU extractions (independent, run in parallel) ---
IB=$(sub $H/extract_imagebind_helios.sbatch)                 # ~9 min  -> imagebind_V0
DINO=$(MODEL=dinov2 sub $H/extract_vit_helios.sbatch)        # ~2 h    -> dinov2_V2_pc
AST=$(sub $H/extract_ast_helios.sbatch)                      # ~9 h    -> ast_V2_pc

# --- cross-channel SE heads (GPU, after their extraction) ---
HDINO=$(EMB_PC=dinov2_V2_pc sub --dependency=afterok:$DINO $H/head_xchannel_helios.sbatch)
HAST=$(EMB_PC=ast_V2_pc     sub --dependency=afterok:$AST  $H/head_xchannel_helios.sbatch)

# --- ImageBind -> bake-off + vote (CPU, after ib extraction; both only need imagebind_V0) ---
BO=$(sub --dependency=afterok:$IB $H/probe_fusion_ares.sbatch)
VOTE=$(EMBS=utica_V2,mantisv2_V1,imagebind_V0 sub --dependency=afterok:$IB $H/voting_head.sbatch)

printf 'IB=%s DINO=%s AST=%s | HDINO=%s HAST=%s | BO=%s VOTE=%s\n' \
  "$IB" "$DINO" "$AST" "$HDINO" "$HAST" "$BO" "$VOTE" | tee notebooks/mdaniol/round_jobids.txt
echo "--- queue (deps shown in last column) ---"
squeue --me -o "%.14i %.18j %.9T %.11M %.18E"
echo "Come back to: HEAD_RESULTS_ast_V2_pc.md, HEAD_RESULTS_dinov2_V2_pc.md, BAKEOFF_SPLIT.md, VOTING_HEAD_RESULTS.md"
