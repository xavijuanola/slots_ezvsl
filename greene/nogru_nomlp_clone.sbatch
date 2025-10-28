#!/bin/bash

#SBATCH --account=pr_230_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --mem=180GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=nogru_nomlp_clone
#SBATCH -o/scratch/xj2376/jobs/slots_ezvsl/nogru_nomlp_clone.out
#SBATCH -e/scratch/xj2376/jobs/slots_ezvsl/nogru_nomlp_clone.err

module purge

# Download CA certificates
CACERT_PATH="/scratch/xj2376/cacert.pem"
curl --silent --show-error --location -o "$CACERT_PATH" https://curl.se/ca/cacert.pem

export SSL_CERT_FILE="$CACERT_PATH"

/scratch/xj2376/code/ssl-xavi_singu/sing \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_1.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_2.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_3.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_4.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_5.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_6.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_7.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_8.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_9.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_10.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_11.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_12.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_audio1.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_audio2.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_audio3.sqf:ro \
--overlay /vast/xj2376/vggsound_hpc/sqf_files/vggsound_audio_similar.sqf:ro << EOF

cd /scratch/xj2376/code/slots_ezvsl

bash scripts/slots/nogru_nomlp_clone.sh

EOF