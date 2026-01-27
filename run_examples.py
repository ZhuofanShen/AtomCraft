import subprocess
import json

afdb_dir = "~/alphafold3/alphafold3_data_save"
hmmer_path = "/home/jupyter-yehlin/.conda/envs/alphafold3_venv"


# Example 1: protein with msa (specify the name, pdb_target_ids, target_type, use_msa)
cmd = [
    "python", "boltzdesign.py",
    "--name", "8znl",
    "--target_type", "protein",
    "--pdb_target_ids", "A",
    "--target_seq", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--use_msa", "True",
    "--msa_max_seqs", "4096",
    "--design_samples", "1",
    "--gpu_id", "0",
    "--suffix", "boltz2_trial1",
    "--semi_greedy_steps", "1",
]
subprocess.run(cmd)


# Example 1-1: protein with msa (specify the name, pdb_target_ids, target_type, use_msa) and repredict only iptm > iptm_cutoff
cmd = [
    "python", "boltzdesign.py",
    "--name", "8znl",
    "--target_type", "protein",
    "--pdb_target_ids", "A",
    "--target_seq", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--use_msa", "True",
    "--msa_max_seqs", "4096",
    "--design_samples", "1",
    "--gpu_id", "0",
    "--suffix", "boltz2_trial1",
    "--semi_greedy_steps", "1",
    "--high_iptm", "True",
    "--i_ptm_cutoff", "0.7",
]
subprocess.run(cmd)


# Example 2: boltz2 with template
cmd = [
    "python", "boltzdesign.py",
    "--name", "8znl",
    "--target_type", "protein",
    "--pdb_target_ids", "A",
    "--use_template", "True",
    "--pdb_path", "8znl",
    "--pdb_target_ids", "B",
    "--design_samples", "1",
    "--gpu_id", "0",
    "--suffix", "boltz2_trial1",
    "--semi_greedy_steps", "1",
]
subprocess.run(cmd)

# Example 3: boltz1 with msa
cmd = [
    "python", "boltzdesign.py",
    "--name", "8znl",
    "--target_type", "protein",
    "--pdb_target_ids", "A",
    "--use_msa", "True",
    "--target_seq", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--msa_max_seqs", "4096",
    "--design_samples", "1",
    "--gpu_id", "0",
    "--boltz_model_version", "boltz1",
    "--suffix", "boltz1_trial1_v2",
    "--semi_greedy_steps", "1",
]
subprocess.run(cmd)



# Example 4: motif scaffolding
cmd = [
    "python", "boltzdesign.py",
    "--name", "8vc8", 
    "--target_type", "small_molecule",
    "--target_seq", "HEM",
    "--pdb_motif_id", "A",
    "--pdb_path", "8vc8",
    "--design_samples", "5",
    "--gpu_id", "0",
    "--suffix", "boltz2_motif_scaffolding",
    "--use_template", "True",
    "--motif_scaffolding", "True",
    "--length_min", "140",
    "--length_max", "150",
    "--motifs", '[{"start_pos": 30, "end_pos": 47}, {"start_pos": 81, "end_pos": 173}]',
    "--min_motif_gap", "15",
    "--af3_docker_name", "alphafold3_yc",
    "--af3_database_settings", afdb_dir,
    "--af3_hmmer_path", hmmer_path,
    "--boltz_model_version", "boltz2",
]
subprocess.run(cmd)


# Example 5 : small molecule PDB
cmd = [
    "python", "boltzdesign.py",
    "--name", "7v11",
    "--target_type", "small_molecule",
    "--target_seq", "OQO",
    "--gpu_id", "0",
    "--design_samples", "1",
    "--suffix", "boltz2_trial1",
    "--semi_greedy_steps", "0",
]
subprocess.run(cmd)

# Example 6 : small molecule PDB
cmd = [
    "python", "boltzdesign.py",
    "--name", "7v11",
    "--target_type", "small_molecule",
    "--target_seq", "OQO",
    "--gpu_id", "0",
    "--design_samples", "1",
    "--boltz_model_version", "boltz1",
    "--suffix", "boltz1_trial1",
    "--semi_greedy_steps", "0",
]
subprocess.run(cmd)

# Example 7 : small molecule PDB
cmd = [
    "python", "boltzdesign.py",
    "--name", "3wc0",
    "--target_type", "small_molecule",
    "--target_seq", "GTP",
    "--gpu_id", "0",
    "--design_samples", "1",
    "--suffix", "boltz2_trial1",
    "--semi_greedy_steps", "1",
]
subprocess.run(cmd)


# Example 6 : DNA/RNA PDB (specify the name, pdb_target_ids, target_type)
cmd = [
    "python", "boltzdesign.py",
    "--name", "5zmc",
    "--target_type", "dna",
    "--target_seq", "GCCCTTCCGGGTCCCC,CGGGGACCCGGAAGGG",
    "--gpu_id", "0",
    "--design_samples", "1",
    "--suffix", "boltz2_trial1",
    "--semi_greedy_steps", "0",
]
subprocess.run(cmd)