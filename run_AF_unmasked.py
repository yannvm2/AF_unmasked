#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import pickle
import argparse
from pathlib import Path
from string import ascii_uppercase, ascii_lowercase

# Third-party imports
import ipywidgets as widgets
from Bio import Align, SeqIO, AlignIO
from Bio.PDB.mmcifio import MMCIFIO

# Project-specific imports
from alphafold.data.prepare_templates import *
from alphafold.data.mmseqs_2_uniprot import *
from colabfold.batch import get_msa_and_templates, msa_to_str
from colabfold.utils import DEFAULT_API_SERVER, get_commit
from colabfold.download import download_alphafold_params
from run_alphafold import predict_structure
from alphafold.data import pipeline, pipeline_multimer
from alphafold.data.tools import hmmsearch, jackhmmer
from alphafold.data import templates
from alphafold.model import model, data, config
from alphafold.relax import relax

# Define combined alphabet variable (since it was used but not defined)
ascii_upperlower = ascii_uppercase + ascii_lowercase

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AlphaFold structure prediction locally with custom settings."
    )
    parser.add_argument(
        "--jobname", type=str, default="8JIF",
        help="Name of the job. Used to identify input/output directories."
    )
    parser.add_argument(
        "--input_dir", type=str, default="INPUTS",
        help="Directory containing the input FASTA and template files."
    )
    parser.add_argument(
        "--model_type", type=str, default="alphafold2_multimer_v2",
        choices=["alphafold2_multimer_v2", "alphafold2_multimer_v3"],
        help="Select the AlphaFold model type."
    )
    parser.add_argument(
        "--predictions_per_model", type=int, default=1,
        help="Number of predictions per model."
    )
    parser.add_argument(
        "--num_recycles", type=int, default=10,
        help="Number of recycles during prediction."
    )
    parser.add_argument(
        "--recycle_early_stop_tolerance", type=float, default=0.5,
        help="Tolerance for early stopping during recycles."
    )
    parser.add_argument(
        "--use_dropout", action="store_true",
        help="Enable dropout during evaluation."
    )
    parser.add_argument(
        "--msa_mode", type=str, default="mmseqs2_uniref",
        choices=["no_MSA", "mmseqs2_uniref", "mmseqs2_uniref_env"],
        help="MSA mode selection."
    )
    parser.add_argument(
        "--msa_depth", type=int, default=1,
        help="Depth of the MSA. Set to a lower number to rely more on template information."
    )
    parser.add_argument(
        "--repeat_template", type=str, default="4 times",
        choices=["1 time", "2 times", "3 times", "4 times"],
        help="Number of times to fill template slots."
    )
    parser.add_argument(
        "--models_to_relax", type=str, default="NONE",
        choices=["NONE", "BEST", "ALL"],
        help="Models to relax."
    )

    return parser.parse_args()


RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

def main():
    args = parse_arguments()
    
    print("Configuration completed successfully!")
    
    # Set up directories and file paths based on jobname and input_dir
    jobname = args.jobname
    out_dir = Path(f"{jobname}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target_fasta = out_dir.joinpath(f"{jobname}.fasta")
    
    fasta_path = Path(f"{args.input_dir}/{jobname}/{jobname}.fasta")
    if not fasta_path.is_file():
        sys.exit(f"Error: FASTA file not found at {fasta_path}")
    
    shutil.copyfile(fasta_path, target_fasta)
    
    if not is_fasta(target_fasta):
        raise ValueError("""The input does not appear to be in fasta format.
Example of fasta format input:
> H1142_A
GLEKDFLPLYFGWFLTK...
> H1142_B
EVQLEESGGGLVQAGGS...
""")
    
    # Display FASTA content
    with open(target_fasta, "r") as f:
        print("Fasta sequences:")
        print(f.read())
        print()
    
    # Map sequences to chains using combined alphabet
    seq2chain = {}
    chain_idx = 0
    for record in SeqIO.parse(target_fasta, "fasta"):
        chain_label = ascii_upperlower[chain_idx]
        seq2chain.setdefault(record.seq, []).append(chain_label)
        chain_idx += 1
    
    # Process template file: auto-detect based on extension (.pdb or .cif)
    template_input_dir = Path(f"{args.input_dir}/{jobname}")
    structure_files = [f for f in template_input_dir.iterdir() if f.suffix.lower() in ['.pdb', '.cif']]

    if not structure_files:
        raise ValueError("No template file with extension .pdb or .cif found in the input directory.")

    if len(structure_files) > 1:
        print("Multiple structure files found; using the first one.")

    structure_file = structure_files[0]

    if structure_file.suffix.lower() == ".pdb":
        template_format = "pdb"
    elif structure_file.suffix.lower() == ".cif":
        template_format = "cif"
    else:
        raise ValueError("Template must be in .pdb or .cif format")

    template_dest = out_dir.joinpath(structure_file.name)
    shutil.copyfile(structure_file, template_dest)
    
    # Load and process template data
    template_model = load_PDB(template_dest, is_mmcif=(template_format == "cif"))
    template_chains = [c.id for c in template_model]
    remove_extra_chains(template_model, template_chains)
    remove_hetatms(template_model)
    template_sequences = [get_fastaseq(template_model, chain) for chain in template_chains]
    
    print("Template sequences:")
    for seq, chain in zip(template_sequences, template_chains):
        print(f"Chain {chain}: {seq}")
    
    # Process target data
    target_chains, target_sequences, target_models = get_target_data(
        [str(target_fasta)],
        chains=None,
        is_fasta=True,
        is_mmcif=False,
    )
    if len(target_chains) > len(template_chains):
        raise AssertionError(
            f"Not enough chains in the template structure to cover all target chains. "
            f"Template chains: {template_chains}, Target chains: {target_chains}"
        )
    
    # Set prediction and MSA parameters from arguments
    model_type = args.model_type
    predictions_per_model = args.predictions_per_model
    num_recycles = args.num_recycles
    recycle_early_stop_tolerance = args.recycle_early_stop_tolerance
    use_dropout = args.use_dropout
    msa_mode = args.msa_mode
    msa_depth = args.msa_depth
    inpaint_clashes = True
    align = True
    align_tool = "blast"
    
    # Create dropdowns for template chain selection (UI element for potential GUI use)
    template_preview = [f"Chain {chain} (seq: {seq[:10]}...)" for chain, seq in zip(template_chains, template_sequences)]
    template_c = [widgets.Dropdown(options=template_preview, value=template_preview[i]) for i in range(len(target_chains))]
    
    # Determine template repetition count
    temp_reps = int(args.repeat_template.split()[0])
    template_chains_selected = [temp.value.split()[1] for temp in template_c]
    
    if len(template_chains_selected) != len(set(template_chains_selected)):
        raise ValueError("Must select a different template chain for each fasta sequence")
    
    append = False
    mmcif_path = Path(out_dir, "template_data", "mmcif_files")
    mmcif_path.mkdir(parents=True, exist_ok=True)
    
    # Perform template slot filling and alignment
    for rep in range(temp_reps):
        print(f"Filling template slot n. {rep+1}...")
        next_id = get_next_id(mmcif_path) if append else "0000"
        io = MMCIFIO()
        template_mmcif_path = os.path.join(out_dir, "template_data", "mmcif_files", f"{next_id}.cif")
        
        if inpaint_clashes:
            template_model = detect_and_remove_clashes(template_model)
            template_sequences = [get_fastaseq(template_model, chain) for chain in template_chains_selected]
        
        io.set_structure(template_model)
        io.save(template_mmcif_path)
        fix_mmcif(template_mmcif_path, template_chains_selected, template_sequences, "2100-01-01")
        
        pdb_seqres_path = Path(out_dir, "template_data", "pdb_seqres.txt").resolve()
        write_seqres(pdb_seqres_path, template_sequences, template_chains_selected, seq_id=next_id, append=append)
        
        af_flagfile_path = Path(out_dir, "template_data", "templates.flag")
        if not af_flagfile_path.is_file():
            with open(af_flagfile_path, "w") as flagfile:
                flagfile.write(f"--template_mmcif_dir={mmcif_path.resolve()}\n")
                flagfile.write(f"--pdb_seqres_database_path={pdb_seqres_path}\n")
                if align:
                    flagfile.write("--use_precomputed_msas\n")
        
        if align:
            if len(target_chains) != len(template_chains_selected):
                raise AssertionError(
                    f"The number of target chains ({target_chains}) doesn't match the number of template chains ({template_chains_selected})."
                )
            for i, (template_chain, template_sequence, target_chain, target_sequence, target_model) in enumerate(
                zip(template_chains_selected, template_sequences, target_chains, target_sequences, target_models)
            ):
                msa_chain = ascii_upperlower[i]
                this_template_model = pickle.loads(pickle.dumps(template_model, -1))
                this_target_model = pickle.loads(pickle.dumps(target_model, -1))
                print(f"Aligning fasta sequence {i+1} (seq: {target_sequence[:10]}...) to template chain {template_chain} (seq: {template_sequence[:10]}...)")
                alignment = do_align(template_sequence, this_template_model, target_sequence, this_target_model, alignment_type="blast")
                sto_alignment = format_alignment_stockholm(alignment, hit_id=next_id, hit_chain=template_chain)
                
                msa_path = f"msas/{msa_chain}"
                Path(out_dir, msa_path).mkdir(parents=True, exist_ok=True)
                with open(Path(out_dir, msa_path, "pdb_hits.sto"), mode="a" if append else "w") as pdb_hits:
                    for line in sto_alignment:
                        pdb_hits.write(line)
        if temp_reps > 1:
            append = True
    
    # Run the AlphaFold prediction
    if msa_mode == "no_MSA":  # same as "no_MSA" on the AF_unmasked paper
        unpaired_msa = []
        query_seqs_unique = []
        for i, ts in enumerate(target_sequences):
            unpaired_msa.append(f"> seq_{i}\n{ts}")
            query_seqs_unique.append(ts)
    else:
        print("Querying ColabFold's MSA server")
        msa_lines = None
        use_templates = False
        custom_template_path = None
        pair_mode = "unpaired"
        pairing_strategy = "greedy"
        host_url = DEFAULT_API_SERVER
        version = get_commit() or ""
        user_agent = f"colabfold/{version}"
        unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features = get_msa_and_templates(
            jobname, target_sequences, msa_lines, out_dir, msa_mode, use_templates,
            custom_template_path, pair_mode, pairing_strategy, host_url, user_agent
        )
  
    # Process MSA results
    for sequence, msa in zip(query_seqs_unique, unpaired_msa):
        for chain in seq2chain[sequence]:
            msa_dir = out_dir.joinpath(f"msas/{chain}")
            msa_dir.mkdir(parents=True, exist_ok=True)
            (msa_dir / "bfd_uniref_hits.a3m").write_text(msa)
            with open(msa_dir / "uniprot_hits.a3m", "w") as pseudo_uniprot:
                pseudo_uniprot.write(f"> {chain}\n{sequence}")
            with open(msa_dir / "uniprot_hits.a3m", "r") as input_handle, open(msa_dir / "uniprot_hits.sto", "w") as output_handle:
                alignments = AlignIO.parse(input_handle, "fasta")
                AlignIO.write(alignments, output_handle, "stockholm")
    
    # Download parameters if not already available
    data_dir = Path("./")
    if not glob.glob(f"{data_dir}/params/*_finished.txt"):
        print("Downloading AF parameters...")
        download_alphafold_params(model_type, data_dir)
    
    # Setup template search and featurizer
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=shutil.which("hmmsearch"),
        hmmbuild_binary_path=shutil.which("hmmbuild"),
        database_path=out_dir.joinpath("template_data", "pdb_seqres.txt")
    )
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=mmcif_path.resolve(),
        max_template_date="2100-01-01",
        max_hits=4,
        kalign_binary_path=shutil.which("kalign"),
        release_dates_path=None,
        obsolete_pdbs_path=None
    )
    
    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=shutil.which("jackhmmer"),
        hhblits_binary_path=shutil.which("hhblits"),
        uniref90_database_path=".",
        mgnify_database_path="",
        bfd_database_path="",
        uniref30_database_path="",
        small_bfd_database_path="",
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=False,
        use_precomputed_msas=True,
        mgnify_max_hits=1,
        uniref_max_hits=1,
        bfd_max_hits=msa_depth,
        no_uniref=True,
        no_mgnify=True
    )
    
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=shutil.which("jackhmmer"),
        uniprot_database_path=None,
        use_precomputed_msas=True,
        max_uniprot_hits=1,
        separate_homomer_msas=True
    )
    
    model_names = ["model_5_multimer_v2"] if model_type == "alphafold2_multimer_v2" else ["model_5_multimer_v3"]
    model_runners = {}
    
    for model_name in model_names:
        model_config = config.model_config(model_name)
        model_config.model.num_ensemble_eval = 1
        model_config.model.embeddings_and_evoformer.cross_chain_templates = True
        model_config.model.num_recycle = num_recycles
        model_config.model.global_config.eval_dropout = use_dropout
        model_config.model.recycle_early_stop_tolerance = recycle_early_stop_tolerance
        
        model_params = data.get_model_haiku_params(model_name=model_name, data_dir=str(data_dir))
        model_runner = model.RunModel(model_config, model_params)
        for i in range(predictions_per_model):
            model_runners[f'{model_name}_pred_{i}'] = model_runner
    
    # Instantiate the Amber relaxer.
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=True)

    predict_structure(
        fasta_path=target_fasta,
        fasta_name=jobname,
        output_dir_base="./",
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        benchmark=False,
        random_seed=0,
        models_to_relax=args.models_to_relax,
        amber_relaxer=amber_relaxer
    )

if __name__ == '__main__':
    main()
