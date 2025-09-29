`compare.py` is designed to work on a SLURM cluster.

Running `python compare.py -h` produces:

```
usage: compare.py [-h] [-w] {attack,plot,gather,config} [job_num]

Run attacks, plots, or configuration.

positional arguments:
  {attack,plot,gather,config}
                        Command to execute
  job_num               Job number (required for 'attack')

options:
  -h, --help            show this help message and exit
  -w, --weak            Use weak strength (default is strong)
```

`python measure.py config` reads in the datafiles from `../original_data_parquet`, generates a set of attacks (unknown attribute, known attributes), and places the attacks into `jobs.json`. This contains jobs for both our approach and the prior approach.

`python measure.py attack <job_num>` executes the attack with `job_num` in `jobs.json`. If the `-w` flag is included, then it runs the attack over the weakly anonymized data, otherwise it uses the strongly anonymized data. It places the results in one of four directories:

```
work_files_strong
work_files_weak
work_files_prior_strong
work_files_prior_weak
```

`python measure.py gather` gathers the results in the four directories, and places them in one of four parquet dataframes:

```
all_secret_known_strong.parquet
all_secret_known_weak.parquet
all_secret_known_prior_strong.parquet
all_secret_known_prior_weak.parquet
```

`python measure.py plot` reads in the four parquet results files, and generates the plots used in the paper, one of the tables, and text output containing other statistics which are used in the paper.

The plots used in the paper are:

* `boxplots_weak.pdf`
* `alc_ours_vs_prior_weak.pdf`
* `alc_ours_vs_prior_weak.pdf`

The table used in the paper is saved as `wrong_conclusion.tex`.

The text output is in `out.txt`. The numbers in the text output that are used in the table are tagged with the string `__use__`.


The files `slurm_script_strong` and `slurm_script_weak` contain the SLURM scripts used to run all jobs on a SLURM cluster.


