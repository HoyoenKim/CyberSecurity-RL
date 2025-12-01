# CyberBattleSim


### On Linux or WSL

The instructions were tested on a Linux Ubuntu distribution (both native and via WSL).

If conda is not installed already, you need to install it by running the `install_conda.sh` script.

```bash
bash install-conda.sh
```

Once this is done, open a new terminal and run the initialization script:
```bash
bash init.sh
```
This will create a conda environmen named `cybersim` with all the required OS and python dependencies.

To activate the environment run:

```bash
conda activate cybersim
```

#### Windows Subsystem for Linux

The supported dev environment on Windows is via WSL.
You first need to install an Ubuntu WSL distribution on your Windows machine,
and then proceed with the Linux instructions (next section).

#### Git authentication from WSL

To authenticate with Git, you can either use SSH-based authentication or
alternatively use the credential-helper trick to automatically generate a
PAT token. The latter can be done by running the following command under WSL
([more info here](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-git)):

```ps
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-manager.exe"
```

#### Docker on WSL

To run your environment within a docker container, we recommend running `docker` via Windows Subsystem on Linux (WSL) using the following instructions:
[Installing Docker on Windows under WSL](https://docs.docker.com/docker-for-windows/wsl-tech-preview/)).

### Windows (unsupported)

This method is not maintained anymore, please prefer instead running under
a WSL subsystem Linux environment.
But if you insist you want to start by installing [Python 3.10](https://www.python.org/downloads/windows/) then in a Powershell prompt run the `./init.ps1` script.

## Getting started quickly using Docker

The quickest method to get up and running is via the Docker container.

> NOTE: For licensing reasons, we do not publicly redistribute any
> build artifact. In particular, the docker registry `spinshot.azurecr.io` referred to
> in the commands below is kept private to the
> project maintainers only.
>
> As a workaround, you can recreate the docker image yourself using the provided `Dockerfile`, publish the resulting image to your own docker registry and replace the registry name in the commands below.

### Running from Docker registry

```bash
commit=7c1f8c80bc53353937e3c69b0f5f799ebb2b03ee
docker login spinshot.azurecr.io
docker pull spinshot.azurecr.io/cyberbattle:$commit
docker run -it spinshot.azurecr.io/cyberbattle:$commit python -m cyberbattle.agents.baseline.run
```

### Recreating the Docker image

```bash
docker build -t cyberbattle:1.1 .
docker run -it -v "$(pwd)":/source --rm cyberbattle:1.1 python -m cyberbattle.agents.baseline.run
```

## Check your environment

Run the following commands to run a simulation with a baseline RL agent:

```bash
python -m cyberbattle.agents.baseline.run --training_episode_count 5 --eval_episode_count 3 --iteration_count 100 --rewardplot_width 80  --chain_size=4 --ownership_goal 0.2

python -m cyberbattle.agents.baseline.run --training_episode_count 5 --eval_episode_count 3 --iteration_count 100 --rewardplot_width 80  --chain_size=4 --reward_goal 50 --ownership_goal 0
```

If everything is setup correctly you should get an output that looks like this:

```bash
torch cuda available=True
###### DQL
Learning with: episode_count=1,iteration_count=10,ϵ=0.9,ϵ_min=0.1, ϵ_expdecay=5000,γ=0.015, lr=0.01, replaymemory=10000,
batch=512, target_update=10
  ## Episode: 1/1 'DQL' ϵ=0.9000, γ=0.015, lr=0.01, replaymemory=10000,
batch=512, target_update=10
Episode 1|Iteration 10|reward:  139.0|Elapsed Time: 0:00:00|###################################################################|
###### Random search
Learning with: episode_count=1,iteration_count=10,ϵ=1.0,ϵ_min=0.0,
  ## Episode: 1/1 'Random search' ϵ=1.0000,
Episode 1|Iteration 10|reward:  194.0|Elapsed Time: 0:00:00|###################################################################|
simulation ended
Episode duration -- DQN=Red, Random=Green
   10.00  ┼
Cumulative rewards -- DQN=Red, Random=Green
  194.00  ┼      ╭──╴
  174.60  ┤      │
  155.20  ┤╭─────╯
  135.80  ┤│     ╭──╴
  116.40  ┤│     │
   97.00  ┤│    ╭╯
   77.60  ┤│    │
   58.20  ┤╯ ╭──╯
   38.80  ┤  │
   19.40  ┤  │
    0.00  ┼──╯
```

## Jupyter notebooks

To quickly get familiar with the project, you can open one of the provided Jupyter notebooks to play interactively with the gymnasium environments.

> Notes on the `.py` notebooks:
> - Our notebooks are checked-in in Git as `.py` files. Those can be opened and run directly  in VSCode or in Jupyter using the [Jupytext extension](https://jupytext.readthedocs.io/en/latest/install.html).
> - The output `.ipynb` files can also be automatically regenerated using [papermill](https://pypi.org/project/papermill/) by running the bash script [run_benchmark.sh](/notebooks/run_benchmark.sh).
> - We also publish a snapshot of the corresponding `.ipynb`-notebooks with the entire output and plots in a separate git tag.
The latest snapshot of the Jupyter notebooks output, including the benchmarks, are
accessible from the following git tag: [/notebooks/benchmarks (latest_benchmark)](https://github.com/microsoft/CyberBattleSim/tree/latest_benchmark/notebooks/benchmarks).


Some notebooks to get started:

- 'Capture The Flag' toy environment notebooks:
  - [Random agent](notebooks/toyctf-random.py)
  - [Interactive session for a human player](notebooks/toyctf-blank.py)
  - [Interactive session - fully solved](notebooks/toyctf-solved.py)

- Chain environment notebooks:
  - [Random agent](notebooks/chainnetwork-random.py)

- Other environments:
  - [Interactive session with a randomly generated environment](notebooks/randomnetwork.py)
  - [Random agent playing on randomly generated networks](notebooks/c2_interactive_interface.py)

- Benchmarks:   The following notebooks show benchmark evaluation of the baseline agents on various environments.

    - [Benchmarking on a given environment](notebooks/notebook_benchmark.py)
    - [Benchmark on chain environments with a basic defender](notebooks/notebook_withdefender.py)
    - [DQL transfer learning evaluation](notebooks/notebook_dql_transfer.py)
    - [Epsilon greedy with credential lookups](notebooks/notebook_randlookups.py)
    - [Tabular Q Learning](notebooks/notebook_tabularq.py)

The latest snapshot of the Jupyter notebooks output, including the benchmarks, are
accessible from the following git tag: [/notebooks/benchmarks (latest_benchmark)](https://github.com/microsoft/CyberBattleSim/tree/latest_benchmark/notebooks/benchmarks).


## How to instantiate the Gym environments?

The following code shows how to create an instance of the OpenAI Gym environment `CyberBattleChain-v0`, an environment based on a [chain-like network structure](cyberbattle/samples/chainpattern/chainpattern.py), with 10 nodes (`size=10`) where the agent's goal is to either gain full ownership of the network (`own_atleast_percent=1.0`) or
break the 80% network availability SLA (`maintain_sla=0.80`), while the network is being monitored and protected by the basic probalistically-modelled defender (`defender_agent=ScanAndReimageCompromisedMachines`):

```python
import cyberbattle._env.cyberbattle_env

cyberbattlechain_defender =
  gym.make('CyberBattleChain-v0',
      size=10,
      attacker_goal=AttackerGoal(
          own_atleast=0,
          own_atleast_percent=1.0
      ),
      defender_constraint=DefenderConstraint(
          maintain_sla=0.80
      ),
      defender_agent=ScanAndReimageCompromisedMachines(
          probability=0.6,
          scan_capacity=2,
          scan_frequency=5))
```
