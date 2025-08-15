"""
TITLE

Description

Created on 13.08.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###             0. CLI              ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import click
import os
from neural_network import pretrain, train

import logging
import shlex
from datetime import datetime
from pathlib import Path

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def _setup_logger(scenario: str):
    """Create Logs/<scenario>_<timestamp>.log and return (logger, logfile_path)."""
    logs_dir = Path(__file__).resolve().parents[2] / "Logs"   # -> /.../src/Logs
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = logs_dir / f"{scenario}_{ts}.log"

    logger = logging.getLogger(f"online_learning.{scenario}.{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, logfile


def _run(cmd: str, logger: logging.Logger, logfile: Path) -> int:
    """Run a shell command, append stdout/stderr to logfile, and log the command."""
    logger.info(f"$ {cmd}")
    qlog = shlex.quote(str(logfile))
    rc = os.system(f"{cmd} >> {qlog} 2>&1")
    if rc != 0:
        logger.warning(f"Command returned non-zero exit code {rc}: {cmd}")
    return rc



@click.command()
@click.argument('scenario', type=str, required=True)
@click.option('--dark-style', is_flag=True, required=False, help='Dark plot style')
@click.option('--plot-ext', type=str, default='.pdf', required=False, help='The extension of the plot, defaults to ".pdf"')

# ---- new tuning options ----
@click.option('--pretrain-epochs', type=int, default=5, show_default=True,
              help='Number of epochs for initial pretraining')
@click.option('--update-epochs', type=int, default=1, show_default=True,
              help='Epochs used during each training update step')
@click.option('--days', type=int, default=1, show_default=True,
              help='Length of scenario blocks in days')
@click.option('--n-scenarios', type=int, default=2, show_default=True,
              help='How many scenarios to generate each round')
@click.option('--latent-dim', type=int, default=64, show_default=True,
              help='Latent dimension for the scenario generator')
@click.option('--seed', type=int, default=42, show_default=True,
              help='Random seed')
@click.option('--batch-size', type=int, default=256, show_default=True,
              help='Batch size used for pretraining / generation')
@click.option('--learning-rate', type=float, default=5e-4, show_default=True,
              help='Learning rate for the scenario generator (pretrain/update)')

@click.pass_context
def CLI(ctx, scenario: str, dark_style: bool, plot_ext: str, pretrain_epochs: int, update_epochs: int, days: int, n_scenarios: int,
        latent_dim: int, seed: int, batch_size: int, learning_rate: float):
    """
    Description of the CLI
    Online learning driver with logging to src/Logs/<scenario>_<timestamp>.log
    
    """
    
    # Set global style of plot
    if dark_style:
        plt.style.use('dark_background')
        fc = 'none'
    else:
        fc = 'white'

    # Store global options in the context object
    ctx.ensure_object(dict)
    ctx.obj['fc'] = fc
    ctx.obj['plot_ext'] = plot_ext
    
    # logging
    logger, logfile = _setup_logger(scenario)
    logger.info("=== Online learning run started ===")
    logger.info(f"scenario={scenario} pretrain_epochs={pretrain_epochs} update_epochs={update_epochs} "
                f"days={days} n_scenarios={n_scenarios} latent_dim={latent_dim} seed={seed} "
                f"batch_size={batch_size} learning_rate={learning_rate}")
    logger.info(f"Logfile: {logfile}")

    # pretraining + initial scenario generation
    logger.info("Starting pretraining...")
    
    epoch = 0    
    # days = 1
    # n_scenarios = 2
    model = pretrain(pretrain_epochs, days=days, n_scenarios=n_scenarios, latent_dim=latent_dim, batch_size=batch_size, learning_rate=learning_rate, seed=seed, logger=logger)
    
    logs_dir = logfile.parent
    ckpt_path = logs_dir / f"{scenario}_model_checkpoint.pth"

    
    
    os.chdir('Balmorel')

    while epoch < update_epochs:
        for runtype in ['capacity', 'dispatch']:
            
            os.system('rm operun/data/*.inc')
            
            if runtype == 'capacity':
                os.system('cp -f operun/capexp_data/*.inc operun/data')
            else:
                os.system('cp -f operun/S.inc operun/data')
                os.system(f'cp -f operun/T_{runtype}.inc operun/data/T.inc')
                os.system(f'mv operun/model/balopt_{runtype}.opt operun/model/balopt.opt')
                
            os.chdir('operun/model')
            os.system(f'gams Balmorel --scenario_name "{scenario}_{runtype}_E{epoch}"')

            os.chdir('../../')

            # Copy the simex folder
            # cp simex -r simex_$name

            if runtype != "capacity":
                # Rename balopt back
                os.system(f'mv "operun/model/balopt.opt" "operun/model/balopt_{runtype}.opt"')
        
        os.system(f'pixi run python analysis/analyse.py adequacy "{scenario}_dispatch" {epoch}')
        os.chdir('../')
        model = train(model, f"{scenario}_dispatch", epoch, n_scenarios=n_scenarios, batch_size=batch_size, logger=logger)
        os.chdir('Balmorel')
        
        # save the model
        model.save_model(str(ckpt_path))
        logger.info(f"Epoch {epoch} completed and model saved.")
        
        epoch += 1


#%% ------------------------------- ###
###            2. Utils             ###
### ------------------------------- ###

@click.pass_context
def plot_style(ctx, fig, ax, name: str, legend: bool = True):
    
    ax.set_facecolor(ctx.obj['fc'])
    
    if legend:
        ax.legend(loc='center', bbox_to_anchor=(.5, 1.15), ncol=3)
    
    fig.savefig(name + ctx.obj['plot_ext'], bbox_inches='tight', transparent=True)
    
    return fig, ax

#%% ------------------------------- ###
###             3. Main             ###
### ------------------------------- ###
if __name__ == '__main__':
    CLI()
