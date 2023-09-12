import typer
from omegaconf import OmegaConf
from turbo.budget_accountant import BudgetAccountant
from turbo.utils.utils import DEFAULT_CONFIG_FILE

app = typer.Typer()


@app.command()
def run(
    omegaconf: str = "turbo/config/turbo.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)

    budget_accountant = BudgetAccountant(config=config.budget_accountant)
    budget_accountant.add_new_block_budget()


if __name__ == "__main__":
    app()
