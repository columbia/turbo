import typer
from omegaconf import OmegaConf
from turbo.client_tasks import TasksClient
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

    query_vector = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 3],
        [0, 0, 0, 4],
        [0, 0, 0, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 7],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 1, 3],
        [0, 0, 1, 4],
        [0, 0, 1, 5],
        [0, 0, 1, 6],
        [0, 0, 1, 7],
        [0, 0, 2, 0],
        [0, 0, 2, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 3],
        [0, 0, 2, 4],
        [0, 0, 2, 5],
        [0, 0, 2, 6],
        [0, 0, 2, 7],
        [0, 0, 3, 0],
        [0, 0, 3, 1],
        [0, 0, 3, 2],
        [0, 0, 3, 3],
        [0, 0, 3, 4],
        [0, 0, 3, 5],
        [0, 0, 3, 6],
        [0, 0, 3, 7],
    ]
    task = {
        "query_id": 0,
        "query": query_vector,
        "nblocks": 1,
        "utility": 100,
        "utility_beta": 0.0001,
    }

    TasksClient(config.tasks_server).send_request(task)


if __name__ == "__main__":
    app()
