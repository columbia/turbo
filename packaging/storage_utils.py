import typer
import redis
import psycopg2
from omegaconf import OmegaConf
from turbo.utils.utils import DEFAULT_CONFIG_FILE

app = typer.Typer()


class RedisHelper:
    def __init__(self, config) -> None:
        self.config = config
        self.kv_store = redis.Redis(host=config.host, port=config.port, db=0)

    def get(self, key):
        return self.kv_store.hgetall(key)

    def delete_all(self):
        res = "success"
        try:
            self.kv_store.flushdb()
        except Exception as error:
            res = "fail"
            print(error)
        return res

    def get_all(self):
        return self.kv_store.keys("*")


class PostgresHelper:
    def __init__(self, config, database) -> None:
        self.config = config
        self.database = database
        # Initialize the PSQL connection
        try:
            # Connect to the PostgreSQL database server
            self.psql_conn = psycopg2.connect(
                host=config.host,
                port=config.port,
                database=database,
                user=config.username,
                password=config.password,
            )
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit(1)

    def get(self, key):
        try:
            cur = self.psql_conn.cursor()
            cmd = f"""SELECT COUNT(*) FROM {self.database}_data WHERE time >= {key} AND time <= {key} GROUP BY time;"""
            cur.execute(cmd)
            res = cur.fetchall()
            cur.close()
            self.psql_conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit(1)
        return res

    def delete_all(self):
        status = "success"
        try:
            cur = self.psql_conn.cursor()
            cmd = f"""DELETE FROM {self.database}_data;"""
            cur.execute(cmd)
            cur.close()
            self.psql_conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            status = "fail"
            print(error)
            exit(1)
        return status

    def get_all(self):
        try:
            cur = self.psql_conn.cursor()
            cmd = f"""SELECT time, COUNT(*) FROM {self.database}_data GROUP BY time;"""
            cur.execute(cmd)
            res = cur.fetchall()
            cur.close()
            self.psql_conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit(1)
        return res


@app.command()
def run(
    omegaconf: str = "turbo/config/turbo.json",
    storage: str = "redis-budgets",  # "redis-budgets", 'postgres', '*'
    function: str = "delete-all",  # "get", "size", "get-all", "delete-all-caches"
    database: str = "covid",
    key: str = "",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)

    if storage == "*":
        if function == "delete-all":
            res = "success"
            try:
                PostgresHelper(config.postgres, database).delete_all()
                RedisHelper(config.cache).delete_all()
                RedisHelper(config.budget_accountant).delete_all()
            except Exception as error:
                res = "fail"
                print(error)

    elif storage == "postgres":
        postgres_helper = PostgresHelper(config.postgres, database)
        if function == "delete-all":
            res = postgres_helper.delete_all()
        elif function == "get":
            res = postgres_helper.get("key")
        elif function == "get-all" or function == "size":
            res = postgres_helper.get_all()
        # elif function == "size":
        # return len(postgres_helper.get_all())

    else:  # Redis
        if storage == "redis-budgets":
            redis_helper = RedisHelper(config.budget_accountant)
        elif storage == "redis-cache":
            redis_helper = RedisHelper(config.cache)
        if function == "delete-all":
            res = redis_helper.delete_all()
        elif function == "get":
            res = redis_helper.get(key)
        elif function == "get-all":
            res = redis_helper.get_all()
        elif function == "size":
            res = len(redis_helper.get_all())
    print(res)


if __name__ == "__main__":
    app()
