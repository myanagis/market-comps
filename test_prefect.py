from prefect import flow, task

@task
def hello_task():
    return "Hello"

@flow
def hello_flow():
    return hello_task()

if __name__ == "__main__":
    print(hello_flow())
