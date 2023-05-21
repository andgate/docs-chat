from invoke import runners, task

from frontend.app import start_gradio


@task
def start_api(c):
    c.run("uvicorn server.server:app --reload --reload-dir server --reload-dir utils")


@task
def start_frontend(c):
    start_gradio()


@task
def start(c):
    start_api(c)
    # start_frontend(c)
