from llava.worker import Worker, WorkerConfig


if __name__ == "__main__":
    config = WorkerConfig(
        "test_worker",
        ""
    )
    worker = Worker(config, debug=False)
    worker.run_loop()