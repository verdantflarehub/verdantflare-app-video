"""Runtime errors shared by HTTP/queue code and the GPU engine."""


class RuntimeErrorCode(RuntimeError):
    def __init__(self, code: str, *, metrics: dict | None = None):
        super().__init__(code)
        self.code = code
        self.runtime_metrics = metrics
