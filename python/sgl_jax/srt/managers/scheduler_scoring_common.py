import concurrent.futures as futures
from dataclasses import dataclass


@dataclass
class _LocalSchedulerRpcEnvelope:
    req_obj: object
    result_future: futures.Future | None = None
