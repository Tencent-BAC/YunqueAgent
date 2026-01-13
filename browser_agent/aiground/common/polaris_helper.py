# -*- coding: utf-8 -*-
"""
polaris_helper

Polaris service discovery helper.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

try:
    from polaris.api import consumer as polaris_consumer
    from polaris.wrapper import String, UserString
except ImportError:
    from aiground.common import fake_consumer as polaris_consumer

from aiground.common.singleton import singleton

LOGGER = logging.getLogger(__name__)


@dataclass
class Node(object):
    service_name: str
    namespace: str
    instance: polaris_consumer.Instance
    container_name: str
    address: str
    ip: str
    port: int
    protocol: str
    weight: float


@singleton
class PolarisHelper(object):
    def __init__(self):
        self._consumer_api = polaris_consumer.create_consumer_by_default_config_file()
        pass

    def lookup(self, endpoint: str) -> Tuple[str, Optional[Node]]:
        if not endpoint.startswith("https+polaris://") and not endpoint.startswith(
            "http+polaris://"
        ):
            return (endpoint, None)
        req_url = urlparse(endpoint)
        scheme = req_url.scheme.split("+")[0]
        netloc = req_url.netloc.split(":")
        service_name = netloc[0]
        namespace = netloc[1]
        node: Node = self.find_one_node(namespace, service_name)
        ret_endpoint = f"{scheme}://{node.address}{req_url.path}"
        if req_url.query:
            ret_endpoint += "?" + req_url.query
        return (ret_endpoint, node)

    def find_one_node(self, namespace: str, service_name: str) -> Node:
        req = polaris_consumer.GetOneInstanceRequest(
            namespace=namespace, service=service_name
        )
        inst = self._consumer_api.get_one_instance(req)
        if inst is None:
            raise ValueError("no instance")
        host = to_str_with_default(inst.get_host())
        node = Node(
            service_name=service_name,
            namespace=namespace,
            instance=inst,
            container_name=to_str_with_default(inst.get_metadata("container_name")),
            address=f"{host}:{inst.get_port()}",
            ip=host,
            port=inst.get_port(),
            protocol=to_str_with_default(inst.get_protocol()),
            weight=inst.get_weight(),
        )
        return node

    def report(self, node: Node, cost: int, success: int):
        LOGGER.debug(f"start report for node: {node.address}")
        inst: polaris_consumer.Instance = node.instance

        service_call_result = polaris_consumer.ServiceCallResult(
            namespace=node.namespace,
            service=node.service_name,
            instance_id=inst.get_id(),
        )
        service_call_result.set_delay(cost)
        service_call_result.set_ret_code(success)
        # Notes:
        # - POLARIS_CALL_RET_OK means success. It's recommended to report business errors as RPC success
        #   to avoid unintended instance removal.
        # - POLARIS_CALL_RET_ERROR means transport/system failure and can trigger circuit breaking.
        if success >= 0:
            ret_status = polaris_consumer.wrapper.POLARIS_CALL_RET_OK
        else:
            ret_status = polaris_consumer.wrapper.POLARIS_CALL_RET_ERROR
        service_call_result.set_ret_status(ret_status)
        self._consumer_api.update_service_call_result(service_call_result)


def to_str_with_default(val, default_value: str = "") -> str:
    """Convert Polaris string wrapper types into Python strings with a default."""
    if val is None:
        return default_value
    if isinstance(val, (String, UserString)):
        if val.data is None:
            return default_value
        if isinstance(val.data, bytes):
            return val.data.decode("utf-8")
    return str(val)
